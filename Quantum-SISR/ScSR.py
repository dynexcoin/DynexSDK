"""
Dynex SDK (beta) Neuromorphic Computing Library
Copyright (c) 2021-2023, Dynex Developers

All rights reserved.

1. Redistributions of source code must retain the above copyright notice, this list of
    conditions and the following disclaimer.
 
2. Redistributions in binary form must reproduce the above copyright notice, this list
   of conditions and the following disclaimer in the documentation and/or other
   materials provided with the distribution.
 
3. Neither the name of the copyright holder nor the names of its contributors may be
   used to endorse or promote products derived from this software without specific
   prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Some parts of the code is based on the following research paper:

"Quantum Annealing for Single Image Super-Resolution" by Han Yao Choong, 
Suryansh Kumar and Luc Van Gool (ETH Zürich)

"""

import numpy as np 
from os import listdir
import os
import time
import pickle
import logging
from tqdm import tqdm
from scipy.signal import convolve2d
from sklearn.preprocessing import normalize
from sklearn import linear_model
from skimage.io import imread
from skimage.color import rgb2ycbcr
from skimage.transform import resize
import dimod
import dynex
import qubo_algorithms
from config_run import config

def extract_lr_feat(img_lr):
    h, w = img_lr.shape
    img_lr_feat = np.zeros((h, w, 4))

    # First order gradient filters
    hf1 = [[-1, 0, 1], ] * 3
    vf1 = np.transpose(hf1)

    img_lr_feat[:, :, 0] = convolve2d(img_lr, hf1, 'same')
    img_lr_feat[:, :, 1] = convolve2d(img_lr, vf1, 'same')

    # Second order gradient filters
    hf2 = [[1, 0, -2, 0, 1], ] * 3
    vf2 = np.transpose(hf2)

    img_lr_feat[:, :, 2] = convolve2d(img_lr, hf2, 'same')
    img_lr_feat[:, :, 3] = convolve2d(img_lr, vf2, 'same')

    return img_lr_feat


def create_list_step(start, stop, step):
    list_step = []
    for i in range(start, stop, step):
        list_step = np.append(list_step, i)
    return list_step


def lin_scale(xh, us_norm):
    hr_norm = np.sqrt(np.sum(np.multiply(xh, xh)))

    if hr_norm > 0:
        lin_scale_factor = 1.2
        s = us_norm * lin_scale_factor / hr_norm
        xh = np.multiply(xh, s)
    return xh


def ScSR(img_lr_y, size, upscale, Dh, Dl, lmbd, overlap, quantum_objects):
    patch_size = config.patch_size

    img_us = resize(img_lr_y, size)
    img_us_height, img_us_width = img_us.shape
    img_hr = np.zeros(img_us.shape)
    img_hr_entropy = np.zeros(img_us.shape)
    cnt_matrix = np.zeros(img_us.shape)

    img_lr_y_feat = extract_lr_feat(img_us)

    gridx = np.append(create_list_step(0, img_us_width - patch_size - 1, patch_size - overlap), img_us_width - patch_size - 1)
    gridy = np.append(create_list_step(0, img_us_height - patch_size - 1, patch_size - overlap), img_us_height - patch_size - 1)

    count = 0
    cardinality = np.zeros(len(gridx)*len(gridy))

    # Neuromorphic sampling on Dynex
    #
    # num_reads and annealing_time have been hard coded. Applicable values depend on image size and desired output quality
    #
    # Note: 
    # This is an example implementation to showcase the core functionality and efficiacy of the
    # implemented algorithm. In real world applications, all QUBOs would be sampled in 1 batch 
    # in parallel on the platform, reducing sampling time to 1 run of a few seconds.
            
    if config.sc_algo=="dynex": 
        logging.info("running dynex in ScSR")

        logging.info("Note: This is an example implementation to showcase the core functionality and efficiacy of the implemented algorithm. In real world applications, all QUBOs would be sampled in 1 batch in parallel on the platform, reducing sampling time to 1 run of a few seconds.");

        n_patches_per_qubo = 8
        qubo_size = n_patches_per_qubo*Dl.shape[1]
        
        create_qubo_start_time = time.time()
        Q_dicts = qubo_algorithms.create_qubo1(img_lr_y, size, Dl, overlap, n_patches_per_qubo)
        logging.info("Create QUBO time: "+str(time.time()-create_qubo_start_time))

        flattened_m = np.zeros(qubo_size*len(Q_dicts))
        logging.info("Number of QUBO problems to solve: %d"%(len(Q_dicts)))

        total_qpu_access_time = 0
        solve_qubo_start_time = time.time()
        for i in range(len(Q_dicts)):
            logging.info("i=%d"%i)
            # Sampling the QUBO formulation on the Dynex platform
            bqm = dimod.BinaryQuadraticModel.from_qubo(Q_dicts[i], 0.0);
            model = dynex.BQM(bqm, logging = False);
            sampler = dynex.DynexSampler(model,  mainnet=False, logging = False, description='Dynex SDK test');
            sampleset = sampler.sample(num_reads=100, annealing_time = 200, debugging=False);
            print('      sampling result:',i+1,'/',len(Q_dicts), ' ground state = ', sampleset.first.energy);
            flattened_m[i*qubo_size:i*qubo_size+len(list(sampleset.first.sample.values()))] = list(sampleset.first.sample.values())
            
        logging.info("Solve QUBO time: "+str(time.time()-solve_qubo_start_time))

    # Quantum Annealing (Hybrid Solvers), submit to DWave hybrid solver
    if config.sc_algo=="qubo_bsc_dwave1": 
        logging.info("running qubo_bsc_dwave1 in ScSR")
        n_patches_per_qubo = 8
        qubo_size = n_patches_per_qubo*Dl.shape[1]
        client = Client.from_config(config_file='dwave.conf')
        solver = client.get_solver(name='hybrid_binary_quadratic_model_version2')

        create_qubo_start_time = time.time()
        Q_dicts = qubo_algorithms.create_qubo1(img_lr_y, size, Dl, overlap, n_patches_per_qubo)
        logging.info("Create QUBO time: "+str(time.time()-create_qubo_start_time))

        flattened_m = np.zeros(qubo_size*len(Q_dicts))
        logging.info("Number of QUBO problems to solve: %d"%(len(Q_dicts)))

        total_qpu_access_time = 0
        solve_qubo_start_time = time.time()
        for i in range(len(Q_dicts)):
            logging.info("i=%d"%i)

            #dwave_real_run = False
            dwave_real_run = True
            if dwave_real_run:
                computation = solver.sample_qubo(Q_dicts[i],time_limit=3.5)
                total_qpu_access_time += computation.sampleset.info['qpu_access_time']
                flattened_m[i*qubo_size:i*qubo_size+len(computation.samples[0])] = computation.samples[0]
        logging.info("total_qpu_access_time="+str(total_qpu_access_time))
        logging.info("Solve QUBO time: "+str(time.time()-solve_qubo_start_time))

    # Quantum Annealing (Direct Solvers), submit to Dwave pure quantum solver
    elif config.sc_algo=="qubo_bsc_dwave2": 
        logging.info("running qubo_bsc_dwave2 in ScSR")
        qubo_size = config.qubo_size
        fec_sampler = quantum_objects["fec_sampler"]

        create_qubo_start_time = time.time()
        Q_dicts,sp_map_index,flattened_m = qubo_algorithms.create_qubo2(img_lr_y, size, Dl, overlap)
        logging.info("Create QUBO time: "+str(time.time()-create_qubo_start_time))

        dwave_samplesets = []
        logging.info("Number of QUBO problems to solve: %d"%(len(Q_dicts)))

        total_qpu_access_time = 0
        solve_qubo_start_time = time.time()
        for i in range(len(Q_dicts)):
            logging.info("i=%d"%i)

            #dwave_real_run = False
            dwave_real_run = True
            if dwave_real_run:
                sampleset = fec_sampler.sample_qubo(Q_dicts[i],num_reads=config.num_reads)
                total_qpu_access_time += sampleset.info['timing']['qpu_access_time']
            else:
                Q_tmp = np.random.randint(2,size=config.qubo_size,dtype=np.int8)
                Q_tmp_dict = {i:Q_tmp[i] for i in range(config.qubo_size)}
                sampleset = dimod.SampleSet.from_samples(dimod.as_samples(Q_tmp_dict), 'BINARY', 0)
            dwave_samplesets.append(sampleset)
            logging.info(str(sampleset.info))
            logging.info("sampleset size="+str(len(sampleset)))
            
        logging.info("total_qpu_access_time="+str(total_qpu_access_time))
        logging.info("Solve QUBO time: "+str(time.time()-solve_qubo_start_time))
        
    total_opt_time = 0
    for m in tqdm(range(0, len(gridx))):
        for n in range(0, len(gridy)):
            xx = int(gridx[m])
            yy = int(gridy[n])

            us_patch = img_us[yy : yy + patch_size, xx : xx + patch_size]
            us_mean = np.mean(np.ravel(us_patch, order='F'))
            us_patch = np.ravel(us_patch, order='F') - us_mean
            us_norm = np.sqrt(np.sum(np.multiply(us_patch, us_patch)))

            feat_patch = img_lr_y_feat[yy : yy + patch_size, xx : xx + patch_size, :]
            feat_patch = np.ravel(feat_patch, order='F')
            feat_norm = np.sqrt(np.sum(np.multiply(feat_patch, feat_patch)))

            if feat_norm > 1:
                y = np.divide(feat_patch, feat_norm)
            else:
                y = feat_patch

            # Lasso Regression:
            if config.sc_algo=="sklearn_lasso": 
                opt_time_start = time.time()
                reg = linear_model.Lasso(alpha=config.lasso_alpha,max_iter=10000)
                reg.fit(Dl,y)
                w = reg.coef_
                total_opt_time += time.time()-opt_time_start
            
            # Dynex neuromorphic sampling:
            elif config.sc_algo=="dynex": 
                w = flattened_m[count*Dl.shape[1]:(count+1)*Dl.shape[1]]*config.bsc_mu
            
            # Dynex neuromorphic sampling, bsc substitution method:
            elif config.sc_algo=="dynex_bsc": 
                opt_time_start = time.time()
                w = qubo_algorithms.qubo_dynex(Dl,y,alpha=config.bsc_alpha,mu=config.bsc_mu)
                total_opt_time += time.time()-opt_time_start
            
            # Classical Annealing:
            elif config.sc_algo=="qubo_bsc": 
                opt_time_start = time.time()
                w = qubo_algorithms.qubo_bsc(Dl,y,alpha=config.bsc_alpha,mu=config.bsc_mu)
                total_opt_time += time.time()-opt_time_start
            
            # Quantum Annealing (Hybrid Solvers):
            elif config.sc_algo=="qubo_bsc_dwave1": 
                w = flattened_m[count*Dl.shape[1]:(count+1)*Dl.shape[1]]*config.bsc_mu

            # Quantum Annealing (Direct Solvers):
            elif config.sc_algo=="qubo_bsc_dwave2": 
                w = flattened_m[count*Dl.shape[1]:(count+1)*Dl.shape[1]]*config.bsc_mu
                subproblems_per_qubo = int(qubo_size/config.subproblem_size)
                patch_samples = np.zeros((config.num_passes,config.num_reads,config.subproblem_size))
                patch_energies = np.zeros((config.num_passes,config.num_reads))
                patch_occurrences = np.zeros((config.num_passes,config.num_reads))
                for i in range(config.num_passes):
                    sampleset_no = int(count/subproblems_per_qubo)*config.num_passes+i
                    offset = (count%subproblems_per_qubo)*config.subproblem_size
                    sampleset = dwave_samplesets[sampleset_no]
                    for j in range(len(sampleset.record)):
                        patch_samples[i][j] = sampleset.record[j][0][offset:offset+config.subproblem_size]
                        patch_energies[i][j] = sampleset.record[j][1]
                        patch_occurrences[i][j] = sampleset.record[j][2]

            cardinality[count] = np.matmul(np.where(np.abs(w)>0,1,0),np.ones(len(w)))

            gibbs_entropy = 0
            if config.sc_algo=="qubo_bsc_dwave2":
                hr_patch = np.zeros(Dh.shape[0])
                p = np.zeros(config.num_reads*config.num_passes)
                Z = 0 #partition function
                min_patch_energy = 1e9
                max_patch_energy = -1e9
                for j in range(config.num_reads):
                    for k in range(config.num_passes):
                        w_tmp = w.copy()
                        w_tmp[sp_map_index[count][k]] = patch_samples[k][j]*config.bsc_mu
                        pZ = np.exp(-config.beta*patch_energies[k][j])*patch_occurrences[k][j]
                        p[j*config.num_passes+k] = pZ
                        Z += pZ
                        hr_patch += pZ*np.dot(Dh, w_tmp)
                        if patch_energies[k][j]<min_patch_energy:
                            min_patch_energy = patch_energies[k][j]
                        if patch_energies[k][j]>max_patch_energy:
                            max_patch_energy = patch_energies[k][j]
                hr_patch = hr_patch/Z
                p = p/Z
                gibbs_entropy = -np.sum(p*np.log(p+1e-9))
            else:
                hr_patch = np.dot(Dh, w)
            
            hr_patch = lin_scale(hr_patch, us_norm)
            hr_patch = np.reshape(hr_patch, (patch_size, -1))
            hr_patch += us_mean

            img_hr[yy : yy + patch_size, xx : xx + patch_size] += hr_patch
            img_hr_entropy[yy : yy + patch_size, xx : xx + patch_size] += gibbs_entropy
            cnt_matrix[yy : yy + patch_size, xx : xx + patch_size] = cnt_matrix[yy : yy + patch_size, xx : xx + patch_size] + 1.0

            count += 1
    
    logging.info("total_opt_time="+str(total_opt_time))

    index_y,index_x = np.where(cnt_matrix < 1)
    assert len(index_y)==len(index_x)
    for i in range(len(index_y)):
        yy = index_y[i]
        xx = index_x[i]
        img_hr[yy][xx] = img_us[yy][xx]
        cnt_matrix[yy][xx] = 1.0

    img_hr = np.divide(img_hr, cnt_matrix)
    img_hr_entropy = np.divide(img_hr_entropy, cnt_matrix)

    logging.info("avg_cardinality="+str(np.mean(cardinality)))

    return img_hr,img_hr_entropy
