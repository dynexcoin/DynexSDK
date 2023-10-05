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
import time
import logging
from skimage.transform import resize
from scipy.signal import convolve2d
import dimod
from dimod import BinaryQuadraticModel, SimulatedAnnealingSampler, BINARY
import dynex
from config_run import config
from qubovert.sim import anneal_qubo
from qubovert import boolean_var

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


########################################################

def update_m(X,y,w,alpha):
    D = X.shape[1]

    m = {i: boolean_var('m(%d)' % i) for i in range(D)}

    A = np.linalg.multi_dot([np.diag(w),X.T,X,np.diag(w)])
    b = -2*np.linalg.multi_dot([np.diag(w),X.T,y])
    b = b + alpha*w*np.sign(w)

    model = 0
    for i in range(D):
        for j in range(D):
            model += m[i]*(A[i][j]+1e-9)*m[j]
        model += (b[i]+1e-9)*m[i]
        
    time_start = time.time()

    res = anneal_qubo(model, num_anneals=1) #num_anneals tunable
    #res = anneal_qubo(model, num_anneals=10)

    model_solution = res.best.state
    
    m = np.array(list(model_solution.values()))
    return(m)


def qubo_bsc(X,y,alpha,mu): #binary sparse coding
    #mu determines strength of individual state
    D = X.shape[1]
    w = np.ones(D)*mu
    m = update_m(X,y,w,alpha)
    return(m*w)

########################################################

def qubo_dynex(X,y,alpha,mu): #binary sparse coding
    #mu determines strength of individual state
    D = X.shape[1]
    w = np.ones(D)*mu
    m = update_m_dynex(X,y,w,alpha)
    return(m*w)

def update_m_dynex(X,y,w,alpha):

    D = X.shape[1]
    m = {i: boolean_var('m(%d)' % i) for i in range(D)}

    # DNX
    bqm = BinaryQuadraticModel.empty(vartype=BINARY) 
    H_vars = [];
    for i in range(D):
        H_vars.append('DNX'+str(i));

    A = np.linalg.multi_dot([np.diag(w),X.T,X,np.diag(w)])
    b = -2*np.linalg.multi_dot([np.diag(w),X.T,y])
    b = b + alpha*w*np.sign(w)

    model = 0
    for i in range(D):
        for j in range(D):
            model += m[i]*(A[i][j]+1e-9)*m[j]
            if H_vars[i] == H_vars[j]:
                bqm.add_linear(H_vars[i], (A[i][j]+1e-9));
            else:
                bqm.add_quadratic(H_vars[i], H_vars[j], (A[i][j]+1e-9))
        model += (b[i]+1e-9)*m[i]
        bqm.add_linear(H_vars[i], (b[i]+1e-9))

    time_start = time.time()
    dnxmodel = dynex.BQM(bqm, logging=False);
    dnxsampler = dynex.DynexSampler(dnxmodel, logging = False, mainnet = False, description = 'Dynex SDK Test');
    sampleset = dnxsampler.sample(num_reads = 10000, annealing_time = 300);
    model_solution = sampleset.first.sample;

    m = np.array(list(model_solution.values()))
    return(m)

########################################################

def create_qubo1(img_lr_y, size, Dl, overlap, n_patches_per_qubo):
    mu = config.bsc_mu
    alpha = config.bsc_alpha
    D = Dl.shape[1]
    qubo_size = n_patches_per_qubo*D
    X = Dl
    w = np.ones(D)*mu
    patch_size = config.patch_size

    img_us = resize(img_lr_y, size)
    img_us_height, img_us_width = img_us.shape
    img_lr_y_feat = extract_lr_feat(img_us)

    gridx = np.append(create_list_step(0, img_us_width - patch_size - 1, patch_size - overlap), img_us_width - patch_size - 1)
    gridy = np.append(create_list_step(0, img_us_height - patch_size - 1, patch_size - overlap), img_us_height - patch_size - 1)

    Q_dicts = [{} for i in range(int(len(gridx)*len(gridy)/n_patches_per_qubo)+1)]

    count = 0
    for m in range(0, len(gridx)):
        for n in range(0, len(gridy)):
            xx = int(gridx[m])
            yy = int(gridy[n])

            feat_patch = img_lr_y_feat[yy : yy + patch_size, xx : xx + patch_size, :]
            feat_patch = np.ravel(feat_patch, order='F')
            feat_norm = np.sqrt(np.sum(np.multiply(feat_patch, feat_patch)))

            if feat_norm > 1:
                y = np.divide(feat_patch, feat_norm)
            else:
                y = feat_patch

            A = np.linalg.multi_dot([np.diag(w),X.T,X,np.diag(w)])
            b = -2*np.linalg.multi_dot([np.diag(w),X.T,y])
            b = b + alpha*w*np.sign(w)

            i = count%n_patches_per_qubo
            dict_index = int(count/n_patches_per_qubo)
            for j in range(0,D):
                Q_dicts[dict_index][(i*D+j,i*D+j)] = A[j][j]+b[j]+1e-9
                for k in range(j+1,D):
                    Q_dicts[dict_index][(i*D+j,i*D+k)] = 2*A[j][k]+1e-9 #*2 because A is symmetric
            count += 1
    return(Q_dicts)


def create_qubo2(img_lr_y, size, Dl, overlap):
    qubo_size = config.qubo_size
    subproblem_size = 32
    num_passes = config.num_passes
    subproblems_per_qubo = int(qubo_size/subproblem_size)

    mu = config.bsc_mu
    alpha = config.bsc_alpha
    D = Dl.shape[1]
    X = Dl
    w = np.ones(D)*mu
    patch_size = config.patch_size

    img_us = resize(img_lr_y, size)
    img_us_height, img_us_width = img_us.shape
    img_lr_y_feat = extract_lr_feat(img_us)

    gridx = np.append(create_list_step(0, img_us_width - patch_size - 1, patch_size - overlap), img_us_width - patch_size - 1)
    gridy = np.append(create_list_step(0, img_us_height - patch_size - 1, patch_size - overlap), img_us_height - patch_size - 1)

    n_qubo = int(len(gridx)*len(gridy)/(subproblems_per_qubo/num_passes))+1
    logging.info("create_qubo2, n_qubo = "+str(n_qubo))
    Q_dicts = [{} for i in range(n_qubo)]
    index = np.zeros((len(gridx)*len(gridy),num_passes,subproblem_size),dtype=np.int32)
    flattened_m = np.zeros(len(gridx)*len(gridy)*D)

    count = 0
    for m in range(0, len(gridx)):
        logging.info("create qubo m="+str(m))
        for n in range(0, len(gridy)):
            xx = int(gridx[m])
            yy = int(gridy[n])

            feat_patch = img_lr_y_feat[yy : yy + patch_size, xx : xx + patch_size, :]
            feat_patch = np.ravel(feat_patch, order='F')
            feat_norm = np.sqrt(np.sum(np.multiply(feat_patch, feat_patch)))

            if feat_norm > 1:
                y = np.divide(feat_patch, feat_norm)
            else:
                y = feat_patch

            A = np.linalg.multi_dot([np.diag(w),X.T,X,np.diag(w)])
            b = -2*np.linalg.multi_dot([np.diag(w),X.T,y])
            b = b + alpha*w*np.sign(w)
            
            Q_dict_patch = {}
            for j in range(0,D):
                Q_dict_patch[(j,j)] = A[j][j]+b[j]+1e-9
                for k in range(j+1,D):
                    Q_dict_patch[(j,k)] = 2*A[j][k]+1e-9 #*2 because A is symmetric
                    
            # SimulatedAnnealingSampler:
            simulated_annealing_parameters = {
                'beta_range': [0.1, 1.0],
                'num_reads': 10,
                'num_sweeps': 10
            }
            print('Invoking SA...');
            sampler = dimod.SimulatedAnnealingSampler()
            response = sampler.sample_qubo(Q_dict_patch, **simulated_annealing_parameters)
            V_best = response.first.sample
            Q_best = response.first.sample
            flattened_m[count*D:(count+1)*D] = Q_best
            
            #Algorithm from Booth et al. 2017 D-Wave technical report
            """
            Q_tmp = np.random.randint(2,size=D,dtype=np.int8)
            init_state_dict = {i:Q_tmp[i] for i in range(D)}
            init_state = dimod.SampleSet.from_samples(dimod.as_samples(init_state_dict), 'BINARY', 0)
            response = TabuSampler().sample_qubo(Q_dict_patch,init_states=init_state,seed=42)
            V_best = response.record[0][1]
            Q_best = response.record[0][0]
            flattened_m[count*D:(count+1)*D] = Q_best

            bqm = dimod.BinaryQuadraticModel.from_qubo(Q_dict_patch)
            decomposer = EnergyImpactDecomposer(size=subproblem_size, rolling=True, rolling_history=1.0)
            state = State.from_sample(Q_best, bqm)

            for i in range(num_passes):
                state = decomposer.run(state).result()
                subproblem_indices = list(state.subproblem.variables)
                index[count][i] = subproblem_indices
            
                offset = (count%subproblems_per_qubo)*subproblem_size
                dict_index = int(count/subproblems_per_qubo)*num_passes+i
                for j in range(0,subproblem_size):
                    spj = subproblem_indices[j]
                    m_tmp = Q_best.copy()
                    m_tmp[subproblem_indices] = 0
                    d_spj = 2*np.dot(A[spj],m_tmp)
                    Q_dicts[dict_index][(offset+j,offset+j)] = A[spj][spj]+d_spj+b[spj]+1e-9
                    for k in range(j+1,subproblem_size):
                        spk = subproblem_indices[k]
                        Q_dicts[dict_index][(offset+j,offset+k)] = 2*A[spj][spk]+1e-9 #*2 because A is symmetric
            count += 1
            """

    logging.info("tabu V_best="+str(V_best))
    logging.info("tabu Q_best="+str(Q_best))

    return(Q_dicts,index,flattened_m)


def qubo_bsc_dwave(X,y,alpha,mu): #binary sparse coding
    #mu determines strength of individual state
    D = X.shape[1]
    w = np.ones(D)*mu
    m = update_m(X,y,w,alpha)
    return(m*w)
