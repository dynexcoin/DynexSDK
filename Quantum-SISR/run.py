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
import os
from os import listdir, mkdir
from os.path import isdir
import time
import math
from tqdm import tqdm
import pickle
import logging
from skimage.io import imread, imsave
from skimage.color import rgb2ycbcr, ycbcr2rgb
from skimage.transform import resize
from skimage.exposure import match_histograms
from sklearn.metrics import mean_squared_error 
from sklearn.preprocessing import normalize
import skimage as ski
import dimod
import networkx as nx
from ScSR import ScSR
from backprojection import backprojection
from config_run import config

log_format = "%(asctime)s | %(message)s"
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.DEBUG, format=log_format, datefmt='%m/%d %I:%M:%S %p')

fh = logging.FileHandler(os.path.join(config.output_dir, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

logging.info("initialize logging")
logging.info("args = "+str(config))

#################################################################################

def normalize_signal(img, channel):
    if np.mean(img[:, :, channel]) * 255 > np.mean(img_lr_ori[:, :, channel]):
        ratio = np.mean(img_lr_ori[:, :, channel]) / (np.mean(img[:, :, channel]) * 255)
        img[:, :, channel] = np.multiply(img[:, :, channel], ratio)
    elif np.mean(img[:, :, channel]) * 255 < np.mean(img_lr_ori[:, :, channel]):
        ratio = np.mean(img_lr_ori[:, :, channel]) / (np.mean(img[:, :, channel]) * 255)
        img[:, :, channel] = np.multiply(img[:, :, channel], ratio)
    return img[:, :, channel]


def normalize_max(img):
    for m in range(img.shape[0]):
        for n in range(img.shape[1]):
            if img[m, n, 0] > 1:
                img[m, n, 0] = 1
            if img[m, n, 1] > 1:
                img[m, n, 1] = 1
            if img[m, n, 2] > 1:
                img[m, n, 2] = 1
    return img


#################################################################################

D_size = config.D_size
US_mag = config.US_mag
lmbd = config.lmbd
patch_size = config.patch_size

# Set which dictionary you want to use
with open(config.Dh_path, 'rb') as f:
    Dh = pickle.load(f)
Dh = normalize(Dh)
with open(config.Dl_path, 'rb') as f:
    Dl = pickle.load(f)
Dl = normalize(Dl)

### SET PARAMETERS
img_lr_dir = config.val_lr_path
img_hr_dir = config.val_hr_path
overlap = config.overlap
upscale = 3
maxIter = 100
img_type = '.png'

#################################################################################

img_lr_file = listdir(img_lr_dir)
img_lr_file = [item for item in img_lr_file if img_type in item]

quantum_objects = {}
if config.sc_algo=="qubo_bsc_dwave2":
    dwave.cloud.config.load_config("dwave.conf")
    Q_tmp = {}
    Dsp = config.subproblem_size
    for i in range(int(config.qubo_size/Dsp)):
        for j in range(Dsp):
            Q_tmp[(i*Dsp+j,i*Dsp+j)] = 0
            for k in range(j+1,Dsp):
                Q_tmp[(i*Dsp+j,i*Dsp+k)] = 0

    sampler = DWaveSampler(solver={'topology__type':'pegasus'})
    G = nx.Graph()
    G.add_edges_from(set(sampler.edgelist))
    start_time = time.time()
    emb = find_embedding(Q_tmp,G.edges)
    logging.info("embedding time: "+str(time.time()-start_time))
    fec_sampler = FixedEmbeddingComposite(sampler,emb)
    quantum_objects["fec_sampler"] = fec_sampler

for i in range(len(img_lr_file)):
    logging.info("image number %d"%i)
    # Read test image
    img_name = img_lr_file[i]
    img_name_dir = list(img_name)
    img_name_dir = np.delete(np.delete(np.delete(np.delete(img_name_dir, -1), -1), -1), -1)
    img_name_dir = ''.join(img_name_dir)
    img_lr = imread( os.path.join(*[img_lr_dir, img_name]) )
    logging.info("img_lr shape: "+str(img_lr.shape))

    # Read and save ground truth image
    img_hr = imread( os.path.join(*[img_hr_dir, img_name]) )
    logging.info("img_hr shape: "+str(img_hr.shape))
    imsave(os.path.join(*[config.output_dir,"%04d_3HR.png"%i]), img_hr)
    img_hr_y = rgb2ycbcr(img_hr)[:, :, 0]

    # Change color space
    img_lr_ori = img_lr
    temp = img_lr
    img_lr = rgb2ycbcr(img_lr)
    img_lr_y = img_lr[:, :, 0]
    img_lr_cb = img_lr[:, :, 1]
    img_lr_cr = img_lr[:, :, 2]

    # Upscale chrominance to color SR images
    img_sr_cb = resize(img_lr_cb, (img_hr.shape[0], img_hr.shape[1]), order=0)
    img_sr_cr = resize(img_lr_cr, (img_hr.shape[0], img_hr.shape[1]), order=0)

    # Super Resolution via Sparse Representation
    start_time = time.time()
    img_sr_y,img_sr_unc = ScSR(img_lr_y, img_hr_y.shape, upscale, Dh, Dl, lmbd, overlap, quantum_objects) #<== depending on sampler
    logging.info("ScSR time: "+str(time.time()-start_time))
    img_sr_y = backprojection(img_sr_y, img_lr_y, maxIter)
    img_sr_hmatched_y = match_histograms(image=img_sr_y,reference=img_lr_y,channel_axis=None)

    # Bicubic interpolation for reference
    img_bc = resize(img_lr_ori, (img_hr.shape[0], img_hr.shape[1]))
    imsave(os.path.join(*[config.output_dir,'%04d_1bicubic.png'%i]), ski.util.img_as_ubyte(img_bc))
    img_bc_y = rgb2ycbcr(img_bc)[:, :, 0]

##########################################################################################

    # Compute RMSE for the illuminance
    rmse_bc_hr = np.sqrt(mean_squared_error(img_hr_y, img_bc_y))
    rmse_bc_hr = np.zeros((1,)) + rmse_bc_hr
    rmse_sr_hr = np.sqrt(mean_squared_error(img_hr_y, img_sr_y))
    rmse_sr_hr = np.zeros((1,)) + rmse_sr_hr
    rmse_srhm_hr = np.sqrt(mean_squared_error(img_hr_y, img_sr_hmatched_y))
    rmse_srhm_hr = np.zeros((1,)) + rmse_srhm_hr

    logging.info('bicubic RMSE: '+str(rmse_bc_hr))
    logging.info('SR RMSE: '+str(rmse_sr_hr))
    logging.info('SR Histogram-matched RMSE: '+str(rmse_srhm_hr))

    y_psnr_bc_hr = 20*math.log10(255.0/rmse_bc_hr)
    y_psnr_sr_hr = 20*math.log10(255.0/rmse_sr_hr)
    y_psnr_srhm_hr = 20*math.log10(255.0/rmse_srhm_hr)
    logging.info('bicubic Y-Channel PSNR: '+str(y_psnr_bc_hr))
    logging.info('SR Y-Channel PSNR: '+str(y_psnr_sr_hr))
    logging.info('SRHM Y-Channel PSNR: '+str(y_psnr_srhm_hr))

##########################################################################################
    
    # Create colored SR images
    img_sr = np.stack((img_sr_y, img_sr_cb, img_sr_cr), axis=2)
    img_sr = ycbcr2rgb(img_sr)
    
    # Signal normalization
    for channel in range(img_sr.shape[2]):
        img_sr[:, :, channel] = normalize_signal(img_sr, channel)
    
    # Maximum pixel intensity normalization
    img_sr = normalize_max(img_sr)
    imsave(os.path.join(*[config.output_dir,'%04d_2SR.png'%i]), ski.util.img_as_ubyte(img_sr))

    img_sr_hmatched = np.stack((img_sr_hmatched_y, img_sr_cb, img_sr_cr), axis=2)
    img_sr_hmatched = ycbcr2rgb(img_sr_hmatched)
    imsave(os.path.join(*[config.output_dir,'%04d_2SRHM.png'%i]), ski.util.img_as_ubyte(img_sr_hmatched))

    img_sr_final = (np.clip(img_sr_hmatched,0,1)*255).astype(np.uint8)
    imsave(os.path.join(*[config.output_dir,'%04d_2SR_final.png'%i]), ski.util.img_as_ubyte(img_sr_final))

    if config.sc_algo=="qubo_bsc_dwave2":
        logging.info("mean unc (gibbs entropy)="+str(np.mean(img_sr_unc)))
        img_sr_unc = img_sr_unc/np.log(config.num_reads)
        img_sr_unc = (np.clip(img_sr_unc,0,1)*255).astype(np.uint8)
        imsave(os.path.join(*[config.output_dir,'%04d_SR_UNC.png'%i]), ski.util.img_as_ubyte(img_sr_unc))
