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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import numpy as np
from easydict import EasyDict as edict

C = edict()
config = C
cfg = C
C.seed = 12345

############################################

#inference settings
#mode_val = "1img"
mode_val = "1img_small"
#mode_val = "valset"
#mode_val = "set5"

#algorithm for sparse coding & settings
#C.sc_algo = "sklearn_lasso" #Lasso Regression
#C.sc_algo = "qubo_bsc" #Classical Annealing
C.sc_algo = "dynex" #Neuromorphic sampling on Dynex
#C.sc_algo = "dynex_bsc" #Neuromorphic sampling on Dynex (simple bsc replacement)
#C.sc_algo = "qubo_bsc_dwave1" #Quantum Annealing (Hybrid Solvers)
#C.sc_algo = "qubo_bsc_dwave2" #Quantum Annealing (Direct Solvers)

C.lasso_alpha = 1e-5
C.bsc_alpha = 0.1
C.bsc_mu = 0.05
C.num_passes = 1
C.num_reads = 100
C.qubo_size = 512
C.subproblem_size = 32
C.beta = 1

C.Dl_path = "data/dicts/Dl_128_US3_L0.1_PS5_test_exp_train_2022_11_22_17_08_59.pkl"
C.Dh_path = "data/dicts/Dh_128_US3_L0.1_PS5_test_exp_train_2022_11_22_17_08_59.pkl"

############################################

C.D_size = 128
C.US_mag = 3
C.lmbd = 0.1
C.patch_size= 5
C.overlap = 3

############################################

C.root_dir = os.path.realpath(".")
print("root_dir:%s"%C.root_dir)

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.exp_name = 'exp_inference_'+exp_time
C.output_dir = os.path.join(*[C.root_dir,'output',C.exp_name])
os.makedirs(C.output_dir, exist_ok=True)
print("exp_name: %s"%C.exp_name)

############################################

val_hr_path = {
    "1img":"data/val_single/HR",
    "1img_small":"data/val_single_small2/HR",
    "valset":"data/val/HR",
    "set5":"data/Set5/set5_4_hr"
}
C.val_hr_path = val_hr_path[mode_val]

val_lr_path = {
    "1img":"data/val_single/LR",
    "1img_small":"data/val_single_small2/LR",
    "valset":"data/val/LR",
    "set5":"data/Set5/set5_4_lr"
}
C.val_lr_path = val_lr_path[mode_val]
