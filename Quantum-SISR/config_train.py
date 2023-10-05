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

C.dict_size   = 128          # dictionary size
C.lmbd        = 0.1          # sparsity regularization
C.patch_size  = 5            # image patch size
C.nSmp        = 100000       # number of patches to sample
C.upscale     = 3            # upscaling factor

############################################

C.root_dir = os.path.realpath(".")
print("root_dir:%s"%C.root_dir)

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.exp_name = 'exp_train_'+exp_time
C.output_dir = os.path.join(*[C.root_dir,'output',C.exp_name])
os.makedirs(C.output_dir, exist_ok=True)

print("exp_name: %s"%C.exp_name)
