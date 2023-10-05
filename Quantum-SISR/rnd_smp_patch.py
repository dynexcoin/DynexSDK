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
from skimage.io import imread
from sample_patches import sample_patches
from tqdm import tqdm

def rnd_smp_patch(img_path, patch_size, num_patch, upscale):
    img_dir = listdir(img_path)

    img_num = len(img_dir)
    nper_img = np.zeros((img_num, 1))

    for i in tqdm(range(img_num)):
        img = imread('{}{}'.format(img_path, img_dir[i]))
        nper_img[i] = img.shape[0] * img.shape[1]

    nper_img = np.floor(nper_img * num_patch / np.sum(nper_img, axis=0))

    for i in tqdm(range(img_num)):
        patch_num = int(nper_img[i])
        img = imread('{}{}'.format(img_path, img_dir[i]))
        H, L = sample_patches(img, patch_size, patch_num, upscale)
        if i == 0:
            Xh = H
            Xl = L
        else:
            Xh = np.concatenate((Xh, H), axis=1)
            Xl = np.concatenate((Xl, L), axis=1)

    return Xh, Xl
