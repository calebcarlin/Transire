# copyright Caleb Michael Carlin (2019)
# Released uder Lesser Gnu Public License (LGPL)
# See LICENSE file for details.

import numpy as np
import scipy as sp
from sklearn.gaussian_process.kernels import _check_length_scale
from scipy.spatial.distance import pdist,cdist,squareform
from sklearn.gaussian_process.kernels import StationaryKernelMixin
from sklearn.gaussian_process.kernels import NormalizedKernelMixin
from sklearn.gaussian_process.kernels import Kernel
from sklearn.gaussian_process.kernels import Hyperparameter
from sklearn.gaussian_process import GaussianProcessRegressor

class RBF_bin(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    """
    Adaptation of RBF kernel to allow for using arrays of bits as
    the inputs and the Sokal-Michener dissimilarity to define the
    distance between two inputs.

    As the code holds closely to the RBF_kernel distributed with
    Scikit-learn, the terms of the BSD License are included at the
    end of the source file.  The inclusion of this code does not represent
    any association with or endorscement of Transire by Scikit-learn
    Developers.
    """

    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5)):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds

    @property
    def anisotropic(self):
    #we don't need anisotropic parameters, so we disble it here
        return False

    @property
    def hyperparameter_length_scale(self):
    #because we can't have anisotropic, this just returns hyperparameters
        return Hyperparameter(
                "length_scale", "numeric", self.length_scale_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):

        X = np.atleast_2d(X)
        length_scale = _check_length_scale(X, self.length_scale)
        if Y is None:
            dists = pdist(X, 'sokalmichener')
            K = np.exp(-0.5*(dists/length_scale)**2)
            K = squareform(K)
            np.fill_diagonal(K,1.0)
        else:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated when Y is None.")
            dists = cdist(X,Y, 'sokalmichener')
            K = np.exp(-0.5*(dists/length_scale)**2)

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                return K, np.empty((X.shape[0], X.shape[0], 0))
            elif not self.anisotropic or length_scale.shape[0] == 1:
                K_gradient = \
                    (K*squareform(dists))[:,:, np.newaxis]
                return K, K_gradient
            elif self.anisotropic:
                K_gradient = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2 \
                    / (length_scale ** 2)
                K_gradient *= K[..., np.newaxis]
                return K, K_gradient
        else:
            return K

    def __repr__(self):
        if self.anisotropic:
            return "{0}(length_scale=[{1}])".format(
                self.__class__.__name__, ", ".join(map("{0:.3g}".format,
                    self.length_scale)))
        else: 
            return "{0}(length_scale={1:.3g})".format(
                self.__class__.__name__, np.ravel(self.length_scale)[0])

#New BSD License
#
#Copyright (c) 2007–2019 The scikit-learn developers.
#All rights reserved.


#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:

#  a. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#  b. Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#  c. Neither the name of the Scikit-learn Developers  nor the names of
#     its contributors may be used to endorse or promote products
#     derived from this software without specific prior written
#     permission. 


#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
#ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
#OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
#DAMAGE.
