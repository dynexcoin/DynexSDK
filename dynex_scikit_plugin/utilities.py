# The following traversal code is adapted from NumPy's implementation.

# Copyright (c) 2005-2022, NumPy Developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above
#        copyright notice, this list of conditions and the following
#        disclaimer in the documentation and/or other materials provided
#        with the distribution.
#
#     * Neither the name of the NumPy Developers nor the names of any
#        contributors may be used to endorse or promote products derived
#        from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Modifications are licensed under the Apache 2.0 Software license.

# Copyright 2023 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import typing

import numpy as np
import numpy.typing as npt

__all__ = ["corrcoef", "cov", "dot_2d"]


def corrcoef(x: npt.ArrayLike, *,
             out: typing.Optional[np.ndarray] = None,
             rowvar: bool = True,
             copy: bool = True,
             ) -> np.ndarray:
    """A drop-in replacement for :func:`numpy.corrcoef`.

    This method is modified to avoid unnecessary memory usage when working with
    :class:`numpy.memmap` arrays.
    It does not support the full range of arguments accepted by
    :func:`numpy.corrcoef`.

    Additionally, in the case that a row of ``x`` is fixed, this method
    will return a correlation value of 0 rather than :class:`numpy.nan`.

    Args:
        x: See :func:`numpy.corrcoef`.

        out: Output argument. This must be the exact kind that would be returned
            if it was not used.

        rowvar: See :func:`numpy.corrcoef`.

        copy: If ``True``, ``x`` is not modified by this function.

    Returns:
        See :func:`numpy.corrcoef`.

    """
    c = cov(x, out=out, rowvar=rowvar, copy=copy)
    try:
        d = np.diag(c)
    except ValueError:
        # scalar covariance
        # nan if incorrect value (nan, inf, 0), 1 otherwise
        return c / c
    stddev = np.sqrt(d.real)

    # the places that stddev == 0 are exactly the places that the columns
    # are fixed. We can safely ignore those when dividing
    np.divide(c, stddev[:, None], out=c, where=stddev[:, None] != 0)
    np.divide(c, stddev[None, :], out=c, where=stddev[None, :] != 0)

    # Clip real and imaginary parts to [-1, 1].  This does not guarantee
    # abs(a[i,j]) <= 1 for complex arrays, but is the best we can do without
    # excessive work.
    np.clip(c.real, -1, 1, out=c.real)
    if np.iscomplexobj(c):
        np.clip(c.imag, -1, 1, out=c.imag)

    return c


def cov(m: npt.ArrayLike, *,
        out: typing.Optional[np.ndarray] = None,
        rowvar: bool = True,
        copy: bool = True,
        ) -> np.ndarray:
    """A drop-in replacement for :func:`numpy.cov`.

    This method is modified to avoid unnecessary memory usage when working with
    :class:`numpy.memmap` arrays.
    It does not support the full range of arguments accepted by
    :func:`numpy.cov`.

    Args:
        m: See :func:`numpy.cov`.

        out: Output argument. This must be the exact kind that would be returned
            if it was not used.

        rowvar: See :func:`numpy.cov`.

        copy: If ``True``, ``x`` is not modified by this function.

    Returns:
        See :func:`numpy.cov`.

    """
    # we want to modify X, so if copy=True we make a copy and re-call
    if copy:
        if hasattr(m, "flush"):
            # we could do a lot of fiddling here, but it's easier to just
            # disallow this case and rely on the user making a modifiable
            # X
            raise ValueError("memmap arrays cannot be copied easily")

        return cov(np.array(m), rowvar=rowvar, copy=False, out=out)

    # handle array-like
    if isinstance(m, np.memmap):
        X = m
    else:
        X = np.atleast_2d(np.asarray(m, dtype=np.result_type(m, np.float64)))

    if X.ndim != 2:
        raise ValueError("X must have 2 dimensions")

    if not rowvar and X.shape[0] != 1:
        X = X.T

    # Get the product of frequencies and weights
    avg = np.average(X, axis=1)

    # Determine the normalization
    fact = max(X.shape[1] - 1, 0)

    X -= avg[:, None]

    if hasattr(m, "flush"):
        X.flush()

    X_T = X.T

    out = dot_2d(X, X_T.conj(), out=out)
    out *= np.true_divide(1, fact)

    if hasattr(out, "flush"):
        out.flush()

    return out


def dot_2d(a: npt.ArrayLike, b: npt.ArrayLike, *,
           out: typing.Optional[np.ndarray] = None,
           chunksize: int = int(1e+9),
           ) -> np.ndarray:
    """A drop-in replacment for :func:`numpy.dot` for 2d arrays.

    This method is modified to avoid unnecessary memory usage when working with
    :class:`numpy.memmap` arrays.

    Args:
        a: See :func:`numpy.dot`. ``a.ndim`` must be 2.
        b: See :func:`numpy.dot`. ``b.ndim`` must be 2.
        out: See :func:`numpy.dot`.
        chunksize: The number of bytes that should be created by each step
            of the multiplication. This is used to keep the total memory
            usage low when multiplying :class:`numpy.memmap` arrays.

    Returns:
        See :func:`numpy.dot`.

    """
    if not isinstance(a, np.memmap):
        a = np.asarray(a)
    if not isinstance(b, np.memmap):
        b = np.asarray(b)

    if a.ndim != 2:
        raise ValueError("a must be a 2d array")
    if b.ndim != 2:
        raise ValueError("b must be a 2d array")

    if out is None:
        out = np.empty((a.shape[0], b.shape[1]), dtype=np.result_type(a, b))
    elif out.shape[0] != a.shape[0] or out.shape[1] != b.shape[1]:
        raise ValueError(f"out must be a ({a.shape[0]}, {b.shape[1]}) array")

    is_memmap = hasattr(out, "flush")

    num_rows = max(chunksize // (out.dtype.itemsize * out.shape[1]), 1)
    for start in range(0, out.shape[0], num_rows):
        np.dot(a[start:start+num_rows, :], b, out=out[start:start+num_rows, :])

        if is_memmap:
            out.flush()

    return out
