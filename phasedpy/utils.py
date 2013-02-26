#!/usr/bin/env python ## utils.py ---

##  Author: sauravtuladhar@gmail.com
## Copyright Saurav R. Tuladhar <sauravtuladhar@gmail.com>
## Version: $Id: utils.py,v 0.0 2013/02/24 03:07:26 saurav Exp$

''' This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>. '''

''' Utilities module: Collection of utility functions for statistical signal processing in general.'''


import numpy as np
from numpy.random import randn
from numpy.linalg.linalg import eig, dot
from matplotlib.mlab import isvector

def hsqrt(A):
    '''
    Computes Hermitian square root of input Hermitian matrix A

    Parameters
    ----------
    A : (N, N) array_like
        A square array of real or complex elements

    Returns
    -------
    B : (N, N) ndarray (matrix)
    Hermitian square root of Ax
    '''
    w, V = eig(A)
    D = np.diag(np.sqrt(w))
    B = dot(V, dot(D, V.T))
    return B

def gencosine(a, b, C=1):
    """
    Computes generalized cosine of angle
    between two vectors a and b in subspace spanned by C

    Parameters
    ----------
    a, b : (N, 1) array_like
            Input vectors of real or complex element

        Returns
        -------
        x : scalar
            Generalized cosine of angle between a and b in subspace of C

    """
    # Validate input
    if not (isvector(a) and isvector(b)):
        raise TypeError("Input not a vector")

    if not C.ndim == 2:
        raise TypeError("Third argument must be a matrix")
    
    nr = np.abs(dot(a.T, dot(C, b)))**2
    dr = dot(a.T, dot(C, a)) * dot(b.T, dot(C, b));
    x = float(nr)/dr
    return x


def crandn(n, m=1):
    '''
    Generates circular complex gaussian random variable, vector or matrix

    Parameters
    ----------
    n, m : scalars
       Specify the the shape of the gausian variable to be generated

    Returns
    -------
    G : (n, m) array_like
        Gaussian random variable        
    '''
    G = (1./np.sqrt(2))*(randn(n, m) + 1j*randn(n, m))
    return G








