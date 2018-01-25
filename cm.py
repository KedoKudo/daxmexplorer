#!/usr/bin/env python

"""
In package module provide basic support for continuum mechanics related
calculations.
All functions should be implemented in numpy.
Usage of scipy package should be imported at function scope.
"""

import numpy as np

def get_deviatoric_defgrad(_defgrad):
    """calculate deviatoric deformation gradient with given defgrad"""
    dim = _defgrad.shape[0]
    return np.power(np.linalg.det(_defgrad), -1.0/dim)*_defgrad

if __name__ == "__main__":
    # module level demo 

    # calculate deviatoric deformation gradient
    defgrad = np.eye(3) + 1e-4*(np.ones(9, dtype=np.float64) - 2.*np.random.random(9)).reshape(3,3)
    print("defgrad:\n{}\ndefgrad_D:\n{}\n".format(defgrad, get_deviatoric_defgrad(defgrad)))
