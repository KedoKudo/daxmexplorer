#!/usr/bin/env python

from __future__ import print_function
import h5py
import numpy as np

class DAXMvoxel(object):
    """
    DAXM voxel stores the crystallograhic information derived from DAXM indexation results.
    By default, all data is recoreded in the APS coordinate system.
    Coordinate system transformation is done via binded method.
    """

    def __init__(self,
                 ref_frame='APS',
                 coords=np.zeros(3, dtype=np.float64),
                 pattern_image=None,
                 scater_vecs=None,
                 plane=None,
                 recip_base=np.eye(3, dtype=np.float64),
                 peaks=None,
                 depth=0,
                ):
        self.ref_frame = ref_frame
        self.coords = coords
        self.pattern_image = pattern_image
        self.scater_vecs = scater_vecs
        self.plane = plane
        self.recip_base = recip_base
        self.peaks = peaks
        self.depth = depth

    def read(self, h5file, label):
        """update self with data stored in given HDF5 archive"""
        pass

    def write(self, h5file=None, label=None):
        """write the DAXM voxel data to a HDF5 archive"""
        pass

    def scater_vecs0(self):
        """return the strain-free scattering vectors calculated from hkl index"""
        return np.dot(self.recip_base, self.plane)

    # def toAPS(self):
    #     pass

    # def toTSL(self):
    #     pass

    # def toXHF(self):
    #     pass

    # def quality(self):
    #     pass

    def deformation_gradientL2(self):
        """extract lattice deformation gradient using least-squares regression(L2 optimization)"""
        # quick summary of the least square solution
        # F* q0 = q
        # ==> F* q0 q0^T = q q0^T
        # ==> F* = (q q0^T)(q0 q0^T)^-1
        #              A       B
        q0 = self.scater_vecs0()
        q = self.scater_vecs

        A = np.dot(q, q0.T)
        B = np.dot(q0, q0.T)
        # Fstar = np.dot(A, np.linalg.pinv(B))

        # F = F*^(-T) = A^-T B^T
        # inverting B can be dangerous
        return np.dot(np.linalg.inv(A).T, B.T)

    def deformation_gradient_opt(self):
        """extract lattice deformation gardient using nonlinear optimization"""

        import scipy.optimize

        def constraint(constraint_f, e):
            return len(constraint_f)*e - np.sum(np.abs(constraint_f))

        def objectiveIce(f, vec0, vec):
            estimate = np.dot(np.eye(3)+f.reshape(3, 3), vec0)
            return np.sum(1.0 - np.einsum('ij,ij->j',
                                          vec/np.linalg.norm(vec, axis=0),
                                          estimate/np.linalg.norm(estimate, axis=0),
                                         )
                         )

        return np.eye(3)+ scipy.optimize.minimize(objectiveIce,
                                                  x0 = np.zeros(3*3),
                                                  args = (self.scater_vecs0(),self.scater_vecs),
                                                  method = 'COBYLA',
                                                  tol = 1e-14,
                                                  constraints = {'type':'ineq',
                                                                 'fun': lambda x: constraint(x,eps),
                                                                },
                                                 ).x.reshape(3,3)

    # def pairPlane2q(self,method=""):
    #     pass


if __name__ == "__main__":

    # read/write HDF5 support

    # strain quantitifiaction with mock DAXM voxel
    N = 30
    eps = 1.0e-4

    test_f = eps*(np.ones(9)-2.*np.random.random(9)).reshape(3,3)
    test_vec0 = (np.ones(3*N)-2.*np.random.random(3*N)).reshape(3, N)
    test_vec  = np.dot(np.eye(3)+test_f, test_vec0)

    from daxm_analyzer import cm
    deviator = cm.get_deviatoric_defgrad

    daxmVoxel = DAXMvoxel(ref_frame='APS',
                          coords=np.zeros(3),
                          pattern_image=None,
                          scater_vecs=test_vec,
                          plane=test_vec0,
                          recip_base=np.eye(3),
                         )

    print("dev_correct\n", deviator(np.eye(3)+test_f))

    print('dev_L2\n', deviator(daxmVoxel.deformation_gradientL2()))

    print('dev_opt\n', deviator(daxmVoxel.deformation_gradient_opt()))
