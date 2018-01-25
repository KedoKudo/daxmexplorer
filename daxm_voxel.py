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
                 coordFrame='APS', 
                 coords=np.zeros(3,dtype=np.float64), 
                 detectorImage=None, 
                 q=None, 
                 plane=None, 
                 recipBase=np.eye(3,dtype=np.float64),
                 ):
        self.coordFrame=coordFrame
        self.coords=coords
        self.detectorImage=detectorImage
        self.q=q
        self.plane=plane
        self.recipBase=recipBase

    def fromH5(self, h5file, label):
        """update self with data stored in given HDF5 archive"""
        pass

    def write2H5(self, h5file=None, label=None):
        """write the DAXM voxel data to a HDF5 archive"""
        pass
    
    def q0(self):
        return np.dot(self.recipBase,self.plane)

    def toAPS(self):
        pass

    def toTSL(self):
        pass

    def toXHF(self):
        pass

    def quality(self):
        pass

    def deformationGradientL2(self):
        # quick summary of the least square solution
        # F* q0 = q
        # ==> F* q0 q0^T = q q0^T
        # ==> F* = (q q0^T)(q0 q0^T)^-1
        #              A       B
        q0 = self.q0()
        A = np.dot(self.q, q0.T)
        B = np.dot(q0    , q0.T)
        # Fstar = np.dot(A, np.linalg.pinv(B))

        # F = F*^(-T) = A^-T B^T
        # inverting B can be dangerous
        return np.dot(np.linalg.inv(A).T, B.T)

    def deformationGradientOptimization(self):
        
        import scipy.optimize

        #---------------
        def constraint(f,e):
        #---------------
            return len(f)*e - np.sum(np.abs(f))

        #---------------
        def objectiveIce(f,vec0,vec):
        #---------------
            estimate = np.dot(np.eye(3,dtype=np.float64)+f.reshape(3,3),vec0)
            return np.sum(1.0 - np.einsum('ij,ij->j',
                                            vec     /np.linalg.norm(vec     ,axis=0),
                                            estimate/np.linalg.norm(estimate,axis=0),
                                        )
                        )

        return np.eye(3)+ scipy.optimize.minimize(
                            objectiveIce,
                            x0 = np.zeros(3*3),
                            args = (vec0,vec),
                            method = 'COBYLA',
                            tol = 1e-14,
                            constraints = {'type':'ineq',
                                            'fun': lambda x: constraint(x,eps),
                                            },
                            ).x.reshape(3,3)

    def pairPlane2q(self,method=""):
        pass


if __name__ == "__main__":

    # read/write HDF5 support

    # strain quantitifiaction with mock DAXM voxel
    N = 30
    eps = 1.0e-4

    f = eps*(np.ones(9,dtype=np.float64)-2.*np.random.random(9)).reshape(3,3)
    vec0 = (np.ones(3*N,dtype=np.float64)-2.*np.random.random(3*N)).reshape(3,N)
    vec  = np.dot(np.eye(3,dtype=np.float64)+f,vec0)

    from daxm_analyzer import cm
    deviator = cm.get_deviatoric_defgrad

    daxm_voxel = DAXMvoxel( coordFrame='APS',
                            coords=np.zeros(3,dtype=np.float64),
                            detectorImage=None,
                            q=vec,
                            plane=vec0,
                            recipBase=np.eye(3,dtype=np.float64),
                            )

    print("dev_correct\n", deviator(np.eye(3,dtype=np.float64)+f))

    print('dev_L2\n',deviator(daxm_voxel.deformationGradientL2()))

    print('dev_opt\n', deviator(daxm_voxel.deformationGradientOptimization()))
