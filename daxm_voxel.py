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
    attributes = ['ref_frame', 'coords', 
                  'scatter_vecs', 'plane', 
                  'recip_base', 'peaks', 
                  'depth',
                 ]

    def __init__(self,
                 ref_frame='APS',
                 coords=np.zeros(3),
                 pattern_image="dummy",
                 scatter_vecs=None,
                 plane=None,
                 recip_base=np.eye(3),
                 peaks=np.random.random((2,3)),
                 depth=0,
                ):
        self.ref_frame = ref_frame
        self.coords = coords
        self.pattern_image = pattern_image
        self.scatter_vecs = scatter_vecs
        self.plane = plane
        self.recip_base = recip_base
        self.peaks = peaks
        self.depth = depth

    def read(self, h5file, voxelName):
        """update self with data stored in given HDF5 archive"""
        imgName = voxelName

        def get_data(h5f, path):
            tmpdst = h5f[path]
            datdst = np.zeros(tmpdst.shape)
            tmpdst.read_direct(datdst)
            return datdst

        with h5py.File(h5file, 'r') as h5f:
            thisvoxel = h5f[imgName]

            self.ref_frame = thisvoxel.attrs['ref_frame']

            self.coords = get_data(thisvoxel, 'coords')
            self.scatter_vecs = get_data(thisvoxel, 'scatter_vecs')
            self.plane = get_data(thisvoxel, 'plane')
            self.recip_base = get_data(thisvoxel, 'recip_base')
            self.peaks = get_data(thisvoxel, 'peaks')
            self.depth = get_data(thisvoxel, 'depth')
            

    def write(self, h5file=None):
        """write the DAXM voxel data to a HDF5 archive"""
        imgName = self.pattern_image.split("/")[-1].replace(".h5", "")
        if h5file is None:
            h5file = imgName + "_data.h5"

        with h5py.File(h5file, 'a') as h5f:
            try:
                del h5f[imgName]
                voxelStatus = 'updated'
            except:
                voxelStatus = 'new'

            h5f.create_dataset("{}/coords".format(imgName), data=self.coords)
            h5f.create_dataset("{}/scatter_vecs".format(imgName), data=self.scatter_vecs)
            h5f.create_dataset("{}/plane".format(imgName), data=self.plane)
            h5f.create_dataset("{}/recip_base".format(imgName), data=self.recip_base)
            h5f.create_dataset("{}/peaks".format(imgName), data=self.peaks)
            h5f.create_dataset("{}/depth".format(imgName), data=self.depth)

            h5f[imgName].attrs['ref_frame'] = self.ref_frame
            h5f[imgName].attrs['voxelStatus'] = voxelStatus

            h5f.flush()

    def scatter_vecs0(self):
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
        q0 = self.scatter_vecs0()
        q = self.scatter_vecs

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
                                                  args = (self.scatter_vecs0(),self.scatter_vecs),
                                                  method = 'BFGS',
                                                  tol = 1e-14,
                                                #   constraints = {'type':'ineq',
                                                #                  'fun': lambda x: constraint(x,eps),
                                                #                 },
                                                 ).x.reshape(3,3)

    # def pairPlane2q(self,method=""):
    #     pass


if __name__ == "__main__":

    # read/write HDF5 support

    # strain quantitifiaction with mock DAXM voxel
    N = 30
    test_eps = 1.0e-4

    test_f = test_eps*(np.ones(9)-2.*np.random.random(9)).reshape(3,3)
    test_vec0 = (np.ones(3*N)-2.*np.random.random(3*N)).reshape(3, N)
    test_vec  = np.dot(np.eye(3)+test_f, test_vec0)

    from daxm_analyzer import cm
    deviator = cm.get_deviatoric_defgrad

    daxmVoxel = DAXMvoxel(ref_frame='APS',
                          coords=np.zeros(3),
                          pattern_image="dummy2",
                          scatter_vecs=test_vec,
                          plane=test_vec0,
                          recip_base=np.eye(3),
                          peaks=np.random.random((2, 10)),
                         )

    print("dev_correct\n", deviator(np.eye(3)+test_f))
    print('dev_L2\n', deviator(daxmVoxel.deformation_gradientL2()))
    print('dev_opt\n', deviator(daxmVoxel.deformation_gradient_opt()))

    # write to single file
    daxmVoxel.write(h5file='dummy_data.h5')

    # read from file 
    daxmvoxel2 = DAXMvoxel()
    daxmvoxel2.read('dummy_data.h5', "dummy2")
    print("duplicate mock DAXM voxel constructed from HDF5:")
    print('dev_L2\n', deviator(daxmvoxel2.deformation_gradientL2()))
    print('dev_opt\n', deviator(daxmvoxel2.deformation_gradient_opt()))
