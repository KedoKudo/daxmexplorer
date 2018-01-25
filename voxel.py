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

    # ** XHF <-> TSL
    theta_1 = -np.pi
    R_XHF2TSL = np.array([[1.0,              0.0,              0.0],
                          [0.0,  np.cos(theta_1), -np.sin(theta_1)],
                          [0.0,  np.sin(theta_1),  np.cos(theta_1)]])
    R_TSL2XHF = R_XHF2TSL.T
    # ** XHF <-> APS
    theta_2 = -0.25*np.pi
    R_XHF2APS = np.array([[1.0,              0.0,              0.0],
                          [0.0,  np.cos(theta_2), -np.sin(theta_2)],
                          [0.0,  np.sin(theta_2),  np.cos(theta_2)]])
    R_APS2XHF = R_XHF2APS.T
    # ** APS <-> TSL
    theta_3 = -0.75*np.pi
    R_APS2TSL = np.array([[1.0,              0.0,              0.0],
                          [0.0,  np.cos(theta_3), -np.sin(theta_3)],
                          [0.0,  np.sin(theta_3),  np.cos(theta_3)]])
    R_TSL2APS = R_APS2TSL.T
    # ** self <-> self
    R_TSL2TSL = R_APS2APS = R_XHF2XHF = np.eye(3)

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

    def tranfer_frame(self, g):
        """transfer reference frame with given orientation matrix, g"""
        # NOTE: g matrix represents passive rotation

        # convert coordinates
        self.coords = np.dot(g, self.coords)

        # convert scattering vectors
        self.scatter_vecs = np.dot(g, self.scatter_vecs)

        # convert reciprocal base
        self.recip_base = np.dot(g, self.recip_base)

    def toAPS(self):
        """convert reference frame to APS frame"""
        # NOTE: plane index and peaks position is tied to its own
        #       frame.

        # set the rotation matrix, but we use the orientation matrix
        if self.ref_frame.upper() == "APS":
            r = self.R_APS2APS
        elif self.ref_frame.upper() == "TSL":
            r = self.R_TSL2APS
        elif self.ref_frame.upper() == "XHF":
            r = self.R_XHF2APS
        else:
            raise ValueError("unknown framework in this voxel: {}".format(self.ref_frame))

        self.tranfer_frame(r.T)

    def toTSL(self):
        """convert reference frame to TSL frame"""
        if self.ref_frame.upper() == "APS":
            r = self.R_APS2TSL
        elif self.ref_frame.upper() == "TSL":
            r = self.R_TSL2TSL
        elif self.ref_frame.upper() == "XHF":
            r = self.R_XHF2TSL
        else:
            raise ValueError("unknown framework in this voxel: {}".format(self.ref_frame))

        self.tranfer_frame(r.T)      

    def toXHF(self):
        """convert reference frame to XHF frame"""
        if self.ref_frame.upper() == "APS":
            r = self.R_APS2XHF
        elif self.ref_frame.upper() == "TSL":
            r = self.R_TSL2XHF
        elif self.ref_frame.upper() == "XHF":
            r = self.R_XHF2XHF
        else:
            raise ValueError("unknown framework in this voxel: {}".format(self.ref_frame))

        self.tranfer_frame(r.T)

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

    def pair_scattervec_plane(self):
        """pair the recorded scattering vectors and the indexation results"""
        new_scatter_vec = np.zeros_like(self.plane)
        new_peaks = np.zeros((2, self.plane.shape[1]))

        from daxmexplorer.vecmath import normalize
        qs = normalize(self.scatter_vecs, axis=0)   # normalize each scatter vector (column stacked)

        for i in range(self.plane.shape[1]):
            q0 = normalize(np.dot(self.recip_base, self.plane[:, i]))
            angular_diff = np.dot(q0.T, qs)
            # pair q0 and qs with the smallest angular difference
            idx = np.argmin(angular_diff)
            new_scatter_vec[:, i] = self.scatter_vecs[:, idx]
            new_peaks[:, i] = self.peaks[:, idx]

        # update scatter vectors
        self.scatter_vecs = new_scatter_vec
        self.peaks = new_peaks


if __name__ == "__main__":

    # read/write HDF5 support

    # strain quantitifiaction with mock DAXM voxel
    N = 30
    test_eps = 1.0e-4

    test_f = test_eps*(np.ones(9)-2.*np.random.random(9)).reshape(3,3)
    test_vec0 = (np.ones(3*N)-2.*np.random.random(3*N)).reshape(3, N)
    test_vec  = np.dot(np.eye(3)+test_f, test_vec0)

    from daxmexplorer import cm
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
