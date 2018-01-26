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
                  'scatter_vec', 'plane', 
                  'recip_base', 'peak', 
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
    
    g_to_from = {
          'APS': {
                   'APS': R_APS2APS,
                   'TSL': R_APS2TSL,
                   'XHF': R_APS2XHF,
                 },
          'TSL': {
                   'APS': R_TSL2APS,
                   'TSL': R_TSL2TSL,
                   'XHF': R_TSL2XHF,
                 },
          'XHF': {
                   'APS': R_XHF2APS,
                   'TSL': R_XHF2TSL,
                   'XHF': R_XHF2XHF,
                 },
    }

    def __init__(self,
                 name=None,
                 ref_frame='APS',
                 coords=np.zeros(3),
                 pattern_image=None,
                 scatter_vec=None,
                 plane=None,
                 recip_base=np.eye(3),
                 peak=np.random.random((2,3)),
                 depth=0,
                ):
        self.name = name
        self.ref_frame = ref_frame
        self.coords = coords
        self.pattern_image = pattern_image
        self.scatter_vec = scatter_vec
        self.plane = plane
        self.recip_base = recip_base
        self.peak = peak
        self.depth = depth

    def __repr__(self):
        return '\n'.join([
          'name: {}'.format(self.name),
          'frame: {}'.format(self.ref_frame),
          'coords: {}'.format(self.coords),
          'image: {}'.format(self.pattern_image),
        ])

    def read(self, h5file, voxelName=None):
        """update self with data stored in given HDF5 archive"""
        if voxelName is None: raise Exception

        self.name = voxelName

        def get_data(h5f, path):
            tmpdst = h5f[path]
            datdst = np.zeros(tmpdst.shape)
            tmpdst.read_direct(datdst)
            return datdst

        with h5py.File(h5file, 'r') as h5f:
            thisvoxel = h5f[voxelName]

            self.pattern_image = thisvoxel.attrs['pattern_image']
            self.ref_frame     = thisvoxel.attrs['ref_frame']

            self.coords       = get_data(thisvoxel, 'coords')
            self.scatter_vec  = get_data(thisvoxel, 'scatter_vec')
            self.plane        = get_data(thisvoxel, 'plane')
            self.recip_base   = get_data(thisvoxel, 'recip_base')
            self.peak         = get_data(thisvoxel, 'peak')
            self.depth        = get_data(thisvoxel, 'depth')
            

    def write(self, h5file=None):
        """write the DAXM voxel data to a HDF5 archive"""
        if None in [self.name,h5file] : raise Exception
        
        with h5py.File(h5file, 'a') as h5f:
            try:
                del h5f[self.name]
                voxelStatus = 'updated'
            except:
                voxelStatus = 'new'

            h5f.create_dataset("{}/coords".format(self.name), data=self.coords)
            h5f.create_dataset("{}/scatter_vec".format(self.name), data=self.scatter_vec)
            h5f.create_dataset("{}/plane".format(self.name), data=self.plane)
            h5f.create_dataset("{}/recip_base".format(self.name), data=self.recip_base)
            h5f.create_dataset("{}/peak".format(self.name), data=self.peak)
            h5f.create_dataset("{}/depth".format(self.name), data=self.depth)

            h5f[self.name].attrs['pattern_image'] = self.pattern_image
            h5f[self.name].attrs['ref_frame']     = self.ref_frame
            h5f[self.name].attrs['voxelStatus']   = voxelStatus

            h5f.flush()

    def scatter_vec0(self):
        """return the strain-free scattering vectors calculated from hkl index"""
        return np.dot(self.recip_base, self.plane)

    def toFrame(self, target=None):
        """transfer reference frame with given orientation matrix, g"""
        if target is None: return
        if target not in g_to_from: raise Exception
        # NOTE: g matrix represents passive rotation

        # convert coordinates
        self.coords = np.dot(g_to_from[target][self.ref_frame], self.coords)

        # convert scattering vectors
        self.scatter_vec = np.dot(g_to_from[target][self.ref_frame], self.scatter_vec)

        # convert reciprocal base
        self.recip_base = np.dot(g_to_from[target][self.ref_frame], self.recip_base)

    # def quality(self):
    #     pass

    def deformation_gradientL2(self):
        """extract lattice deformation gradient using least-squares regression(L2 optimization)"""
        # quick summary of the least square solution
        # F* q0 = q
        # ==> F* q0 q0^T = q q0^T
        # ==> F* = (q q0^T)(q0 q0^T)^-1
        #              A       B
        q0 = self.scatter_vec0()
        q = self.scatter_vec

        A = np.dot(q, q0.T)
        B = np.dot(q0, q0.T)
        # Fstar = np.dot(A, np.linalg.pinv(B))

        # F = F*^(-T) = A^-T B^T
        # inverting B can be dangerous
        return np.dot(np.linalg.inv(A).T, B.T)

    def deformation_gradient_opt(self, eps=1e-1):
        """extract lattice deformation gardient using nonlinear optimization"""
        # NOTE: a large bound guess is better than a smaller bound

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
        
        def objectiveKratos(f, vec0, vec):
            estimate = np.dot(np.eye(3)+f.reshape(3, 3), vec0)

            # angular difference
            angdif = vec/np.linalg.norm(vec,axis=0) - estimate/np.linalg.norm(estimate,axis=0)
            angdif = np.average(np.linalg.norm(angdif, axis=0))

            # length difference
            lendif = np.log(np.linalg.norm(vec,axis=0)/np.linalg.norm(estimate,axis=0))
            tmp = np.absolute(np.linalg.norm(vec,axis=0) - 1)
            msk = np.where(tmp > 1e-8)
            lendif = np.average(np.absolute(lendif[msk]))

            return angdif + lendif

        return np.eye(3)+ scipy.optimize.minimize(objectiveKratos,
                                                  x0 = np.zeros(3*3),
                                                  args = (self.scatter_vec0(),self.scatter_vec),
                                                  method = 'COBYLA',
                                                  tol = 1e-14,
                                                  constraints = {'type':'ineq',
                                                                 'fun': lambda x: constraint(x,eps),
                                                                },
                                                 ).x.reshape(3,3)

    def pair_scattervec_plane(self):
        """pair the recorded scattering vectors and the indexation results"""
        new_scatter_vec = np.zeros_like(self.plane)
        new_peak = np.zeros((2, self.plane.shape[1]))

        from daxmexplorer.vecmath import normalize
        qs = normalize(self.scatter_vec, axis=0)   # normalize each scatter vector (column stacked)

        for i in range(self.plane.shape[1]):
            q0 = normalize(np.dot(self.recip_base, self.plane[:, i]))
            angular_diff = np.dot(q0.T, qs)
            # pair q0 and qs with the smallest angular difference
            idx = np.argmin(angular_diff)
            new_scatter_vec[:, i] = self.scatter_vec[:, idx]
            new_peak[:, i] = self.peak[:, idx]

        # update scatter vectors
        self.scatter_vec = new_scatter_vec
        self.peak = new_peak


if __name__ == "__main__":

    # read/write HDF5 support

    # strain quantitifiaction with mock DAXM voxel
    N = 30
    test_eps = 1.0e-3

    test_f = test_eps*(np.ones(9)-2.*np.random.random(9)).reshape(3,3)
    test_vec0 = (np.ones(3*N)-2.*np.random.random(3*N)).reshape(3, N)
    test_vec  = np.dot(np.eye(3)+test_f, test_vec0)

    # make a mixed set of scattering vectors
    from daxmexplorer.vecmath import normalize
    test_vec[:, 1:N/2] = normalize(test_vec[:, 1:N/2], axis=0)

    from daxmexplorer import cm
    deviator = cm.get_deviatoric_defgrad

    daxmVoxel = DAXMvoxel(name='Pooh',
                          ref_frame='APS',
                          coords=np.ones(3),
                          pattern_image='hidden',
                          scatter_vec=test_vec,
                          plane=test_vec0,
                          recip_base=np.eye(3),
                          peak=np.random.random((2, 10)),
                         )

    print("dev_correct\n", deviator(np.eye(3)+test_f))
    print('dev_L2\n', deviator(daxmVoxel.deformation_gradientL2()))
    print('delta_L2', np.linalg.norm(deviator(np.eye(3)+test_f) - deviator(daxmVoxel.deformation_gradientL2()) ) )
    print('F_opt\n', daxmVoxel.deformation_gradient_opt(eps=test_eps))
    print('dev_opt\n', deviator(daxmVoxel.deformation_gradient_opt(eps=test_eps)))
    print('delta_opt', np.linalg.norm(deviator(np.eye(3)+test_f) - deviator(daxmVoxel.deformation_gradient_opt(eps=1e-1)) ) )
    print('deltaFull_opt', np.linalg.norm(np.eye(3)+test_f - daxmVoxel.deformation_gradient_opt(eps=1e-1) ) )

    # write to single file
    print (daxmVoxel)
    daxmVoxel.write(h5file='dummy_data.h5')

    # read from file 
    daxmVoxel = DAXMvoxel()
    daxmVoxel.read('dummy_data.h5', 'Pooh')
#    daxmVoxel.name = 'copy'
    print(daxmVoxel)
