#!/usr/bin/env python

from __future__ import print_function
import h5py
import numpy as np
import sys
from daxmexplorer.vecmath import normalize
from daxmexplorer.cxtallite import OrientationMatrix

class DAXMvoxel(object):
    """
    DAXM voxel stores the crystallograhic information derived from DAXM indexation results.
    By default, all data is recoreded in the APS coordinate system.
    Coordinate system transformation is done via binded method.

    NOTE:
    (a,b) -> a and b column stacked
    (a;b) -> a and b row stacked

    @para:
    name:             voxel ID, used as the group name in HDF5 archive
    ref_frame:        reference frame, by default using "APS"
    coords:           voxel position
    pattern_image:    associated reconstructed micro-Laue diffraction image name (H5)
    scatter_vec:      measured scattering vectors (qx;qy;qz)
    plane:            Miller index of indexed planes (h;k;l)
    recip_base:       reciprocal base of the voxel (a*,b*,c*)
    peak:             diffraction peak coordinates on CCD(x;y)
    depth:            wire position
    lattice_constant: lattice constant

    """
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
                 lattice_constant=np.random.random(6),
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
        self.lattice_constant = lattice_constant
        self.opt_rst = None
        self.strain = None

    def __repr__(self):
        return '\n'.join([
          'name: {}'.format(self.name),
          'frame: {}'.format(self.ref_frame),
          'coords: {}'.format(self.coords),
          'image: {}'.format(self.pattern_image),
        ])

    @property
    def eulers(self):
        """ Calculate the Bunge Euler angle representation"""
        astar = self.recip_base[:, 0]
        bstar = self.recip_base[:, 1]
        cstar = self.recip_base[:, 2]
        # calcualte the real base
        c = normalize(np.cross(astar, bstar))
        a = normalize(np.cross(bstar, cstar))
        b = normalize(np.cross(c, a))
        # get the orientation matrix representation
        g = np.column_stack((a, b, c))
        return OrientationMatrix(g).toEulers()


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

            self.coords                  = get_data(thisvoxel, 'coords')
            self.scatter_vec             = get_data(thisvoxel, 'scatter_vec')
            self.plane                   = get_data(thisvoxel, 'plane')
            self.recip_base              = get_data(thisvoxel, 'recip_base')
            self.peak                    = get_data(thisvoxel, 'peak')
            self.depth                   = get_data(thisvoxel, 'depth')
            self.lattice_constant        = get_data(thisvoxel, 'lattice_constant')

            if "{}/{}".format(thisvoxel, 'strain') in h5f.keys():
                self.strain = get_data(thisvoxel, 'strain')

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
            h5f.create_dataset("{}/lattice_constant".format(self.name), data=self.lattice_constant)

            if self.strain is not None:
                h5f.create_dataset("{}/strain".format(self.name), data=self.strain)

            h5f[self.name].attrs['pattern_image'] = self.pattern_image
            h5f[self.name].attrs['ref_frame']     = self.ref_frame
            h5f[self.name].attrs['voxelStatus']   = voxelStatus


            h5f.flush()

    def scatter_vec0(self, match_measured=False):
        """return the strain-free scattering vectors calculated from hkl index"""
        q0 = np.dot(self.recip_base, self.plane)
        if match_measured:
            idx_unit_q = np.where(np.absolute(np.linalg.norm(self.scatter_vec, axis=0) - 1) <= 1e-4)
            q0[:, idx_unit_q] /= np.linalg.norm(q0[:, idx_unit_q], axis=0)

        return q0

    def toFrame(self, target=None):
        """transfer reference frame with given orientation matrix, g"""
        g_to_from = self.g_to_from
        if target is None: return
        if target not in g_to_from: raise Exception
        # NOTE: g matrix represents passive rotation

        # convert coordinates
        self.coords = np.dot(g_to_from[target][self.ref_frame], self.coords)

        # convert scattering vectors
        self.scatter_vec = np.dot(g_to_from[target][self.ref_frame], self.scatter_vec)

        # convert reciprocal base
        self.recip_base = np.dot(g_to_from[target][self.ref_frame], self.recip_base)

        self.ref_frame = target

    def deformation_gradientL2(self):
        """extract lattice deformation gradient using least-squares regression(L2 optimization)"""
        # quick summary of the least square solution
        # F* q0 = q
        # ==> F* q0 q0^T = q q0^T
        # ==> F* = (q q0^T)(q0 q0^T)^-1
        #              A       B
        q0 = self.scatter_vec0(match_measured=True)
        q = self.scatter_vec

        A = np.dot(q, q0.T)
        B = np.dot(q0, q0.T)
        # Fstar = np.dot(A, np.linalg.pinv(B))
        # F = F*^(-T) = A^-T B^T
        # inverting B can be dangerous
        return np.dot(np.linalg.inv(A).T, B.T)

    def deformation_gradient_opt(self, 
                                 eps=1e-1, 
                                 tol=1e-14,
                                 maxiter=5e6):
        """extract lattice deformation gardient using nonlinear optimization"""
        # NOTE: a large bound guess is better than a smaller bound

        def constraint(constraint_f, e):
            return len(constraint_f)*e - np.sum(np.abs(constraint_f))

        def objectiveIce(f, vec0, vec):
            estimate = np.dot(np.eye(3)+f.reshape(3, 3), vec0)
            return np.sum(1.0 - np.einsum('ij,ij->j',
                                          vec/np.linalg.norm(vec, axis=0),
                                          estimate/np.linalg.norm(estimate, axis=0),
                                         )
                         )
        
        def objective_rmsNorm(f, vec0, vec):
            # NOTE:
            # The threshold here cannot be too tight
            idx_unit_q = np.where(np.absolute(np.linalg.norm(vec,axis=0) - 1.0) < 1e-4)

            # NOTE:
            # An objective function should remain pure:
            # do not modify input, work with its copy
            vec0_matched = np.copy(vec0)
            # the normalization here might not be necessary
            vec0_matched[:,idx_unit_q] /= np.linalg.norm(vec0_matched[:,idx_unit_q], axis=0)

            estimate = np.dot(np.eye(3)+f.reshape(3,3), vec0_matched)
            estimate[:,idx_unit_q] /= np.linalg.norm(estimate[:,idx_unit_q], axis=0)
            return np.sqrt(np.mean(np.square(np.linalg.norm(vec-estimate,axis=0)/np.linalg.norm(vec,axis=0))))

        def objective_smrNorm(f, vec0, vec):
            # NOTE:
            # The threshold here cannot be too tight
            idx_unit_q = np.where(np.absolute(np.linalg.norm(vec,axis=0) - 1.0) < 1e-4)

            # NOTE:
            # An objective function should remain pure:
            # do not modify input, work with its copy
            vec0_matched = np.copy(vec0)
            # the normalization here might not be necessary
            vec0_matched[:,idx_unit_q] /= np.linalg.norm(vec0_matched[:,idx_unit_q], axis=0)

            estimate = np.dot(np.eye(3)+f.reshape(3,3), vec0_matched)
            estimate[:,idx_unit_q] /= np.linalg.norm(estimate[:,idx_unit_q], axis=0)
            return np.square(np.mean(np.sqrt(np.linalg.norm(vec-estimate,axis=0)/np.linalg.norm(vec,axis=0))))
    
        def objectiveDante(f, vec0, vec):
            estimate = np.dot(np.eye(3)+f.reshape(3, 3), vec0)

            # angular difference
            angdiff = vec/np.linalg.norm(vec,axis=0) - estimate/np.linalg.norm(estimate,axis=0)
            angdiff = np.sqrt(np.mean(np.sum(np.square(angdiff), axis=0)))

            # length difference
            idx_full_q = np.where(np.absolute(np.linalg.norm(vec,axis=0) - 1) > 1e-10)
            lendiff = np.linalg.norm(estimate[:, idx_full_q],axis=0) / np.linalg.norm(vec[:, idx_full_q],axis=0)
            lendiff = np.sqrt(np.mean(np.square(np.log(lendiff))))

            return angdiff + lendiff

        import scipy.optimize

        q0_opt = self.scatter_vec0()
        q_opt  = self.scatter_vec

        self.opt_rst = scipy.optimize.minimize(objective_rmsNorm,
                                               x0 = np.zeros(3*3),
                                               args = (q0_opt,q_opt),
                                        #   method = 'Nelder-mead',  # demo error ~ 1e-14
                                        #   method = 'BFGS',         # demo error ~ 1e-8 
                                               method = 'COBYLA',       # demo error ~ 1e-14
                                               tol = tol,
                                               constraints = {'type':'ineq',
                                                              'fun': lambda x: constraint(x,eps),
                                                             },
                                               options={'maxiter':int(maxiter),
                                                       },
                                                )
        # print(self.opt_rst)
        fstar = np.eye(3) + self.opt_rst.x.reshape(3,3)
        return np.transpose(np.linalg.inv(fstar))

    def pair_scattervec_plane(self):
        """pair the recorded scattering vectors and the indexation results"""
        old_scatter_vec = np.array(self.scatter_vec)

        if self.peak.shape[0] < old_scatter_vec.shape[0]:
            old_peaks = np.zeros((2, self.scatter_vec.shape[1]))
        else:
            old_peaks = np.array(self.peak)

        new_scatter_vec = np.zeros(self.plane.shape)
        new_peak = np.zeros((2, self.plane.shape[1]))

        qs = normalize(old_scatter_vec, axis=0)   # normalize each scatter vector (column stacked)
        q0 = normalize(np.dot(self.recip_base, self.plane), axis=0)

        for i in range(self.plane.shape[1]):
            angular_diff = np.absolute(1.0 - np.dot(q0[:, i].T, qs))
            # pair q0 and qs with the smallest angular difference
            idx = np.argmin(angular_diff)
            new_scatter_vec[:, i] = old_scatter_vec[:, idx]
            new_peak[:, i] = old_peaks[:, idx]

            # remove the paired entry
            qs = np.delete(qs, idx, axis=1)
            old_scatter_vec = np.delete(old_scatter_vec, idx, axis=1)
            old_peaks = np.delete(old_peaks, idx, axis=1)

        # update scatter vectors
        self.scatter_vec = new_scatter_vec
        self.peak = new_peak

        return None


if __name__ == "__main__":
    import sys

    # ----- strain quantification demo ----- #
    # test the accuracy of extracted lattice deformation gradient
    N = 30  # n_indexedPeaks
    n = 0   # n_fullq
    test_eps = 1e-2  # strain level (ish)
    # test_eps = 0

    test_df = test_eps*(np.ones(9)-2.*np.random.random(9)).reshape(3,3)  # F - I
    test_f = np.eye(3) + test_df
    test_fstar = np.transpose(np.linalg.inv(test_f))

    test_recip_base = np.eye(3) * 1.55
    print("reciprocal base:\n", test_recip_base)

    tmpidx = np.arange(-10, 10)
    tmpidx = np.delete(tmpidx, 10)
    test_plane = np.random.choice(tmpidx, N*3, replace=True).reshape(3, N)
    print("hkl index:\n", test_plane, "\n")

    test_vec0 = np.dot(test_recip_base, test_plane)
    test_vec = np.dot(test_fstar, test_vec0)  # measured strained scattering vectors
    test_vec[:, n:] /= np.linalg.norm(test_vec[:, n:], axis=0)

    print("mimic shuffling of q vectors at APS")
    print("ordered q:\n", test_vec[:, :5])
    test_vec = test_vec[:, np.random.permutation(test_vec.shape[1])]
    print("unordered q in xml file:\n", test_vec[:, :5])

    daxmVoxel = DAXMvoxel(name='Cloud',
                          ref_frame='APS',
                          coords=np.ones(3),
                          pattern_image='hidden',
                          scatter_vec=test_vec,
                          plane=test_plane,
                          recip_base=test_recip_base,
                          peak=np.random.random((2, N)),
                         )

    daxmVoxel.pair_scattervec_plane()
    print("reordered q:\n", daxmVoxel.scatter_vec[:, :5])
    print("test pairing complete.\n")

    from daxmexplorer.cm import get_deviatoric_defgrad
    deviator = get_deviatoric_defgrad

    # ----- L2 method ----- #
    test_f_L2 = daxmVoxel.deformation_gradientL2()

    print("F correct\n", test_f)
    print("F L2\n", test_f_L2)
    print("\t-->with error:{}".format(np.linalg.norm(test_f - test_f_L2)))
    print("-"*20)
    print("F_D correct\n", deviator(test_f))
    print("F_D L2\n", deviator(test_f_L2))
    print("\t-->with error:{}".format(np.linalg.norm(deviator(test_f) - deviator(test_f_L2))))
    print("="*20 + "\n")

    # ----- opt method ----- #
    test_f_opt = daxmVoxel.deformation_gradient_opt()
    print(daxmVoxel.opt_rst, "\n")

    print("F correct\n", test_f)
    print("F opt\n", test_f_opt)
    print("\t-->with error:{}".format(np.linalg.norm(test_f - test_f_opt)))
    print("-"*20)
    print("F_D correct\n", deviator(test_f)-np.eye(3))
    print("F_D opt\n", deviator(test_f_opt)-np.eye(3))
    print("\t-->with error:{}".format(np.linalg.norm(deviator(test_f) - deviator(test_f_opt))))
    print("="*20 + "\n")

    # ----- HDF5 support demo ----- #
    # write and read data to HDF5 archive
    daxmVoxel.write(h5file='dummy_data.h5')
    print("export DAXM voxel\n", daxmVoxel, "\n")
    daxmVoxel = DAXMvoxel()
    daxmVoxel.read('dummy_data.h5', 'Cloud')
    daxmVoxel.name = 'Sephiroth'
    print("read in Cloud, change to\n", daxmVoxel)
    print(daxmVoxel.recip_base)
    print(daxmVoxel.eulers)
