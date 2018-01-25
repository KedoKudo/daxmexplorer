#!/usr/bin/env python

"""
Parse DAXM indexation results (xml) to instances of daxm voxel
objects.
The script can be envoked as individual programe via CMD or imported
as a parser func for interactive data analysis.
"""

import numpy as np
import xml.etree.cElementTree as ET
from daxm_analyzer.daxm_voxel import DAXMvoxel


def parse_xml(xmlfile,
              namespace={'step':'http://sector34.xray.aps.anl.gov/34ide:indexResult'}, 
              h5file=None):
    voxels = []
    tree = ET.parse(xmlfile)
    root = tree.getroot()
    ns = namespace

    for i in range(len(root)):
        step = root[i]
        # locate next indexed voxel in xml file
        target_str = 'step:indexing/step:pattern/step:recip_lattice/step:astar'
        astar = step.find(target_str, ns)
        if astar is None:
            continue

        # get coords 
        xsample = float(step.find('step:Xsample', ns).text)
        ysample = float(step.find('step:Ysample', ns).text)
        zsample = float(step.find('step:Zsample', ns).text)
        depth = float(step.find('step:depth', ns).text)
        coords = np.array([xsample, ysample, -zsample+depth])

        # get pattern image name
        h5img = step.find('step:detector/step:inputImage', ns).text

        # get peaks
        xpix = step.find('step:detector/step:peaksXY/step:Xpixel', ns).text
        ypix = step.find('step:detector/step:peaksXY/step:Ypixel', ns).text
        xpix = np.nan if xpix is None else map(float, xpix.split())
        ypix = np.nan if ypix is None else map(float, ypix.split())
        peaks = np.stack((xpix, ypix))

        # get scattering vectors 
        qx = step.find('step:detector/step:peaksXY/step:Qx', ns).text
        qy = step.find('step:detector/step:peaksXY/step:Qy', ns).text
        qz = step.find('step:detector/step:peaksXY/step:Qz', ns).text
        qx = map(float, qx.split())
        qy = map(float, qy.split())
        qz = map(float, qz.split())
        scatter_vecs = np.stack((qx, qy, qz))

        # get reciprocal base (a*, b*, c*)
        astar_str = 'step:indexing/step:pattern/step:recip_lattice/step:astar'
        bstar_str = 'step:indexing/step:pattern/step:recip_lattice/step:bstar'
        cstar_str = 'step:indexing/step:pattern/step:recip_lattice/step:cstar'
        astar = map(float, step.find(astar_str, ns).text.split())
        bstar = map(float, step.find(bstar_str, ns).text.split())
        cstar = map(float, step.find(cstar_str, ns).text.split())
        recip_base = np.column_stack((astar, bstar, cstar))

        # get plane index (hkl)
        h = step.find('step:indexing/step:pattern/step:hkl_s/step:h', ns).text
        k = step.find('step:indexing/step:pattern/step:hkl_s/step:k', ns).text
        l = step.find('step:indexing/step:pattern/step:hkl_s/step:l', ns).text
        h = map(float, h.split())
        k = map(float, k.split())
        l = map(float, l.split())
        plane = np.stack((h, k, l))

        # create the DAXM voxel 
        tmpvoxel = DAXMvoxel(ref_frame='APS',
                             coords=coords,
                             pattern_image=h5img,
                             scatter_vecs=scatter_vecs,
                             plane=plane,
                             recip_base=recip_base,
                             peaks=peaks,
                             depth=depth,
                            )

        # pair scattering vectors with plane index
        tmpvoxel.pair_scattervec_plane()
      
        if h5file is not None:
            tmpvoxel.write(h5file=h5file)
     
        voxels.append(tmpvoxel)

    return voxels


if __name__ == "__main__":
    import sys
    xmlfile = sys.argv[1]
    print("parsing xml file: {}".format(xmlfile))

    h5file = xmlfile.replace(".xml", ".h5")

    parse_xml(xmlfile, h5file=h5file)
