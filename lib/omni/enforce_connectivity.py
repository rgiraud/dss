import glob 
import numpy as np

#Compile enforce_connectivity cython code
enforce_connectivity_cython_so_name = glob.glob("./lib/omni/_enforce_connectivity*.so")
if (len(enforce_connectivity_cython_so_name)==0):
    from setuptools import setup
    from Cython.Build import cythonize
    import shutil
    setup(
        ext_modules = cythonize("lib/omni/_enforce_connectivity.pyx"), 
        include_dirs=[np.get_include()],
        script_args=['build_ext', '--inplace']
    )
    shutil.move(glob.glob("_enforce_connectivity*.so")[0], "./lib/omni")

