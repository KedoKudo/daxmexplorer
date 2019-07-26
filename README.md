# DAXMexplorer

This is a package developed for processing indexation results from __beamline 34-ID-E__ from __APS__.

## Installation

### Required third-party packages 

__DAXMexplorer__ uses _Cython_ to compile a fast module for crystallographic calculations. 
After cloning and add the repository to your _PYTHON\_PATH_, please install Cython through 

```sh
pip install Cython
```

for system Python, or use the following command if using Anaconda Python. 

```sh
conda install Cython
```

### Compile with GCC

Use the following comand to compile 

```sh
make all
```

Now you should be able to use daxmexplorer to process your data.

## Sample usage

See `examples/virtualDAXM.py` for usage example.
