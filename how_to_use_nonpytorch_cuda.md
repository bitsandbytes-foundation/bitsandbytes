## How to use a CUDA version that is different from PyTorch

Some features of bitsandbytes may need a newer CUDA version than regularly supported by PyTorch binaries from conda / pip. In that case you can use the following instructions to load a precompiled bitsandbytes binary that works for you.

## Installing or determining the CUDA installation

Determine the path of the CUDA version that you want to use. Common paths paths are:
```bash
/usr/local/cuda
/usr/local/cuda-XX.X
```

where XX.X is the CUDA version number.

You can also install CUDA version that you need locally with a script provided by bitsandbytes as follows:

```bash
wget https://raw.githubusercontent.com/TimDettmers/bitsandbytes/main/cuda_install.sh
# Syntax cuda_install CUDA_VERSION INSTALL_PREFIX EXPORT_TO_BASH
#   CUDA_VERSION in {110, 111, 112, 113, 114, 115, 116, 117, 118, 120, 121, 122}
#   EXPORT_TO_BASH in {0, 1} with 0=False and 1=True 

# For example, the following installs CUDA 11.7 to ~/local/cuda-11.7 and exports the path to your .bashrc
bash cuda install 117 ~/local 1 
```

## Setting the environmental variables BNB_CUDA_VERSION, and LD_LIBRARY_PATH

To manually override the PyTorch installed CUDA version you need to set to variable, like so:

```bash
export BNB_CUDA_VERSION=<VERSION>
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<PATH>
```

For example, to use the local install path from above:

```bash
export BNB_CUDA_VERSION=117
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/tim/local/cuda-11.7
```

It is best to add these lines to the `.bashrc` file to make them permanent.

If you now launch bitsandbytes with these environmental variables the PyTorch CUDA version will be overridden by the new CUDA version and a different bitsandbytes library is loaded (in this case version 117).
