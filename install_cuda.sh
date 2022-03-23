wget https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda_11.1.1_455.32.00_linux.run
bash cuda_11.1.1_455.32.00_linux.run --no-drm --no-man-page --override --installpath=~/local --librarypath=~/local/lib --toolkitpath=~/local/cuda-11.1/ --toolkit --silent
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/local/cuda-11.1/lib64/" >> ~/.bashrc
echo "export PATH=$PATH:~/local/cuda-11.1/bin/" >> ~/.bashrc
source ~/.bashrc
