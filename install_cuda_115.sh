wget wget https://developer.download.nvidia.com/compute/cuda/11.5.1/local_installers/cuda_11.5.1_495.29.05_linux.run 
bash cuda_11.5.1_495.29.05_linux.run --no-drm --no-man-page --override --installpath=~/local --librarypath=~/local/lib --toolkitpath=~/local/cuda-11.5/ --toolkit --silent
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/local/cuda-11.5/lib64/" >> ~/.bashrc
echo "export PATH=$PATH:~/local/cuda-11.5/bin/" >> ~/.bashrc
source ~/.bashrc
