FILE115=:cuda_11.5.1_495.29.05_linux.run
FILE111=:cuda_11.1.1_455.32.00_linux.run
URL115=:https://developer.download.nvidia.com/compute/cuda/11.5.1/local_installers/cuda_11.5.1_495.29.05_linux.run
URL111=:https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda_11.1.1_455.32.00_linux.run


CUDA_VERSION=$1

if [[ -n "$CUDA_VERSION" ]]; then
  if   [[ "$CUDA_VERSION" -eq "111" ]]; then
    FILE=cuda_11.1.1_455.32.00_linux.run
    URL=https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda_11.1.1_455.32.00_linux.run
    FOLDER=cuda-11.1
  elif [[ "$CUDA_VERSION" -eq "115" ]]; then
    FILE=cuda_11.5.1_495.29.05_linux.run
    URL=https://developer.download.nvidia.com/compute/cuda/11.5.1/local_installers/cuda_11.5.1_495.29.05_linux.run
    FOLDER=cuda-11.5
  else
    echo "argument error: No cuda version passed as input. Choose among: {111, 115}"
  fi
else
    echo "argument error: No cuda version passed as input. Choose among: {111, 115}"
fi

if [[ -n "$CUDA_VERSION" ]]; then
  echo $URL
  echo $FILE
  wget $URL
  bash $FILE --no-drm --no-man-page --override --installpath=~/local --librarypath=~/local/lib --toolkitpath=~/local/$FOLDER/ --toolkit --silent
  echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/local/$FOLDER/lib64/" >> ~/.bashrc
  echo "export PATH=$PATH:~/local/$FOLDER/bin/" >> ~/.bashrc
  source ~/.bashrc
else
  echo ""
fi



