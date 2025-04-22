URL118=https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
URL120=https://developer.download.nvidia.com/compute/cuda/12.0.1/local_installers/cuda_12.0.1_525.85.12_linux.run
URL121=https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_530.30.02_linux.run
URL122=https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda_12.2.2_535.104.05_linux.run
URL123=https://developer.download.nvidia.com/compute/cuda/12.3.2/local_installers/cuda_12.3.2_545.23.08_linux.run
URL124=https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_550.54.15_linux.run
URL125=https://developer.download.nvidia.com/compute/cuda/12.5.1/local_installers/cuda_12.5.1_555.42.06_linux.run
URL126=https://developer.download.nvidia.com/compute/cuda/12.6.2/local_installers/cuda_12.6.2_560.35.03_linux.run

CUDA_VERSION=$1
BASE_PATH=$2
EXPORT_BASHRC=$3

if [[ -n "$CUDA_VERSION" ]]; then
  if [[ "$CUDA_VERSION" -eq "118" ]]; then
    URL=$URL118
    FOLDER=cuda-11.8
  elif [[ "$CUDA_VERSION" -eq "120" ]]; then
    URL=$URL120
    FOLDER=cuda-12.0
  elif [[ "$CUDA_VERSION" -eq "121" ]]; then
    URL=$URL121
    FOLDER=cuda-12.1
  elif [[ "$CUDA_VERSION" -eq "122" ]]; then
    URL=$URL122
    FOLDER=cuda-12.2
  elif [[ "$CUDA_VERSION" -eq "123" ]]; then
    URL=$URL123
    FOLDER=cuda-12.3
  elif [[ "$CUDA_VERSION" -eq "124" ]]; then
    URL=$URL124
    FOLDER=cuda-12.4
  elif [[ "$CUDA_VERSION" -eq "125" ]]; then
    URL=$URL125
    FOLDER=cuda-12.5
  elif [[ "$CUDA_VERSION" -eq "126" ]]; then
    URL=$URL126
    FOLDER=cuda-12.6
  else
    echo "argument error: No cuda version passed as input. Choose among versions 118 to 126"
  fi
else
    echo "argument error: No cuda version passed as input. Choose among versions 118 to 126"
fi

FILE=$(basename $URL)

if [[ -n "$CUDA_VERSION" ]]; then
  echo $URL
  echo $FILE
  wget $URL
  bash $FILE --no-drm --no-man-page --override --toolkitpath=$BASE_PATH/$FOLDER/ --toolkit --silent
  if [ "$EXPORT_BASHRC" -eq "1" ]; then
    echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$BASE_PATH/$FOLDER/lib64" >> ~/.bashrc
    echo "export PATH=\$PATH:$BASE_PATH/$FOLDER/bin" >> ~/.bashrc
    source ~/.bashrc
  fi
else
  echo ""
fi
