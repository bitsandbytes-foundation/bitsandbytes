URL110=https://developer.download.nvidia.com/compute/cuda/11.0.3/local_installers/cuda_11.0.3_450.51.06_linux.run
URL111=https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda_11.1.1_455.32.00_linux.run
URL112=https://developer.download.nvidia.com/compute/cuda/11.2.2/local_installers/cuda_11.2.2_460.32.03_linux.run
URL113=https://developer.download.nvidia.com/compute/cuda/11.3.1/local_installers/cuda_11.3.1_465.19.01_linux.run
URL114=https://developer.download.nvidia.com/compute/cuda/11.4.4/local_installers/cuda_11.4.4_470.82.01_linux.run
URL115=https://developer.download.nvidia.com/compute/cuda/11.5.2/local_installers/cuda_11.5.2_495.29.05_linux.run
URL116=https://developer.download.nvidia.com/compute/cuda/11.6.2/local_installers/cuda_11.6.2_510.47.03_linux.run
URL117=https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/cuda_11.7.1_515.65.01_linux.run
URL118=https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
URL120=https://developer.download.nvidia.com/compute/cuda/12.0.1/local_installers/cuda_12.0.1_525.85.12_linux.run
URL121=https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_530.30.02_linux.run
URL122=https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda_12.2.2_535.104.05_linux.run
URL123=https://developer.download.nvidia.com/compute/cuda/12.3.2/local_installers/cuda_12.3.2_545.23.08_linux.run
URL124=https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_550.54.15_linux.run
URL125=https://developer.download.nvidia.com/compute/cuda/12.5.0/local_installers/cuda_12.5.0_555.42.02_linux.run

CUDA_VERSION=$1
BASE_PATH=$2
EXPORT_BASHRC=$3

if [[ -n "$CUDA_VERSION" ]]; then
  if   [[ "$CUDA_VERSION" -eq "110" ]]; then
    URL=$URL110
    FOLDER=cuda-11.0
  elif [[ "$CUDA_VERSION" -eq "111" ]]; then
    URL=$URL111
    FOLDER=cuda-11.1
  elif [[ "$CUDA_VERSION" -eq "112" ]]; then
    URL=$URL112
    FOLDER=cuda-11.2
  elif [[ "$CUDA_VERSION" -eq "113" ]]; then
    URL=$URL113
    FOLDER=cuda-11.3
  elif [[ "$CUDA_VERSION" -eq "114" ]]; then
    URL=$URL114
    FOLDER=cuda-11.4
  elif [[ "$CUDA_VERSION" -eq "115" ]]; then
    URL=$URL115
    FOLDER=cuda-11.5
  elif [[ "$CUDA_VERSION" -eq "116" ]]; then
    URL=$URL116
    FOLDER=cuda-11.6
  elif [[ "$CUDA_VERSION" -eq "117" ]]; then
    URL=$URL117
    FOLDER=cuda-11.7
  elif [[ "$CUDA_VERSION" -eq "118" ]]; then
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
  else
    echo "argument error: No cuda version passed as input. Choose among versions 110 to 125"
  fi
else
    echo "argument error: No cuda version passed as input. Choose among versions 110 to 125"
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
