SET bitsandbytesroot=%cd%

conda env create -f environment.yml
conda activate 8-bit

mkdir dependencies
cd dependencies
git clone https://github.com/GerHobbelt/pthread-win32
cd pthread-win32
mkdir build
cd build
cmake ..
cmake --build ./ -j4 --config Release


cd %bitsandbytesroot%
mkdir build
cd build
cmake ..
cmake --build ./ -j4 --config Release




Known compile warnings:
C:\bitsandbytes\csrc\ops.cu(480): warning : variable "tilesize" was declared but never referenced [C:\bitsandbytes\build\csrc\bitsandbytes.vcxproj] - Safe to ignore.
CUDACOMPILE : ptxas warning : Value of threads per SM for entry _Z9kQuantizePfS_Phi is out of range. .minnctapersm will be ignored [C:\bitsandbytes\build\csrc\bitsandbytes.vcxproj]




