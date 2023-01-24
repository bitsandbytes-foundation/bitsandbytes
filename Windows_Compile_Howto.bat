SET bitsandbytesroot=%cd%

Rem conda env create -f environment.yml
Rem conda activate 8-bit

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
Rem Note: You can target a specific cuda version assuming you have it installed with visual studio integration with: ```cmake .. -T cuda=12.0 ```
Rem instead of ```cmake ..```
cmake ..
cmake --build ./ -j4 --config Release




Rem Known compile warnings:
Rem C:\bitsandbytes\csrc\ops.cu(480): warning : variable "tilesize" was declared but never referenced [C:\bitsandbytes\build\csrc\bitsandbytes.vcxproj] - Safe to ignore.
Rem CUDACOMPILE : ptxas warning : Value of threads per SM for entry _Z9kQuantizePfS_Phi is out of range. .minnctapersm will be ignored [C:\bitsandbytes\build\csrc\bitsandbytes.vcxproj]




