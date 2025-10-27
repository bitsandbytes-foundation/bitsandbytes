
set INTEL_DLE_URL=https://registrationcenter-download.intel.com/akdlm/IRC_NAS/75d4eb97-914a-4a95-852c-7b9733d80f74/intel-deep-learning-essentials-2025.1.3.8_offline.exe

curl -o intel-dle-installer.exe %INTEL_DLE_URL%
start /wait "Intel DLE Install" intel-dle-installer.exe -r -a --eula=accept -p=NEED_VS2022_INTEGRATION=0 > intel_dle_log.txt 2>&1
type intel_dle_log.txt

if ERRORLEVEL 1 (
    echo Failed to install Intel Deep Learning Essentials
    exit /b 1
)

call "%ProgramFiles(x86)%\Intel\oneAPI\setvars.bat"
if ERRORLEVEL 1 (
    echo Failed to setup environment
    exit /b 1
)

cmake -G Ninja -DCOMPUTE_BACKEND=xpu -DCMAKE_BUILD_TYPE=Release .
cmake --build . --config Release

if ERRORLEVEL 1 (
    echo Build failed
    exit /b 1
)

set output_dir=output\%build_os%\x86_64
if not exist "%output_dir%" mkdir "%output_dir%"
copy bitsandbytes\*.dll "%output_dir%\" 2>nul
