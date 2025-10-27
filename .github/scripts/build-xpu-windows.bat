
set INTEL_DLE_URL=https://registrationcenter-download.intel.com/akdlm/IRC_NAS/75d4eb97-914a-4a95-852c-7b9733d80f74/intel-deep-learning-essentials-2025.1.3.8_offline.exe

echo ::group::Debugging Information
echo Current Directory: %CD%
set
echo ::endgroup::

curl -o intel-dle-installer.exe %INTEL_DLE_URL%

echo ::group::Intel Deep Learning Essentials Installation
start /wait "Intel DLE Install" intel-dle-installer.exe -r yes --log intel_dle_log.txt --silent -a  --eula=accept -p=NEED_VS2022_INTEGRATION=0
type intel_dle_log.txt
if ERRORLEVEL 1 (
    echo Failed to install Intel Deep Learning Essentials
    exit /b 1
)
echo ::endgroup::

echo ::group::Environment Setup
call "%ProgramFiles(x86)%\Intel\oneAPI\setvars.bat"
if ERRORLEVEL 1 (
    echo Failed to setup environment
    exit /b 1
)
echo ::endgroup::

echo ::group::Building with XPU backend
cmake -G Ninja -DCOMPUTE_BACKEND=xpu -DCMAKE_BUILD_TYPE=Release .
cmake --build . --config Release
echo ::endgroup::

if ERRORLEVEL 1 (
    echo Build failed
    exit /b 1
)

set output_dir=output\%build_os%\x86_64
if not exist "%output_dir%" mkdir "%output_dir%"
copy bitsandbytes\*.dll "%output_dir%\" 2>nul
