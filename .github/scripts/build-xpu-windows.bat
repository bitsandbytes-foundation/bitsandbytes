set INTEL_DLE_URL=https://registrationcenter-download.intel.com/akdlm/IRC_NAS/75d4eb97-914a-4a95-852c-7b9733d80f74/intel-deep-learning-essentials-2025.1.3.8_offline.exe
set INTEL_DLE_TMP=%RUNNER_TEMP%\intel_dle
set INTEL_DLE_LOG=%RUNNER_TEMP%\intel_dle_log.txt

echo ::group::Intel Deep Learning Essentials Installation
curl -o intel-dle-installer.exe %INTEL_DLE_URL%
start /wait "Intel DLE Install" intel-dle-installer.exe -f %INTEL_DLE_TMP% -l %INTEL_DLE_LOG% --silent -a --eula=accept -p=NEED_VS2022_INTEGRATION=0
type %INTEL_DLE_LOG%
if ERRORLEVEL 1 (
    echo Failed to install Intel Deep Learning Essentials
    exit /b 1
)
echo ::endgroup::

echo ::group::Build Environment Setup
call "%ProgramFiles(x86)%\Intel\oneAPI\setvars.bat"
cmake -G Ninja -DCOMPUTE_BACKEND=xpu -DCMAKE_BUILD_TYPE=Release .
if ERRORLEVEL 1 (
    echo Failed to setup environment
    exit /b 1
)
echo ::endgroup::

echo ::group::Building with XPU backend
cmake --build . --config Release
if ERRORLEVEL 1 (
    echo Build failed
    exit /b 1
)
echo ::endgroup::

set output_dir=output\%build_os%\x86_64
if not exist "%output_dir%" mkdir "%output_dir%"
copy bitsandbytes\*.dll "%output_dir%\" 2>nul
