# Installs the CUDA 13.4 preview toolkit for Windows arm64.
$ErrorActionPreference = "Stop"

$url = "https://packages.nvidia.com/prerelease/cuda/13.4.0/local_installers/cuda_13.4.0_windows_arm64.exe"
$subPackages = @(
    "nvcc_13.4", "crt_13.4", "nvvm_13.4", "nvptxcompiler_13.4",
    "cudart_13.4", "cublas_13.4", "cublas_dev_13.4", "thrust_13.4"
)
$cudaPath = "$env:ProgramFiles\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
$installer = Join-Path $env:RUNNER_TEMP "cuda_13.4.0.exe"

Write-Host "Downloading $url"
curl.exe --fail --location --retry 3 --output $installer $url
if ($LASTEXITCODE -ne 0) { throw "Failed to download $url" }

Write-Host "Installing subpackages: $($subPackages -join ' ')"
$proc = Start-Process -FilePath $installer -ArgumentList (@("-s", "-n") + $subPackages) -Wait -PassThru
Remove-Item $installer -Force
if ($proc.ExitCode -ne 0) { throw "CUDA installer exited with code $($proc.ExitCode)" }

if (-not (Test-Path "$cudaPath\bin\nvcc.exe")) { throw "nvcc not found under $cudaPath" }

Write-Host "Installed CUDA to $cudaPath"
& "$cudaPath\bin\nvcc.exe" --version

"CUDA_PATH=$cudaPath" | Out-File $env:GITHUB_ENV -Append
"$cudaPath\bin" | Out-File $env:GITHUB_PATH -Append
