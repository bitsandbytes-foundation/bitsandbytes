# Installs the CUDA Toolkit subpackages needed to build bitsandbytes on a Windows x64 or ARM64 runner,
# using NVIDIA's network installer.
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern("^\d+\.\d+\.\d+$")]
    [string]$CudaVersion,
    [string[]]$ExtraSubPackages = @()
)

$ErrorActionPreference = "Stop"

$version = [version]$CudaVersion
$cudaShortVersion = "$($version.Major).$($version.Minor)"
$isArm64 = $env:RUNNER_ARCH -eq "ARM64"

# From 13.4 the installer name includes the architecture; ARM64 installers start at 13.4.
if ($version -ge [version]"13.4") {
    $arch = if ($isArm64) { "arm64" } else { "x86_64" }
    $installerName = "cuda_${CudaVersion}_windows_${arch}_network.exe"
} elseif ($isArm64) {
    throw "CUDA $CudaVersion has no Windows ARM64 installer (13.4 or newer is required)"
} else {
    $installerName = "cuda_${CudaVersion}_windows_network.exe"
}
$url = "https://developer.download.nvidia.com/compute/cuda/$CudaVersion/network_installers/$installerName"

$packages = @("nvcc", "cudart", "cublas", "cublas_dev")
# CUDA 13 split crt, nvvm and nvptxcompiler out of nvcc.
if ($version.Major -ge 13) { $packages += @("crt", "nvvm", "nvptxcompiler") }
# The CCCL subpackage is named "thrust" through 13.4 and "cccl" from 13.5 on.
$packages += if ($version -ge [version]"13.5") { "cccl" } else { "thrust" }
$packages += $ExtraSubPackages
$subPackages = $packages | ForEach-Object { "${_}_$cudaShortVersion" }

$cudaPath = "$env:ProgramFiles\NVIDIA GPU Computing Toolkit\CUDA\v$cudaShortVersion"
$installer = Join-Path $env:RUNNER_TEMP $installerName

Write-Host "Downloading $url"
curl.exe --fail --location --retry 3 --output $installer $url
if ($LASTEXITCODE -ne 0) { throw "Failed to download $url" }

Write-Host "Installing subpackages: $($subPackages -join ' ')"
$proc = Start-Process -FilePath $installer -ArgumentList (@("-s", "-n") + $subPackages) -Wait -PassThru
if ($proc.ExitCode -ne 0) { throw "CUDA installer exited with code $($proc.ExitCode)" }
# The installer can keep a handle on its own executable after exiting, so cleanup is best-effort.
Remove-Item $installer -Force -ErrorAction SilentlyContinue

if (-not (Test-Path "$cudaPath\bin\nvcc.exe")) { throw "nvcc not found under $cudaPath" }

Write-Host "Installed CUDA to $cudaPath"
& "$cudaPath\bin\nvcc.exe" --version

"CUDA_PATH=$cudaPath" | Out-File $env:GITHUB_ENV -Append
"$cudaPath\bin" | Out-File $env:GITHUB_PATH -Append
