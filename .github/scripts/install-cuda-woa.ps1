# Installs CUDA Toolkit 13.4 or 13.5 for Windows ARM64.
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("13.4.0", "13.5.0")]
    [string]$CudaVersion
)

$ErrorActionPreference = "Stop"

$versionParts = $CudaVersion.Split(".")
$cudaShortVersion = "$($versionParts[0]).$($versionParts[1])"
if ($CudaVersion -eq "13.4.0") {
    $baseUrl = "https://packages.nvidia.com/prerelease/cuda"
    $ccclComponent = "thrust"
} else {
    $baseUrl = "https://developer.download.nvidia.com/compute/cuda"
    $ccclComponent = "cccl"
}
$url = "$baseUrl/$CudaVersion/local_installers/cuda_${CudaVersion}_windows_arm64.exe"
$subPackages = @(
    "nvcc_$cudaShortVersion", "crt_$cudaShortVersion", "nvvm_$cudaShortVersion", "nvptxcompiler_$cudaShortVersion",
    "cudart_$cudaShortVersion", "cublas_$cudaShortVersion", "cublas_dev_$cudaShortVersion",
    "${ccclComponent}_$cudaShortVersion"
)
$cudaPath = "$env:ProgramFiles\NVIDIA GPU Computing Toolkit\CUDA\v$cudaShortVersion"
$installer = Join-Path $env:RUNNER_TEMP "cuda_${CudaVersion}_windows_arm64.exe"

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
