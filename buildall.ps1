# 11.6
#Remove-Item -Recurse -Force ./build
Remove-Item -Recurse -Force ./build_11.6_0
Remove-Item -Recurse -Force ./build_11.6_1
Remove-Item -Recurse -Force ./build_11.7_0
Remove-Item -Recurse -Force ./build_11.7_1

$Funcs = {
    function configure([string]$cuda_version, [string]$no_cuda_blast)
    {
        cmake -S . -B "./build_${cuda_version}_${no_cuda_blast}" -G "Visual Studio 17 2022" -T "cuda=${cuda_version}" -D "NO_CUBLASLT=${no_cuda_blast}"
    }
    function build([string]$cuda_version, [string]$no_cuda_blast)
    {     
        cmake --build "./build_${cuda_version}_${no_cuda_blast}" --target libbitsandbytes_cuda --config Release
    }
    function build_with_cpu([string]$cuda_version, [string]$no_cuda_blast)
    {
        cmake --build "./build_${cuda_version}_${no_cuda_blast}" --target libbitsandbytes_cuda --config Release
        cmake --build "./build_${cuda_version}_${no_cuda_blast}" --target libbitsandbytes_cpu --config Release
    }
}


$j1 = Start-Job -ScriptBlock { configure "11.6" "0"; build_with_cpu "11.6" "0"} -InitializationScript $Funcs
$j2 = Start-Job -ScriptBlock { configure "11.6" "1"; build "11.6" "1" } -InitializationScript $Funcs
$j3 = Start-Job -ScriptBlock { configure "11.7" "0"; build "11.7" "0" } -InitializationScript $Funcs
$j4 = Start-Job -ScriptBlock { configure "11.7" "1"; build "11.7" "1" } -InitializationScript $Funcs

Get-Job | Wait-Job

Receive-Job $j1
Receive-Job $j2
Receive-Job $j3
Receive-Job $j4


