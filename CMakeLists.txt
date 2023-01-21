# Set cmake version, name the whole project, set it's languages, set cpp language version to 20 (23 is "latest", 20 1 step before..)
cmake_minimum_required (VERSION 3.25.0)
project(bitsandbytes LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
if(WIN32)
# The trick is to set this here instead of in the CMakeLists.txt inside csrc/ .. For whatever reason just doesn't work there.
	set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS 1)
ENDIF()

# Recurse into the csrc dir
add_subdirectory (csrc)
