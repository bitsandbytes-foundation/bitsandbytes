#[=======================================================================[.rst:
SYCLConfig
-------

Library to verify SYCL compatability of CMAKE_CXX_COMPILER
and passes relevant compiler flags.

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``SYCLTOOLKIT_FOUND``
  True if the system has the SYCL library.
``SYCL_COMPILER``
  SYCL compiler executable.
``SYCL_INCLUDE_DIR``
  Include directories needed to use SYCL.
``SYCL_LIBRARY_DIR``
  Libaray directories needed to use SYCL.
``SYCL_FLAGS``
  SYCL specific flags for the compiler.
``SYCL_LANGUAGE_VERSION``
  The SYCL language spec version by Compiler.

#]=======================================================================]

include(FindPackageHandleStandardArgs)

set(SYCL_ROOT "")
if(DEFINED ENV{SYCL_ROOT})
  set(SYCL_ROOT $ENV{SYCL_ROOT})
elseif(DEFINED ENV{CMPLR_ROOT})
  set(SYCL_ROOT $ENV{CMPLR_ROOT})
else()
  if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    set(SYCL_ROOT "/opt/intel/oneapi/compiler/latest")
  elseif(CMAKE_SYSTEM_NAME MATCHES "Windows")
    set(SYCL_ROOT "C:/Program Files (x86)/Intel/oneAPI/compiler/latest")
  endif()
  if(NOT EXISTS ${SYCL_ROOT})
    set(SYCL_ROOT "")
  endif()
endif()

string(COMPARE EQUAL "${SYCL_ROOT}" "" nosyclfound)
if(nosyclfound)
  set(SYCL_FOUND False)
  set(SYCL_REASON_FAILURE "SYCL library not set!!")
  set(SYCL_NOT_FOUND_MESSAGE "${SYCL_REASON_FAILURE}")
  return()
endif()

# Find SYCL compiler executable.
find_program(
  SYCL_COMPILER
  NAMES icx
  PATHS "${SYCL_ROOT}"
  PATH_SUFFIXES bin bin64
  NO_DEFAULT_PATH
  )

function(parse_sycl_compiler_version version_number)
  # Execute the SYCL compiler with the --version flag to match the version string.
  execute_process(COMMAND ${SYCL_COMPILER} --version OUTPUT_VARIABLE SYCL_VERSION_STRING)
  string(REGEX REPLACE "Intel\\(R\\) (.*) Compiler ([0-9]+\\.[0-9]+\\.[0-9]+) (.*)" "\\2"
               SYCL_VERSION_STRING_MATCH ${SYCL_VERSION_STRING})
  string(REPLACE "." ";" SYCL_VERSION_LIST ${SYCL_VERSION_STRING_MATCH})
  # Split the version number list into major, minor, and patch components.
  list(GET SYCL_VERSION_LIST 0 VERSION_MAJOR)
  list(GET SYCL_VERSION_LIST 1 VERSION_MINOR)
  list(GET SYCL_VERSION_LIST 2 VERSION_PATCH)
  # Calculate the version number in the format XXXXYYZZ, using the formula (major * 10000 + minor * 100 + patch).
  math(EXPR VERSION_NUMBER_MATCH "${VERSION_MAJOR} * 10000 + ${VERSION_MINOR} * 100 + ${VERSION_PATCH}")
  set(${version_number} "${VERSION_NUMBER_MATCH}" PARENT_SCOPE)
endfunction()

if(SYCL_COMPILER)
  parse_sycl_compiler_version(SYCL_COMPILER_VERSION)
endif()

if(NOT SYCL_COMPILER_VERSION)
  set(SYCL_FOUND False)
  set(SYCL_REASON_FAILURE "Cannot parse sycl compiler version to get SYCL_COMPILER_VERSION!")
  set(SYCL_NOT_FOUND_MESSAGE "${SYCL_REASON_FAILURE}")
  return()
endif()

# Find include path from binary.
find_file(
  SYCL_INCLUDE_DIR
  NAMES include
  HINTS ${SYCL_ROOT}
  NO_DEFAULT_PATH
  )

# Find include/sycl path from include path.
find_file(
  SYCL_INCLUDE_SYCL_DIR
  NAMES sycl
  HINTS ${SYCL_ROOT}/include/
  NO_DEFAULT_PATH
  )

# Due to the unrecognized compilation option `-fsycl` in other compiler.
list(APPEND SYCL_INCLUDE_DIR ${SYCL_INCLUDE_SYCL_DIR})

# Find library directory from binary.
find_file(
  SYCL_LIBRARY_DIR
  NAMES lib lib64
  HINTS ${SYCL_ROOT}
  NO_DEFAULT_PATH
  )

set(COMPATIBLE_SYCL_TOOLKIT_VERSION 20249999)
# By default, we use libsycl.so on Linux and sycl.lib on Windows as the SYCL library name.
if (SYCL_COMPILER_VERSION VERSION_LESS_EQUAL COMPATIBLE_SYCL_TOOLKIT_VERSION)
  # Don't use if(LINUX) here since this requires cmake>=3.25 and file is installed
  # and used by other projects.
  # See: https://cmake.org/cmake/help/v3.25/variable/LINUX.html
  if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    set(sycl_lib_suffix "-preview")
  elseif(CMAKE_SYSTEM_NAME MATCHES "Windows")
    # On Windows, the SYCL library is named sycl7.lib until COMPATIBLE_SYCL_TOOLKIT_VERSION.
    # sycl.lib is supported in the later version.
    set(sycl_lib_suffix "7")
  endif()
endif()

# Find SYCL library fullname.
find_library(
  SYCL_LIBRARY
  NAMES "sycl${sycl_lib_suffix}"
  HINTS ${SYCL_LIBRARY_DIR}
  NO_DEFAULT_PATH
)

# Find OpenCL library fullname, which is a dependency of oneDNN.
find_library(
  OCL_LIBRARY
  NAMES OpenCL
  HINTS ${SYCL_LIBRARY_DIR}
  NO_DEFAULT_PATH
)

if((NOT SYCL_LIBRARY) OR (NOT OCL_LIBRARY))
  set(SYCL_FOUND False)
  set(SYCL_REASON_FAILURE "SYCL library is incomplete!!")
  set(SYCL_NOT_FOUND_MESSAGE "${SYCL_REASON_FAILURE}")
  return()
endif()

find_package_handle_standard_args(
  SYCL
  FOUND_VAR SYCL_FOUND
  REQUIRED_VARS SYCL_INCLUDE_DIR SYCL_LIBRARY_DIR SYCL_LIBRARY
  REASON_FAILURE_MESSAGE "${SYCL_REASON_FAILURE}"
  VERSION_VAR SYCL_COMPILER_VERSION
  )

if(NOT SYCL_FOUND)
  set(SYCLTOOLKIT_FOUND FALSE)
  return()
endif()

if(SYCLTOOLKIT_FOUND)
  return()
endif()

set(SYCLTOOLKIT_FOUND TRUE)

include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)

if(WIN32)
  set(SYCL_EXECUTABLE_NAME icx)
else()
  set(SYCL_EXECUTABLE_NAME icpx)
endif()

if(NOT SYCL_ROOT)
  execute_process(
    COMMAND which ${SYCL_EXECUTABLE_NAME}
    OUTPUT_VARIABLE SYCL_CMPLR_FULL_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(NOT EXISTS "${SYCL_CMPLR_FULL_PATH}")
    message(WARNING "Cannot find ENV{CMPLR_ROOT} or icpx, please setup SYCL compiler Tool kit enviroment before building!!")
    return()
  endif()

  get_filename_component(SYCL_BIN_DIR "${SYCL_CMPLR_FULL_PATH}" DIRECTORY)
  set(SYCL_ROOT ${SYCL_BIN_DIR}/..)
endif()

find_program(
  SYCL_COMPILER
  NAMES ${SYCL_EXECUTABLE_NAME}
  PATHS "${SYCL_ROOT}"
  PATH_SUFFIXES bin bin64
  NO_DEFAULT_PATH
  )

string(COMPARE EQUAL "${SYCL_COMPILER}" "" nocmplr)
if(nocmplr)
  set(SYCLTOOLKIT_FOUND False)
  set(SYCL_REASON_FAILURE "SYCL: CMAKE_CXX_COMPILER not set!!")
  set(SYCL_NOT_FOUND_MESSAGE "${SYCL_REASON_FAILURE}")
endif()

# Function to write a test case to verify SYCL features.

function(SYCL_CMPLR_TEST_WRITE src macro_name)

  set(cpp_macro_if "#if")
  set(cpp_macro_endif "#endif")

  set(SYCL_CMPLR_TEST_CONTENT "")
  string(APPEND SYCL_CMPLR_TEST_CONTENT "#include <iostream>\nusing namespace std;\n")
  string(APPEND SYCL_CMPLR_TEST_CONTENT "int main(){\n")

  # Feature tests goes here

  string(APPEND SYCL_CMPLR_TEST_CONTENT "${cpp_macro_if} defined(${macro_name})\n")
  string(APPEND SYCL_CMPLR_TEST_CONTENT "cout << \"${macro_name}=\"<<${macro_name}<<endl;\n")
  string(APPEND SYCL_CMPLR_TEST_CONTENT "${cpp_macro_endif}\n")

  string(APPEND SYCL_CMPLR_TEST_CONTENT "return 0;}\n")

  file(WRITE ${src} "${SYCL_CMPLR_TEST_CONTENT}")

endfunction()

# Function to Build the feature check test case.

function(SYCL_CMPLR_TEST_BUILD error TEST_SRC_FILE TEST_EXE)

  set(SYCL_CXX_FLAGS_LIST "${SYCL_CXX_FLAGS}")
  string(REPLACE "-Wno-stringop-overflow" "" SYCL_CXX_FLAGS_LIST "${SYCL_CXX_FLAGS_LIST}")
  separate_arguments(SYCL_CXX_FLAGS_LIST)

  execute_process(
    COMMAND "${SYCL_COMPILER}"
    ${SYCL_CXX_FLAGS_LIST}
    ${TEST_SRC_FILE}
    "-o"
    ${TEST_EXE}
    WORKING_DIRECTORY ${SYCL_CMPLR_TEST_DIR}
    OUTPUT_VARIABLE output ERROR_VARIABLE output
    OUTPUT_FILE ${SYCL_CMPLR_TEST_DIR}/Compile.log
    RESULT_VARIABLE result
    TIMEOUT 60
    )

  # Verify if test case build properly.
  if(result)
    message("SYCL: feature test compile failed!!")
    message("compile output is: ${output}")
  endif()

  set(${error} ${result} PARENT_SCOPE)

endfunction()

function(SYCL_CMPLR_TEST_RUN error TEST_EXE)

  execute_process(
    COMMAND ${TEST_EXE}
    WORKING_DIRECTORY ${SYCL_CMPLR_TEST_DIR}
    OUTPUT_VARIABLE output ERROR_VARIABLE output
    RESULT_VARIABLE result
    TIMEOUT 60
    )

  if(test_result)
    set(SYCLTOOLKIT_FOUND False)
    set(SYCL_REASON_FAILURE "SYCL: feature test execution failed!!")
  endif()

  set(test_result "${result}" PARENT_SCOPE)
  set(test_output "${output}" PARENT_SCOPE)

  set(${error} ${result} PARENT_SCOPE)

endfunction()

function(SYCL_CMPLR_TEST_EXTRACT test_output macro_name)

  string(REGEX REPLACE "\n" ";" test_output_list "${test_output}")

  set(${macro_name} "")
  foreach(strl ${test_output_list})
     if(${strl} MATCHES "^${macro_name}=([A-Za-z0-9_]+)$")
       string(REGEX REPLACE "^${macro_name}=" "" extracted_sycl_lang "${strl}")
       set(${macro_name} ${extracted_sycl_lang})
     endif()
  endforeach()

  set(${macro_name} "${extracted_sycl_lang}" PARENT_SCOPE)
endfunction()

set(SYCL_FLAGS "")
set(SYCL_LINK_FLAGS "")
list(APPEND SYCL_FLAGS "-fsycl")
list(APPEND SYCL_LINK_FLAGS "-fsycl")

set(SYCL_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SYCL_FLAGS}")

string(FIND "${CMAKE_CXX_FLAGS}" "-Werror" has_werror)
if(${has_werror} EQUAL -1)
  # Create a clean working directory.
  set(SYCL_CMPLR_TEST_DIR "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/TESTSYCLCMPLR")
  file(REMOVE_RECURSE ${SYCL_CMPLR_TEST_DIR})
  file(MAKE_DIRECTORY ${SYCL_CMPLR_TEST_DIR})

  # Create the test source file
  set(TEST_SRC_FILE "${SYCL_CMPLR_TEST_DIR}/sycl_features.cpp")
  set(TEST_EXE "${TEST_SRC_FILE}.exe")
  SYCL_CMPLR_TEST_WRITE(${TEST_SRC_FILE} "SYCL_LANGUAGE_VERSION")

  # Build the test and create test executable
  SYCL_CMPLR_TEST_BUILD(error ${TEST_SRC_FILE} ${TEST_EXE})
  if(error)
    return()
  endif()

  # Execute the test to extract information
  SYCL_CMPLR_TEST_RUN(error ${TEST_EXE})
  if(error)
    return()
  endif()

  # Extract test output for information
  SYCL_CMPLR_TEST_EXTRACT(${test_output} "SYCL_LANGUAGE_VERSION")

  # As per specification, all the SYCL compatible compilers should
  # define macro  SYCL_LANGUAGE_VERSION
  string(COMPARE EQUAL "${SYCL_LANGUAGE_VERSION}" "" nosycllang)
  if(nosycllang)
    set(SYCLTOOLKIT_FOUND False)
    set(SYCL_REASON_FAILURE "SYCL: It appears that the ${SYCL_COMPILER} does not support SYCL")
    set(SYCL_NOT_FOUND_MESSAGE "${SYCL_REASON_FAILURE}")
  endif()

  message(DEBUG "The SYCL Language Version is ${SYCL_LANGUAGE_VERSION}")

  # Include in Cache
  set(SYCL_LANGUAGE_VERSION "${SYCL_LANGUAGE_VERSION}" CACHE STRING "SYCL Language version")
endif()

# Create a clean working directory.
set(SYCL_CMPLR_TEST_DIR "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/TESTSYCLCMPLR")
file(REMOVE_RECURSE ${SYCL_CMPLR_TEST_DIR})
file(MAKE_DIRECTORY ${SYCL_CMPLR_TEST_DIR})
# Create the test source file
set(TEST_SRC_FILE "${SYCL_CMPLR_TEST_DIR}/llvm_features.cpp")
set(TEST_EXE "${TEST_SRC_FILE}.exe")
SYCL_CMPLR_TEST_WRITE(${TEST_SRC_FILE} "__INTEL_LLVM_COMPILER")
# Build the test and create test executable
SYCL_CMPLR_TEST_BUILD(error ${TEST_SRC_FILE} ${TEST_EXE})
if(error)
  message(FATAL_ERROR "Can not build SYCL_CMPLR_TEST")
endif()
# Execute the test to extract information
SYCL_CMPLR_TEST_RUN(error ${TEST_EXE})
if(error)
  message(FATAL_ERROR "Can not run SYCL_CMPLR_TEST")
endif()
# Extract test output for information
SYCL_CMPLR_TEST_EXTRACT(${test_output} "__INTEL_LLVM_COMPILER")

# Check whether the value of __INTEL_LLVM_COMPILER macro was successfully extracted
string(COMPARE EQUAL "${__INTEL_LLVM_COMPILER}" "" nosycllang)
if(nosycllang)
  set(SYCLTOOLKIT_FOUND False)
  set(SYCL_REASON_FAILURE "Can not find __INTEL_LLVM_COMPILER}")
  set(SYCL_NOT_FOUND_MESSAGE "${SYCL_REASON_FAILURE}")
endif()


# Include in Cache
set(__INTEL_LLVM_COMPILER "${__INTEL_LLVM_COMPILER}" CACHE STRING "Intel llvm compiler")

message(DEBUG "The SYCL compiler is ${SYCL_COMPILER}")
message(DEBUG "The SYCL Flags are ${SYCL_FLAGS}")
