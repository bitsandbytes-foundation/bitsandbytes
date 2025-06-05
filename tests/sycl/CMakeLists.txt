set(TEST_SYCL_ROOT ${PROJECT_SOURCE_DIR}/tests/sycl)
set(TEST_SYCL_SYCL_SRCS ${TEST_SYCL_ROOT}/simple_kernel.cpp)
set(TEST_SYCL_CXX_SRCS ${TEST_SYCL_ROOT}/main.cpp)

# test building executable
sycl_add_executable(
  test_sycl_build_standalone
  SYCL_SOURCES ${TEST_SYCL_SYCL_SRCS}
  CXX_SOURCES ${TEST_SYCL_CXX_SRCS} ${TEST_SYCL_CXX_SRCS})

install(TARGETS test_sycl_build_standalone DESTINATION bin)

# test building archive static library
sycl_add_library(
  sycl_simple_kernel_test
  STATIC
  SYCL_SOURCES ${TEST_SYCL_SYCL_SRCS})

add_executable(
  test_sycl_build_archive
  ${TEST_SYCL_CXX_SRCS})
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  # Windows
  target_link_libraries(test_sycl_build_archive
  "-WHOLEARCHIVE:\"$<TARGET_FILE:sycl_simple_kernel_test>\"")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # Linux
  target_link_libraries(test_sycl_build_archive
    "-Wl,--whole-archive,\"$<TARGET_FILE:sycl_simple_kernel_test>\" -Wl,--no-whole-archive")
endif()
add_dependencies(test_sycl_build_archive sycl_simple_kernel_test)

# SYCL runtime library could be a transitive link library of
# ``test_sycl_build_archive``, if using,
# ``
# target_link_libraries(
#   test_sycl_build_archive
#   sycl_simple_kernel_test)
# ``
# Instead, we use explicit linkage option '--whole-archive', which is required
# by linkage of device object modules archived in the static library. Then
# explicit linkage configuration of SYCL runtime library is required.
target_link_libraries(test_sycl_build_archive ${SYCL_LIBRARY})

install(TARGETS test_sycl_build_archive DESTINATION bin)
