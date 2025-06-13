#include <iostream>
#include "simple_kernel.hpp"

void test_simple_kernel() {
  int numel = 1024;
  float a[1024];

  // a simple sycl kernel
  itoa(a, numel);

  bool success = true;
  for (int i = 0; i < numel; i++) {
    if (a[i] != i) {
      success = false;
      break;
    }
  }

  if (success) {
    std::cout << "Pass" << std::endl;
  } else {
    std::cout << "Fail" << std::endl;
  }
}

int main(int argc, char* argv[]) {
  test_simple_kernel();
  return 0;
}
