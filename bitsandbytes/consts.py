from pathlib import Path
import platform

DYNAMIC_LIBRARY_SUFFIX = {
    "Darwin": ".dylib",
    "Linux": ".so",
    "Windows": ".dll",
}.get(platform.system(), ".so")

PACKAGE_DIR = Path(__file__).parent
PACKAGE_GITHUB_URL = "https://github.com/TimDettmers/bitsandbytes"
NONPYTORCH_DOC_URL = "https://github.com/TimDettmers/bitsandbytes/blob/main/docs/source/nonpytorchcuda.mdx"
