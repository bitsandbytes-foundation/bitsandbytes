import os
import subprocess
import sys
from urllib.request import urlretrieve

cuda_versions = {
    "110": "https://developer.download.nvidia.com/compute/cuda/11.0.3/local_installers/cuda_11.0.3_450.51.06_linux.run",
    "111": "https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda_11.1.1_455.32.00_linux.run",
    "112": "https://developer.download.nvidia.com/compute/cuda/11.2.2/local_installers/cuda_11.2.2_460.32.03_linux.run",
    "113": "https://developer.download.nvidia.com/compute/cuda/11.3.1/local_installers/cuda_11.3.1_465.19.01_linux.run",
    "114": "https://developer.download.nvidia.com/compute/cuda/11.4.4/local_installers/cuda_11.4.4_470.82.01_linux.run",
    "115": "https://developer.download.nvidia.com/compute/cuda/11.5.2/local_installers/cuda_11.5.2_495.29.05_linux.run",
    "116": "https://developer.download.nvidia.com/compute/cuda/11.6.2/local_installers/cuda_11.6.2_510.47.03_linux.run",
    "117": "https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/cuda_11.7.1_515.65.01_linux.run",
    "118": "https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run",
    "120": "https://developer.download.nvidia.com/compute/cuda/12.0.1/local_installers/cuda_12.0.1_525.85.12_linux.run",
    "121": "https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_530.30.02_linux.run",
    "122": "https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda_12.2.2_535.104.05_linux.run",
    "123": "https://developer.download.nvidia.com/compute/cuda/12.3.2/local_installers/cuda_12.3.2_545.23.08_linux.run",
    "124": "https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_550.54.15_linux.run",
    "125": "https://developer.download.nvidia.com/compute/cuda/12.5.0/local_installers/cuda_12.5.0_555.42.02_linux.run",
}


def install_cuda(version, base_path, download_path):
    formatted_version = f"{version[:-1]}.{version[-1]}"
    folder = f"cuda-{formatted_version}"
    install_path = os.path.join(base_path, folder)

    if os.path.exists(install_path):
        print(f"Removing existing CUDA version {version} at {install_path}...")
        subprocess.run(["rm", "-rf", install_path], check=True)

    url = cuda_versions[version]
    filename = url.split("/")[-1]
    filepath = os.path.join(download_path, filename)

    if not os.path.exists(filepath):
        print(f"Downloading CUDA version {version} from {url}...")
        urlretrieve(url, filepath)
    else:
        print(f"Installer for CUDA version {version} already downloaded.")

    # Make the installer executable
    subprocess.run(["chmod", "+x", filepath], check=True)

    # Install CUDA
    print(f"Installing CUDA version {version}...")
    install_command = [
        "bash",
        filepath,
        "--no-drm",
        "--no-man-page",
        "--override",
        "--toolkitpath=" + install_path,
        "--toolkit",
        "--silent",
    ]

    print(f"Running command: {' '.join(install_command)}")

    try:
        subprocess.run(install_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Installation failed for CUDA version {version}: {e}")
        return
    finally:
        # Delete the installer file
        os.remove(filepath)

    print(f"CUDA version {version} installed at {install_path}")


def main():
    user_base_path = os.path.expanduser("~/cuda")
    system_base_path = "/usr/local/cuda"
    base_path = user_base_path  # default to user-specific installation
    download_path = "/tmp"  # default download path

    if len(sys.argv) < 2:
        print("Usage: python install_cuda.py <version/all> [user/system] [download_path]")
        sys.exit(1)

    version = sys.argv[1]
    if len(sys.argv) > 2:
        base_path = system_base_path if sys.argv[2] == "system" else user_base_path
    if len(sys.argv) > 3:
        download_path = sys.argv[3]

    if not os.path.exists(base_path):
        os.makedirs(base_path)
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    # Install CUDA version(s)
    if version == "all":
        for ver in cuda_versions.keys():
            install_cuda(ver, base_path, download_path)
    elif version in cuda_versions:
        install_cuda(version, base_path, download_path)
    else:
        print(f"Invalid CUDA version: {version}. Available versions are: {', '.join(cuda_versions.keys())}")
        sys.exit(1)


if __name__ == "__main__":
    main()
