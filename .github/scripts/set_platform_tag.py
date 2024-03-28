import argparse
import platform
import sys


def get_platform_tag(architecture):
    system = platform.system()

    if system == "Linux":
        tag = "manylinux_2_24_x86_64" if architecture == "x86_64" else "manylinux_2_24_aarch64"
    elif system == "Darwin":
        tag = "macosx_13_1_x86_64" if architecture == "x86_64" else "macosx_13_1_arm64"
    elif system == "Windows":
        tag = "win_amd64" if architecture == "x86_64" else "win_arm64"
    else:
        sys.exit(f"Unsupported system: {system}")

    return tag


def main():
    parser = argparse.ArgumentParser(description="Determine platform tag.")
    parser.add_argument("arch", type=str, help="Architecture (e.g., x86_64, aarch64)")
    args = parser.parse_args()

    tag = get_platform_tag(args.arch)

    print(tag)  # This will be captured by the GitHub Actions workflow


if __name__ == "__main__":
    main()
