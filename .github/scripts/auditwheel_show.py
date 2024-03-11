import argparse
import subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wheels", nargs="*")
    args = ap.parse_args()
    if not args.wheels:
        ap.error("At least one wheel must be provided.")
    for whl in args.wheels:
        print(f"### `{whl}`")

        audit_wheel_output = subprocess.run(
            ["auditwheel", "show", whl],
            capture_output=True,
            text=True,
            errors="backslashreplace",
        )

        if audit_wheel_output.stdout:
            print(audit_wheel_output.stdout)

        if audit_wheel_output.stderr:
            print(f"**Error:**\n```{audit_wheel_output.stderr}```")

        print("---")


if __name__ == "__main__":
    main()
