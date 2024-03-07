import glob
import os
import subprocess


def append_to_summary(content):
    print(content + "\n")  # only for debugging now
    with open(os.getenv("GITHUB_STEP_SUMMARY"), "a") as summary_file:
        summary_file.write(content + "\n")


subprocess.run(["pip", "install", "-q", "auditwheel"])

wheel_files = glob.glob("**/*.whl", recursive=True)
print(wheel_files)  # only for debugging now

if not wheel_files:
    append_to_summary("No wheel files found in `dist/` directory.")
    exit(0)

for whl in wheel_files:
    append_to_summary("---")
    append_to_summary("### 🎡 Auditing wheel: `" + whl + "`")

    audit_wheel_output = subprocess.run(
        ["auditwheel", "show", whl], capture_output=True, text=True
    )

    if audit_wheel_output.stdout:
        append_to_summary(audit_wheel_output.stdout)

    if audit_wheel_output.stderr:
        append_to_summary("**Error:**\n```\n" + audit_wheel_output.stderr + "```")
