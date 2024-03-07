import glob
import os
import subprocess


def append_to_summary(content):
    with open(os.getenv("GITHUB_STEP_SUMMARY"), "a") as summary_file:
        summary_file.write(content + "\n")


subprocess.run(["pip", "install", "-q", "auditwheel"])

wheel_files = glob.glob("dist/*.whl")

for whl in wheel_files:
    append_to_summary("---")
    append_to_summary("### 🎡 Auditing wheel: `" + whl + "`\n")

    audit_wheel_output = subprocess.run(
        ["auditwheel", "show", whl], capture_output=True, text=True
    )

    if audit_wheel_output.stdout:
        append_to_summary(audit_wheel_output.stdout + "\n")

    if audit_wheel_output.stderr:
        append_to_summary("**Error:**\n```\n" + audit_wheel_output.stderr + "```\n")

    append_to_summary("\n🐍 **Slithering on to the next one...** 🐍\n")
