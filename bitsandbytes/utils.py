import shlex
import subprocess
import sys
from typing import Tuple


def execute_and_return(command_string: str) -> Tuple[str, str]:
    def _decode(subprocess_err_out_tuple):
        return tuple(
            to_decode.decode("UTF-8").strip()
            for to_decode in subprocess_err_out_tuple
        )

    def execute_and_return_decoded_std_streams(command_string):
        return _decode(
            subprocess.Popen(
                shlex.split(command_string),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ).communicate()
        )

    std_out, std_err = execute_and_return_decoded_std_streams(command_string)
    return std_out, std_err


def print_stderr(s: str) -> None:
    print(s, file=sys.stderr)


def warn_of_missing_prerequisite(s: str) -> None:
    print_stderr("WARNING, missing pre-requisite: " + s)
