import sys

def print_err(s: str) -> None:
    print(s, file=sys.stderr)

def warn_of_missing_prerequisite(s: str) -> None:
    print_err('WARNING, missing pre-requisite: ' + s)
