import os
from typing import Dict


def to_be_ignored(env_var: str, value: str) -> bool:
    ignorable = {
        "PWD",  # PWD: this is how the shell keeps track of the current working dir
        "OLDPWD",
        "SSH_AUTH_SOCK",  # SSH stuff, therefore unrelated
        "SSH_TTY",
        "HOME",  # Linux shell default
        "TMUX",  # Terminal Multiplexer
        "XDG_DATA_DIRS",  # XDG: Desktop environment stuff
        "XDG_GREETER_DATA_DIR",  # XDG: Desktop environment stuff
        "XDG_RUNTIME_DIR",
        "MAIL",  # something related to emails
        "SHELL",  # binary for currently invoked shell
        "DBUS_SESSION_BUS_ADDRESS",  # hardware related
        "PATH",  # this is for finding binaries, not libraries
        "LESSOPEN",  # related to the `less` command
        "LESSCLOSE",
        "_",  # current Python interpreter
    }
    return env_var in ignorable


def might_contain_a_path(candidate: str) -> bool:
    return "/" in candidate


def is_active_conda_env(env_var: str) -> bool:
    return "CONDA_PREFIX" == env_var


def is_other_conda_env_var(env_var: str) -> bool:
    return "CONDA" in env_var


def is_relevant_candidate_env_var(env_var: str, value: str) -> bool:
    return is_active_conda_env(env_var) or (
        might_contain_a_path(value) and not
        is_other_conda_env_var(env_var) and not
        to_be_ignored(env_var, value)
    )


def get_potentially_lib_path_containing_env_vars() -> Dict[str, str]:
    return {
        env_var: value
        for env_var, value in os.environ.items()
        if is_relevant_candidate_env_var(env_var, value)
    }
