from typing import Dict

from bitsandbytes.backends.base import Backend

backends: Dict[str, Backend] = {}

def register_backend(backend_name: str, backend_instance: Backend):
    backends[backend_name.lower()] = backend_instance
