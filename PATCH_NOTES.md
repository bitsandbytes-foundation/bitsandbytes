v0.0.21
- Ampere, RTX 30 series GPUs not compatible with the library.

v0.0.22:

- Fixed an error where a `reset_parameters()` call on the `StableEmbedding` would lead to an error in older PyTorch versions (from 1.7.0).
