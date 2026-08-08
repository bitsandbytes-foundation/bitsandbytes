"""Debug script: verify swizzle scale loading produces correct GEMM output."""
import ctypes
import os
import torch


def get_lib():
    lib_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bitsandbytes")
    for suffix in ["cuda131", "cuda130"]:
        lib_path = os.path.join(lib_dir, f"libbitsandbytes_{suffix}.so")
        if os.path.exists(lib_path):
            return ctypes.cdll.LoadLibrary(lib_path)
    raise RuntimeError("Could not find lib")


def swizzle_scales_gpu(lib, flat_scales, rows, scale_K):
    """Swizzle flat scales using cscale_to_blocked."""
    n_row_blocks = (rows + 127) // 128
    n_col_blocks = (scale_K + 3) // 4
    out_size = n_row_blocks * n_col_blocks * 128 * 4
    swizzled = torch.zeros(out_size, dtype=torch.uint8, device=flat_scales.device)
    stream = torch.cuda.current_stream()
    lib.cscale_to_blocked(
        ctypes.c_void_p(flat_scales.data_ptr()),
        ctypes.c_void_p(swizzled.data_ptr()),
        ctypes.c_int(rows),
        ctypes.c_int(scale_K),
        ctypes.c_void_p(stream.cuda_stream),
    )
    torch.cuda.synchronize()
    return swizzled


def swizzle_offset_cpu(row, col, n_col_blocks):
    """Python version of swizzled_scale_offset."""
    block_row = row >> 7
    block_col = col >> 2
    r = row & 127
    c = col & 3
    block_idx = block_row * n_col_blocks + block_col
    return block_idx * 512 + (r & 31) * 16 + (r >> 5) * 4 + c


def main():
    lib = get_lib()

    M, N, K = 16, 8, 64
    scale_K = K // 16  # = 4
    n_col_blocks = (scale_K + 3) // 4  # = 1

    # Flat scales: all 0x38 (UE4M3 for 1.0)
    A_flat = torch.full((M * scale_K,), 0x38, dtype=torch.uint8, device="cuda")
    B_flat = torch.full((N * scale_K,), 0x38, dtype=torch.uint8, device="cuda")

    # Swizzle
    A_sw = swizzle_scales_gpu(lib, A_flat, M, scale_K)
    B_sw = swizzle_scales_gpu(lib, B_flat, N, scale_K)

    # Dump first 64 bytes of swizzled A scales
    A_sw_cpu = A_sw.cpu().numpy()
    print(f"Swizzled A scales (first 64 bytes, M={M}, scale_K={scale_K}):")
    for i in range(min(64, len(A_sw_cpu))):
        val = A_sw_cpu[i]
        marker = " <-- expected 0x38" if val == 0x38 else ""
        print(f"  [{i:3d}] = 0x{val:02x}{marker}")

    print()

    # Verify: for each row, the 4 bytes at swizzled offset should be 0x38
    print("Checking kernel read offsets:")
    for row in range(M):
        offset = swizzle_offset_cpu(row, 0, n_col_blocks)
        vals = [A_sw_cpu[offset + i] for i in range(4)]
        ok = all(v == 0x38 for v in vals)
        status = "OK" if ok else "WRONG"
        print(f"  row {row:2d}: offset={offset:3d}, bytes={[f'0x{v:02x}' for v in vals]} {status}")

    print()

    # Run GEMM
    A_packed = torch.full((M * K // 2,), 0x22, dtype=torch.uint8, device="cuda")
    B_packed = torch.full((N * K // 2,), 0x22, dtype=torch.uint8, device="cuda")
    D_out = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    workspace = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    stream = torch.cuda.current_stream()

    lib.cgemm_nvfp4_bf16(
        ctypes.c_void_p(A_packed.data_ptr()),
        ctypes.c_void_p(B_packed.data_ptr()),
        ctypes.c_void_p(A_sw.data_ptr()),
        ctypes.c_void_p(B_sw.data_ptr()),
        ctypes.c_void_p(D_out.data_ptr()),
        ctypes.c_void_p(workspace.data_ptr()),
        ctypes.c_int(M),
        ctypes.c_int(N),
        ctypes.c_int(K),
        ctypes.c_void_p(stream.cuda_stream),
    )
    torch.cuda.synchronize()

    D = D_out.float()
    print(f"GEMM output (M={M}, N={N}, K={K}):")
    for row in range(M):
        vals = [f"{D[row, col].item():5.1f}" for col in range(N)]
        print(f"  row {row:2d}: {' '.join(vals)}")


if __name__ == "__main__":
    main()
