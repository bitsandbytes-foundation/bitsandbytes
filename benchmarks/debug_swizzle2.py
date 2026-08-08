"""Debug: test if kernel works with FLAT scales (it should NOT if swizzle code is active)."""
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


def run_gemm(lib, A_packed, B_packed, A_scales, B_scales, M, N, K):
    D_out = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    workspace = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    stream = torch.cuda.current_stream()
    lib.cgemm_nvfp4_bf16(
        ctypes.c_void_p(A_packed.data_ptr()),
        ctypes.c_void_p(B_packed.data_ptr()),
        ctypes.c_void_p(A_scales.data_ptr()),
        ctypes.c_void_p(B_scales.data_ptr()),
        ctypes.c_void_p(D_out.data_ptr()),
        ctypes.c_void_p(workspace.data_ptr()),
        ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K),
        ctypes.c_void_p(stream.cuda_stream),
    )
    torch.cuda.synchronize()
    return D_out.float()


def main():
    lib = get_lib()
    M, N, K = 16, 8, 64
    scale_K = K // 16

    A_packed = torch.full((M * K // 2,), 0x22, dtype=torch.uint8, device="cuda")
    B_packed = torch.full((N * K // 2,), 0x22, dtype=torch.uint8, device="cuda")
    A_scales_flat = torch.full((M * scale_K,), 0x38, dtype=torch.uint8, device="cuda")
    B_scales_flat = torch.full((N * scale_K,), 0x38, dtype=torch.uint8, device="cuda")

    # Test 1: flat scales (OLD format) — should FAIL if swizzle code is compiled in
    print("=== Test 1: FLAT scales (old format) ===")
    D_flat = run_gemm(lib, A_packed, B_packed, A_scales_flat, B_scales_flat, M, N, K)
    all_64_flat = torch.allclose(D_flat, torch.full((M, N), 64.0, device="cuda"))
    print(f"All 64.0: {all_64_flat}")
    print(f"First row: {D_flat[0].tolist()}")

    # Test 2: swizzled scales (NEW format) — should PASS if swizzle code is compiled in
    print("\n=== Test 2: SWIZZLED scales (new format) ===")
    A_scales_sw = swizzle_scales_gpu(lib, A_scales_flat, M, scale_K)
    B_scales_sw = swizzle_scales_gpu(lib, B_scales_flat, N, scale_K)
    D_sw = run_gemm(lib, A_packed, B_packed, A_scales_sw, B_scales_sw, M, N, K)
    all_64_sw = torch.allclose(D_sw, torch.full((M, N), 64.0, device="cuda"))
    print(f"All 64.0: {all_64_sw}")
    print(f"First row: {D_sw[0].tolist()}")

    if all_64_flat and not all_64_sw:
        print("\n>>> KERNEL IS STILL USING FLAT INDEXING (old code)")
    elif not all_64_flat and all_64_sw:
        print("\n>>> KERNEL IS USING SWIZZLED INDEXING (new code)")
    elif all_64_flat and all_64_sw:
        print("\n>>> BOTH PASS — uniform scales, can't distinguish")
    else:
        print("\n>>> BOTH FAIL — something else is broken")


if __name__ == "__main__":
    main()
