"""Debug CUDA graph capture for NVFP4 GEMM."""
import ctypes as ct
import torch

def get_ptr(t):
    return ct.c_void_p(t.data_ptr())

def main():
    from bitsandbytes.cextension import lib

    device = torch.device("cuda")
    gpu = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"GPU: {gpu} (SM {cap[0]}.{cap[1]})")

    num_experts, max_M, N, K = 8, 128, 13696, 4096
    half_K = K // 2

    A_bat = torch.randint(0, 255, (num_experts * max_M * half_K,),
                          dtype=torch.uint8, device=device)
    B_all = torch.randint(0, 255, (num_experts * N * half_K,),
                          dtype=torch.uint8, device=device)

    lib.cgemm_nvfp4_moe_sm100_sfa_size.restype = ct.c_size_t
    lib.cgemm_nvfp4_moe_sm100_sfb_size.restype = ct.c_size_t
    sfa_bytes = lib.cgemm_nvfp4_moe_sm100_sfa_size(
        ct.c_int(N), ct.c_int(max_M), ct.c_int(K), ct.c_int(num_experts))
    sfb_bytes = lib.cgemm_nvfp4_moe_sm100_sfb_size(
        ct.c_int(N), ct.c_int(max_M), ct.c_int(K), ct.c_int(num_experts))
    SFA = torch.randint(0, 255, (max(sfa_bytes, 1),), dtype=torch.uint8, device=device)
    SFB = torch.randint(0, 255, (max(sfb_bytes, 1),), dtype=torch.uint8, device=device)

    D_out = torch.empty(num_experts * max_M, N, dtype=torch.bfloat16, device=device)
    alpha = torch.tensor([1.0], dtype=torch.float32, device=device)

    lib.cgemm_nvfp4_moe_sm100_workspace_size.restype = ct.c_size_t
    ws_size = lib.cgemm_nvfp4_moe_sm100_workspace_size(
        ct.c_int(N), ct.c_int(max_M), ct.c_int(K), ct.c_int(num_experts))
    workspace = torch.empty(max(ws_size, 1), dtype=torch.uint8, device=device)

    stream = torch.cuda.current_stream()
    stream_ptr = ct.c_void_p(stream.cuda_stream)

    print(f"Stream ptr: {stream.cuda_stream}")
    print(f"Workspace size: {ws_size}")

    # Init
    lib.cgemm_nvfp4_moe_sm100_init.restype = ct.c_int
    ret = lib.cgemm_nvfp4_moe_sm100_init(
        ct.c_int(N), ct.c_int(max_M), ct.c_int(K), ct.c_int(num_experts),
        get_ptr(A_bat), get_ptr(B_all),
        get_ptr(SFA), get_ptr(SFB),
        get_ptr(D_out), get_ptr(alpha),
        get_ptr(workspace), ct.c_size_t(ws_size), stream_ptr,
    )
    print(f"Init returned: {ret}")
    if ret != 0:
        print("Init failed!")
        return

    # Warmup eager run
    lib.cgemm_nvfp4_moe_sm100_run.restype = ct.c_int
    for i in range(3):
        ret = lib.cgemm_nvfp4_moe_sm100_run(stream_ptr)
        print(f"Eager run {i}: {ret}")
    torch.cuda.synchronize()
    print("Eager runs OK")

    # Test 1: BF16 bmm graph (sanity check graph capture works)
    A_bf = torch.randn(8, 128, 4096, dtype=torch.bfloat16, device=device)
    B_bf = torch.randn(8, 4096, 13696, dtype=torch.bfloat16, device=device)
    C_bf = torch.empty(8, 128, 13696, dtype=torch.bfloat16, device=device)
    torch.bmm(A_bf, B_bf, out=C_bf)
    torch.cuda.synchronize()

    g1 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g1):
        torch.bmm(A_bf, B_bf, out=C_bf)
    g1.replay()
    torch.cuda.synchronize()
    print("BF16 graph capture: OK")

    # Test 2: NVFP4 GEMM graph capture
    print("Attempting NVFP4 graph capture...")

    # The graph capture stream — check what stream it uses
    g2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g2):
        cap_stream = torch.cuda.current_stream()
        cap_stream_ptr = ct.c_void_p(cap_stream.cuda_stream)
        print(f"  Capture stream ptr: {cap_stream.cuda_stream}")
        ret = lib.cgemm_nvfp4_moe_sm100_run(cap_stream_ptr)
        # Note: ret might not be meaningful during capture
    print(f"  Graph capture complete, run returned: {ret}")

    # Replay
    print("Replaying NVFP4 graph...")
    g2.replay()
    torch.cuda.synchronize()
    print("NVFP4 graph replay: OK")

    # Timing
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):
        g2.replay()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / 100
    flops = 2 * num_experts * max_M * N * K
    tflops = flops / (ms * 1e-3) / 1e12
    print(f"NVFP4 GEMM graph: {ms:.3f} ms, {tflops:.1f} TFLOPS")


if __name__ == "__main__":
    main()
