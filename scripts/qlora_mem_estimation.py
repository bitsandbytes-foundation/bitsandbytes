import argparse


def estimate_saved_mem_due_to_qlora_activation_bug(hidden_size: int, seqlen: int, num_layers: int) -> float:
    return 7.5 * hidden_size * 2 * num_layers * seqlen / 1e9


def estimate_activations(hidden_size: int, seqlen: int, batch_size: int, multiplicator: int, num_layers: int) -> float:
    return seqlen * hidden_size * 2 * batch_size * multiplicator * num_layers / 1e9


def estimate_activations_good_implementation(hidden_size: int, seqlen: int, batch_size: int, num_layers: int) -> float:
    """Estimate for a good implementation"""
    return estimate_activations(hidden_size, seqlen, batch_size, 8, num_layers)


def estimate_activations_bad_implementation(hidden_size: int, seqlen: int, batch_size: int, num_layers: int) -> float:
    """Estimate for a bad implementation"""
    return estimate_activations(hidden_size, seqlen, batch_size, 16, num_layers)


def calculate_parameter_related_memory(param_count: int) -> float:
    """Calculate parameter-related memory for a given number of parameters"""
    return (param_count * 4 + param_count * 8) * 0.01 + param_count * 0.5


def main():
    parser = argparse.ArgumentParser(description="Estimate memory usage and savings due to QLoRA activation bug fix.")
    parser.add_argument("--seqlen", type=int, required=True, help="Sequence length")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size")
    parser.add_argument(
        "--model_size", type=str, required=True, choices=["70B", "405B"], help="Model size (70B or 405B)"
    )
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs (default is 1)")
    parser.add_argument("--vram_per_gpu", type=float, default=80.0, help="VRAM per GPU in GB (default is 80GB)")
    parser.add_argument(
        "--include_bug_estimate", action="store_true", help="Include memory estimates if the bug were present"
    )

    args = parser.parse_args()
    seqlen = args.seqlen
    batch_size = args.batch_size
    model_size = args.model_size
    num_gpus = args.num_gpus
    vram_per_gpu = args.vram_per_gpu
    include_bug_estimate = args.include_bug_estimate

    llama_config = {
        "70B": {"hidden_size": 8192, "num_layers": 80},
        "405B": {"hidden_size": 16384, "num_layers": 126},
    }
    hidden_size = llama_config[model_size]["hidden_size"]
    num_layers = llama_config[model_size]["num_layers"]

    param_count = 405e9 if model_size == "405B" else 70e9
    param_memory_gb = calculate_parameter_related_memory(param_count) / 1e9  # Convert to GB

    saved_mem = estimate_saved_mem_due_to_qlora_activation_bug(hidden_size, seqlen, num_layers)
    activations_good = estimate_activations_good_implementation(hidden_size, seqlen, batch_size, num_layers)
    activations_bad = estimate_activations_bad_implementation(hidden_size, seqlen, batch_size, num_layers)

    total_mem_good = param_memory_gb + activations_good
    total_mem_bad = param_memory_gb + activations_bad

    max_memory_per_gpu = vram_per_gpu * num_gpus

    print(f"\nModel Size: {model_size}")
    print(f"Sequence Length: {seqlen}")
    print(f"Batch Size: {batch_size}")
    print(f"Number of GPUs: {num_gpus}")
    print(f"VRAM per GPU: {vram_per_gpu} GB")

    print("\n--- Memory Estimates ---")
    print(f"Parameter-related Memory: {param_memory_gb:.2f} GB")
    print(f"Saved Memory Due to QLoRA Activation Bug Fix: {saved_mem:.2f} GB")
    print(f"Activation Memory (Good Implementation): {activations_good:.2f} GB")
    print(f"Activation Memory (Bad Implementation): {activations_bad:.2f} GB")
    print(f"Total Memory (Good Implementation): {total_mem_good:.2f} GB")
    print(f"Total Memory (Bad Implementation): {total_mem_bad:.2f} GB")
    print(f"Total Memory per GPU (Good Implementation): {total_mem_good / num_gpus:.2f} GB")
    print(f"Total Memory per GPU (Bad Implementation): {total_mem_bad / num_gpus:.2f} GB")

    if include_bug_estimate:
        total_mem_good_with_bug = total_mem_good + saved_mem
        total_mem_bad_with_bug = total_mem_bad + saved_mem

        print("\n--- Memory Estimates if Bug Were Present ---")
        print(f"Total Memory (Good Implementation with Bug): {total_mem_good_with_bug:.2f} GB")
        print(f"Total Memory (Bad Implementation with Bug): {total_mem_bad_with_bug:.2f} GB")
        print(f"Total Memory per GPU (Good Implementation with Bug): {total_mem_good_with_bug / num_gpus:.2f} GB")
        print(f"Total Memory per GPU (Bad Implementation with Bug): {total_mem_bad_with_bug / num_gpus:.2f} GB")

        if total_mem_good_with_bug > max_memory_per_gpu:
            print(
                f"Warning: Total memory usage (good implementation with bug) exceeds available VRAM across GPUs ({max_memory_per_gpu:.2f} GB)"
            )

        if total_mem_bad_with_bug > max_memory_per_gpu:
            print(
                f"Warning: Total memory usage (bad implementation with bug) exceeds available VRAM across GPUs ({max_memory_per_gpu:.2f} GB)"
            )

    if total_mem_good > max_memory_per_gpu:
        print(
            f"Warning: Total memory usage (good implementation) exceeds available VRAM across GPUs ({max_memory_per_gpu:.2f} GB)"
        )

    if total_mem_bad > max_memory_per_gpu:
        print(
            f"Warning: Total memory usage (bad implementation) exceeds available VRAM across GPUs ({max_memory_per_gpu:.2f} GB)"
        )


if __name__ == "__main__":
    main()
