Steps:

1. Run `python speed_benchmark/speed_benchmark.py` which times operations and writes their time to `speed_benchmark/info_a100_py2.jsonl` (feel free to change the name for your profiling).
2. Run `python speed_benchmark/make_plot_with_jsonl.py`, which produces the `speed_benchmark/plot_with_info.pdf`.