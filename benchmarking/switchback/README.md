Steps:

1. Run `python speed_benchmark/speed_benchmark.py` which times operations and writes their time to `speed_benchmark/info_a100_py2.jsonl` (change the name of the jsonl to a different name for your profiling).
2. Run `python speed_benchmark/make_plot_with_jsonl.py`, which produces the `speed_benchmark/plot_with_info.pdf`. Again make sure you change the jsonl which is being processed.
