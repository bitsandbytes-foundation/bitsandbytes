graph [
  directed 1
  node [
    id 0
    label "assert_all_approx_close"
  ]
  node [
    id 1
    label "print"
  ]
  node [
    id 2
    label "__init__"
  ]
  node [
    id 3
    label "super"
  ]
  node [
    id 4
    label "forward"
  ]
  node [
    id 5
    label "tick"
  ]
  node [
    id 6
    label "tock"
  ]
  node [
    id 7
    label "reset"
  ]
  node [
    id 8
    label "setup"
  ]
  node [
    id 9
    label "teardown"
  ]
  node [
    id 10
    label "test_estimate_quantiles"
  ]
  node [
    id 11
    label "test_quantile_quantization"
  ]
  node [
    id 12
    label "range"
  ]
  node [
    id 13
    label "test_dynamic_quantization"
  ]
  node [
    id 14
    label "sum"
  ]
  node [
    id 15
    label "len"
  ]
  node [
    id 16
    label "test_dynamic_blockwise_quantization"
  ]
  node [
    id 17
    label "id_formatter"
  ]
  node [
    id 18
    label "test_percentile_clipping"
  ]
  node [
    id 19
    label "quant"
  ]
  node [
    id 20
    label "dequant"
  ]
  node [
    id 21
    label "mm_dequant"
  ]
  node [
    id 22
    label "quant_multi"
  ]
  node [
    id 23
    label "quant_multi_chunk"
  ]
  node [
    id 24
    label "quant_minmax"
  ]
  node [
    id 25
    label "mean"
  ]
  node [
    id 26
    label "float"
  ]
  node [
    id 27
    label "test_approx_igemm"
  ]
  node [
    id 28
    label "test_stable_embedding"
  ]
  node [
    id 29
    label "test_igemm"
  ]
  node [
    id 30
    label "get_test_dims"
  ]
  node [
    id 31
    label "test_dim3_igemm"
  ]
  node [
    id 32
    label "test_minmax_igemm"
  ]
  node [
    id 33
    label "min_max"
  ]
  node [
    id 34
    label "test_ibmm"
  ]
  node [
    id 35
    label "test_vector_quant"
  ]
  node [
    id 36
    label "int"
  ]
  node [
    id 37
    label "test_nvidia_transform"
  ]
  node [
    id 38
    label "str"
  ]
  node [
    id 39
    label "test_igemmlt_int"
  ]
  node [
    id 40
    label "test_igemmlt_half"
  ]
  node [
    id 41
    label "test_bench_8bit_training"
  ]
  node [
    id 42
    label "test_dequant_mm"
  ]
  node [
    id 43
    label "test_colrow_absmax"
  ]
  node [
    id 44
    label "test_double_quant"
  ]
  node [
    id 45
    label "test_integrated_igemmlt"
  ]
  node [
    id 46
    label "zip"
  ]
  node [
    id 47
    label "test_igemmlt_row_scale"
  ]
  node [
    id 48
    label "test_row_scale_bench"
  ]
  node [
    id 49
    label "test_transform"
  ]
  node [
    id 50
    label "test_overflow"
  ]
  node [
    id 51
    label "test_coo_double_quant"
  ]
  node [
    id 52
    label "test_spmm_coo"
  ]
  node [
    id 53
    label "test_spmm_bench"
  ]
  node [
    id 54
    label "test_integrated_sparse_decomp"
  ]
  node [
    id 55
    label "test_matmuls"
  ]
  node [
    id 56
    label "test_spmm_coo_very_sparse"
  ]
  node [
    id 57
    label "getattr"
  ]
  node [
    id 58
    label "out_func"
  ]
  node [
    id 59
    label "test_coo2csr"
  ]
  node [
    id 60
    label "test_coo2csc"
  ]
  node [
    id 61
    label "test_spmm_coo_dequant"
  ]
  node [
    id 62
    label "test_bench_matmul"
  ]
  node [
    id 63
    label "test_zeropoint"
  ]
  node [
    id 64
    label "quant_zp"
  ]
  node [
    id 65
    label "test_extract_outliers"
  ]
  node [
    id 66
    label "test_blockwise_cpu_large"
  ]
  node [
    id 67
    label "test_fp8_quant"
  ]
  node [
    id 68
    label "test_few_bit_quant"
  ]
  node [
    id 69
    label "test_kbit_quantile_estimation"
  ]
  node [
    id 70
    label "test_bench_dequantization"
  ]
  node [
    id 71
    label "test_4bit_quant"
  ]
  node [
    id 72
    label "list"
  ]
  node [
    id 73
    label "product"
  ]
  node [
    id 74
    label "test_4bit_compressed_stats"
  ]
  node [
    id 75
    label "test_bench_4bit_dequant"
  ]
  node [
    id 76
    label "test_normal_map_tree"
  ]
  node [
    id 77
    label "test_gemv_4bit"
  ]
  node [
    id 78
    label "test_managed"
  ]
  node [
    id 79
    label "test_gemv_eye_4bit"
  ]
  edge [
    source 0
    target 1
  ]
  edge [
    source 2
    target 3
  ]
  edge [
    source 6
    target 1
  ]
  edge [
    source 7
    target 1
  ]
  edge [
    source 11
    target 12
  ]
  edge [
    source 13
    target 12
  ]
  edge [
    source 13
    target 1
  ]
  edge [
    source 13
    target 14
  ]
  edge [
    source 13
    target 15
  ]
  edge [
    source 16
    target 12
  ]
  edge [
    source 16
    target 14
  ]
  edge [
    source 16
    target 15
  ]
  edge [
    source 16
    target 17
  ]
  edge [
    source 18
    target 12
  ]
  edge [
    source 25
    target 14
  ]
  edge [
    source 25
    target 26
  ]
  edge [
    source 25
    target 15
  ]
  edge [
    source 27
    target 12
  ]
  edge [
    source 27
    target 17
  ]
  edge [
    source 29
    target 12
  ]
  edge [
    source 29
    target 30
  ]
  edge [
    source 29
    target 17
  ]
  edge [
    source 31
    target 12
  ]
  edge [
    source 31
    target 30
  ]
  edge [
    source 31
    target 17
  ]
  edge [
    source 32
    target 12
  ]
  edge [
    source 32
    target 33
  ]
  edge [
    source 32
    target 22
  ]
  edge [
    source 32
    target 21
  ]
  edge [
    source 32
    target 25
  ]
  edge [
    source 32
    target 30
  ]
  edge [
    source 32
    target 17
  ]
  edge [
    source 34
    target 12
  ]
  edge [
    source 34
    target 30
  ]
  edge [
    source 34
    target 17
  ]
  edge [
    source 35
    target 12
  ]
  edge [
    source 35
    target 0
  ]
  edge [
    source 35
    target 36
  ]
  edge [
    source 35
    target 30
  ]
  edge [
    source 35
    target 17
  ]
  edge [
    source 37
    target 38
  ]
  edge [
    source 37
    target 12
  ]
  edge [
    source 37
    target 30
  ]
  edge [
    source 37
    target 17
  ]
  edge [
    source 39
    target 12
  ]
  edge [
    source 39
    target 30
  ]
  edge [
    source 39
    target 17
  ]
  edge [
    source 40
    target 12
  ]
  edge [
    source 40
    target 17
  ]
  edge [
    source 41
    target 1
  ]
  edge [
    source 41
    target 12
  ]
  edge [
    source 42
    target 12
  ]
  edge [
    source 42
    target 0
  ]
  edge [
    source 42
    target 36
  ]
  edge [
    source 42
    target 30
  ]
  edge [
    source 42
    target 17
  ]
  edge [
    source 43
    target 12
  ]
  edge [
    source 43
    target 17
  ]
  edge [
    source 44
    target 12
  ]
  edge [
    source 44
    target 1
  ]
  edge [
    source 44
    target 30
  ]
  edge [
    source 44
    target 17
  ]
  edge [
    source 45
    target 12
  ]
  edge [
    source 45
    target 46
  ]
  edge [
    source 45
    target 30
  ]
  edge [
    source 47
    target 12
  ]
  edge [
    source 47
    target 1
  ]
  edge [
    source 47
    target 14
  ]
  edge [
    source 47
    target 15
  ]
  edge [
    source 47
    target 46
  ]
  edge [
    source 47
    target 30
  ]
  edge [
    source 48
    target 12
  ]
  edge [
    source 48
    target 1
  ]
  edge [
    source 49
    target 12
  ]
  edge [
    source 49
    target 30
  ]
  edge [
    source 49
    target 17
  ]
  edge [
    source 50
    target 1
  ]
  edge [
    source 50
    target 12
  ]
  edge [
    source 51
    target 12
  ]
  edge [
    source 51
    target 30
  ]
  edge [
    source 51
    target 17
  ]
  edge [
    source 52
    target 12
  ]
  edge [
    source 52
    target 0
  ]
  edge [
    source 52
    target 30
  ]
  edge [
    source 52
    target 17
  ]
  edge [
    source 53
    target 12
  ]
  edge [
    source 53
    target 1
  ]
  edge [
    source 54
    target 12
  ]
  edge [
    source 54
    target 30
  ]
  edge [
    source 54
    target 17
  ]
  edge [
    source 55
    target 1
  ]
  edge [
    source 56
    target 57
  ]
  edge [
    source 56
    target 1
  ]
  edge [
    source 56
    target 58
  ]
  edge [
    source 56
    target 0
  ]
  edge [
    source 56
    target 17
  ]
  edge [
    source 61
    target 1
  ]
  edge [
    source 61
    target 0
  ]
  edge [
    source 61
    target 12
  ]
  edge [
    source 62
    target 12
  ]
  edge [
    source 62
    target 1
  ]
  edge [
    source 63
    target 64
  ]
  edge [
    source 63
    target 1
  ]
  edge [
    source 65
    target 12
  ]
  edge [
    source 66
    target 12
  ]
  edge [
    source 66
    target 1
  ]
  edge [
    source 67
    target 12
  ]
  edge [
    source 68
    target 12
  ]
  edge [
    source 69
    target 12
  ]
  edge [
    source 70
    target 1
  ]
  edge [
    source 70
    target 12
  ]
  edge [
    source 71
    target 72
  ]
  edge [
    source 71
    target 73
  ]
  edge [
    source 74
    target 12
  ]
  edge [
    source 75
    target 12
  ]
  edge [
    source 76
    target 72
  ]
  edge [
    source 76
    target 12
  ]
  edge [
    source 77
    target 12
  ]
  edge [
    source 77
    target 36
  ]
  edge [
    source 77
    target 0
  ]
  edge [
    source 77
    target 14
  ]
  edge [
    source 77
    target 15
  ]
  edge [
    source 79
    target 30
  ]
]
