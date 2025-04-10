import time

import pytest
import torch

import bitsandbytes as bnb
from bitsandbytes import functional as F

k = 20

torch.set_printoptions(precision=5,
                       sci_mode=False,
                       linewidth=120,
                       edgeitems=20,
                       threshold=10000)

import triton
import triton.language as tl


@triton.jit
def dequant_kernel(a_ptr, c_ptr, quant_ptr, absmax_ptr, num_paired_elements,
                   QUANT_BLOCK: tl.constexpr, SPLIT_SIZE: tl.constexpr):
    PAIRED_QUANT_BLOCK = QUANT_BLOCK // 2

    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    block_start = pid * SPLIT_SIZE
    offsets = block_start + tl.arange(0, SPLIT_SIZE)
    mask = offsets < num_paired_elements

    a = tl.load(a_ptr + offsets, mask)
    a = a.to(tl.uint8, bitcast=True)

    # higher 4bits from uint8 packed tensor
    higher = (a & 0xf)
    # lower 4bits
    lower = a >> 4

    # apply conversion
    higher_nf4 = tl.load(quant_ptr + higher)
    lower_nf4 = tl.load(quant_ptr + lower)

    abs_blocks_lim = (
        num_paired_elements // PAIRED_QUANT_BLOCK
    ) * PAIRED_QUANT_BLOCK + num_paired_elements % PAIRED_QUANT_BLOCK
    abs_offsets = offsets // PAIRED_QUANT_BLOCK
    mask_blocked = offsets < abs_blocks_lim
    absmax = tl.load(absmax_ptr + abs_offsets, mask_blocked)

    # apply scales
    mul_high = higher_nf4 * absmax
    mul_low = lower_nf4 * absmax

    out_dq = tl.interleave(mul_low, mul_high)

    out_block_start = pid * SPLIT_SIZE * 2
    offs = out_block_start + tl.arange(0, SPLIT_SIZE * 2)
    mask = offs < num_paired_elements * 2
    tl.store(c_ptr + offs, out_dq, mask)


def dequant_nf4_fp16(A_nf4: torch.Tensor, out_B: torch.Tensor,
                     quant_state_code: torch.Tensor, absmax: torch.Tensor,
                     quant_blocksize):
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    A_nf4 = A_nf4.to(device=DEVICE)
    out_B = out_B.to(device=DEVICE)
    quant_state_code = quant_state_code.to(device=DEVICE)
    absmax = absmax.to(device=DEVICE)
    if A_nf4.dtype != torch.uint8:
        print("[Warning] Forcing conversion of {A_nf4.dtype} to uint8.")
        bytes_value = A_nf4.cpu().numpy().tobytes()
        A_nf4 = torch.frombuffer(bytes_value,
                                 dtype=torch.uint8).to(A_nf4.device)

    print("-----------\n")
    print("inp shape: ", A_nf4.shape)

    row, columns = A_nf4.shape
    # It's will be processed as an array, so
    # actual length is row * col
    # Elements are in uint8 format, so interleaved
    # so total amount of data is 2 * elem_count
    number_of_paired_elements = row * columns
    # we assume that split_size > quant_blocksize
    split_size = 128
    # output written will be split_size * 2

    grid = (number_of_paired_elements // split_size + 1, )
    print(" shapes: ", A_nf4.shape, out_B.shape, quant_state_code.shape,
          absmax.shape)
    print(" grid: ", grid)
    print("absmax: ", absmax)
    print("-----------\n")
    dequant_kernel[grid](A_nf4, out_B, quant_state_code, absmax,
                         number_of_paired_elements, quant_blocksize,
                         split_size)
    return out_B


def get_absmax(B: torch.Tensor, blocksize: int):
    n = B.numel()

    blocks = n // blocksize
    blocks += 1 if n % blocksize > 0 else 0
    absmax = torch.zeros((blocks, ), device=B.device, dtype=B.dtype)
    rem = n % blocksize
    has_rem = rem > 0

    # Scale tensor to [-1, 1]
    B_reshaped = B.reshape(n)
    B_com = B_reshaped[:n - rem]
    B_com_reshaped = B_com.reshape(n // blocksize, blocksize)
    absmax[:blocks - has_rem] = torch.abs(B_com_reshaped).max(dim=-1)[0]
    if has_rem:
        absmax[-1] = torch.abs(B_reshaped[n - rem:]).max()
    return absmax


def dequantize_nf4(a: torch.Tensor, out: torch.Tensor,
                   quant_range: torch.Tensor, absmax: torch.Tensor, blocksize):
    a = a.reshape(-1)
    out_dq = torch.empty(a.size(0) * 2, dtype=torch.int32)
    n = out_dq.numel()
    print("initial A: ", a)
    print("initial A: ", a.shape)
    # higher 4bits from uint8 packed tensor
    out_dq[1::2] = a & 0xF
    # lower 4bits
    out_dq[::2] = a >> 4
    out_dq = quant_range[out_dq]
    print("ref out_dq nf4: ", out_dq)
    blocks = n // blocksize
    blocks += 1 if n % blocksize > 0 else 0
    rem = n % blocksize

    has_rem = rem > 0
    if has_rem:
        assert False and "not implemented"
    else:
        print("out_dq reshaped: ", out_dq.view(-1, blocksize).shape)
        # print("absmax reshaped: ", absmax.view(-1, 1))
        print("absmax reshaped: ", absmax.view(-1, 1).shape)
        print("[dequantize_nf4] out shape: ", out.shape)
        print("ref mul: ", out_dq.view(-1, blocksize) * absmax.view(-1, 1))
        out = (out_dq.view(-1, blocksize) * absmax.view(-1, 1)).reshape(
            out.shape).to(out.dtype)
    return out


if triton.runtime.driver.active.get_current_target().backend != "xpu":
    raise RuntimeError("Device is not xpu: ",
                       triton.runtime.driver.active.get_current_target())
else:
    print("curr target: ", triton.runtime.driver.active.get_current_target())
torch.manual_seed(0)


def mm4_ref(batch=1, seq=1, model=2, hidden=128):
    # A = torch.randn(batch, seq, model, device="cpu").half()
    B = torch.empty(hidden, model, dtype=torch.float16, device="cpu")

    quant_state_code = [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ]

    # 128 2
    torch.nn.init.xavier_uniform_(B)

    # B_nf4, state_nf4 = F.quantize_nf4(B)
    B = torch.HalfTensor([[0.14709, -0.10175], [0.20178, -0.13318],
                          [0.01237, 0.00336], [-0.11096, 0.13342],
                          [0.04407, 0.11542], [0.00546, 0.07849],
                          [0.20056, 0.04071], [-0.05621, 0.07513],
                          [0.05728, -0.00399], [-0.14978, -0.18835],
                          [0.06146, 0.10199], [-0.15027, -0.00105],
                          [-0.19067, -0.20264], [-0.07471, 0.07532],
                          [-0.18335, 0.15063], [0.04153, -0.03168],
                          [0.18237, -0.12482], [0.13538, -0.12231],
                          [0.09607, 0.17334], [0.13196, -0.11749],
                          [-0.19971, -0.07111], [-0.21399, 0.13635],
                          [0.18945, -0.09418], [0.04742, -0.10242],
                          [-0.13196, -0.01217], [0.20605, 0.00692],
                          [0.11224, -0.13245], [0.20642, 0.15320],
                          [-0.13599, 0.19006], [-0.07050, -0.13013],
                          [-0.13135, -0.09082], [0.13635, -0.05621],
                          [0.02748, -0.19641], [0.08832, 0.10913],
                          [0.11017, 0.04846], [0.03925, 0.10760],
                          [-0.10992, 0.17139], [-0.13171, 0.20728],
                          [0.05728, -0.19299], [0.18274, -0.10217],
                          [0.12360, -0.06396], [-0.00105, 0.00189],
                          [-0.18237, -0.08789], [-0.15039, 0.20288],
                          [-0.06671, -0.02434], [0.11182, -0.09003],
                          [0.18127, 0.01657], [0.02686, -0.06168],
                          [-0.14014, 0.13660], [0.07471, -0.09381],
                          [-0.07532, -0.08118], [0.14417, -0.03943],
                          [0.18213, 0.07404], [-0.09229, -0.19995],
                          [0.11560, 0.14282], [-0.12000, 0.18628],
                          [-0.08185, 0.21130], [-0.03084, 0.21252],
                          [0.06989, 0.20203], [-0.12927, 0.13489],
                          [-0.19092, 0.08228], [0.12634, 0.14392],
                          [0.02896, -0.00315], [-0.07111, 0.03378],
                          [-0.19299, -0.16003], [0.08289, -0.12085],
                          [-0.03860, 0.09924], [-0.00650, 0.14453],
                          [0.14209, -0.01825], [0.13538, -0.02959],
                          [-0.06482, 0.04407], [0.16907, -0.20483],
                          [-0.00881, 0.20898], [-0.00294, -0.00734],
                          [0.10992, 0.11981], [-0.15466, -0.06439],
                          [-0.18213, -0.09607], [-0.04153, -0.00587],
                          [-0.16321, -0.15088], [-0.05518, -0.05957],
                          [-0.09900, -0.16907], [-0.17371, -0.10303],
                          [0.14209, -0.03293], [0.07111, 0.00587],
                          [0.02937, -0.19763], [0.03336, 0.20728],
                          [0.21375, -0.03925], [-0.08521, -0.14246],
                          [-0.20850, -0.16321], [0.15552, 0.08539],
                          [-0.14648, 0.01678], [0.01469, -0.11792],
                          [-0.16223, 0.20312], [-0.14124, -0.14185],
                          [0.06885, -0.01721], [-0.10571, 0.07239],
                          [-0.17493, 0.08203], [-0.11981, 0.15173],
                          [0.06506, 0.00755], [0.01154, 0.12634],
                          [0.08142, 0.11768], [0.15845, 0.04614],
                          [0.00525, -0.21399], [-0.14270, -0.19092],
                          [-0.15735, -0.18066], [0.01889, -0.19788],
                          [-0.12549, -0.10132], [-0.03943, 0.13782],
                          [0.18384, 0.15991], [0.03314, -0.17944],
                          [0.06464, 0.13843], [-0.15845, 0.08557],
                          [0.18311, 0.03818], [0.07239, -0.14368],
                          [0.19177, -0.10492], [-0.20557, -0.01993],
                          [0.16223, 0.21423], [0.14355, 0.01196],
                          [-0.12695, -0.09485], [0.18103, -0.12646],
                          [0.06653, -0.13928], [0.05286, 0.03925],
                          [0.02811, 0.17725], [-0.19934, -0.03461],
                          [-0.03986, -0.14392], [0.06042, 0.07849],
                          [-0.00965, 0.06232], [0.14062, 0.02119]])

    B_nf4 = torch.Tensor([[226], [241], [135], [45], [173], [123], [249], [75],
                          [167], [16], [172], [23], [0], [59], [14], [149],
                          [226], [210], [206], [210], [4], [13], [243], [162],
                          [22], [247], [209], [254], [31], [66], [19], [212],
                          [144], [205], [218], [157], [46], [31], [160], [226],
                          [212], [119], [3], [31], [70], [211], [232], [148],
                          [30], [179], [51], [229], [235], [48], [222], [47],
                          [63], [95], [191], [45], [11], [222],
                          [151], [73], [1], [178], [92], [126], [230], [213],
                          [74], [224], [127], [119], [221], [20], [3], [87],
                          [17], [68], [33], [18], [229], [183], [144], [159],
                          [245], [49], [1], [236], [24], [130], [31], [17],
                          [182], [43], [27], [46], [183], [141], [189], [234],
                          [112], [16], [17], [128], [34], [94], [238], [145],
                          [190], [28], [233], [177], [242], [6], [239], [232],
                          [35], [226], [177], [169], [158], [5], [81], [171],
                          [122], [232]])
    B_nf4 = B_nf4.to(dtype=torch.uint8)
    blocksize = 64

    absmax = get_absmax(B, blocksize)
    quant_state_code = torch.HalfTensor(quant_state_code)

    B_dq = torch.empty_like(B, dtype=torch.float16)
    B_dq = dequantize_nf4(B_nf4, B_dq, quant_state_code, absmax, blocksize)

    out_B = torch.empty_like(B, dtype=torch.float16)
    dequant_nf4_fp16(B_nf4, out_B, quant_state_code, absmax, blocksize)

    print("from_kernel: ", out_B.view(-1, blocksize))
    print("from_py: ", B_dq.view(-1, blocksize))
    max_diff = (out_B - B_dq)
    assert torch.allclose(
        out_B, B_dq, atol=1e-2, rtol=0
    ), f"dequantized weight not close to original, max diff: {max_diff} First failed"

    B_1 = torch.HalfTensor([[-0.01041, 0.14868, 0.05603, 0.19592],
                            [0.20935, 0.17786, -0.13501, 0.20911],
                            [-0.04852, -0.01625, -0.01375, 0.13208],
                            [-0.00479, 0.15198, 0.19678, -0.12140],
                            [-0.13477, 0.07831, -0.18201, 0.03333],
                            [-0.17603, 0.18201, -0.10101, 0.18811],
                            [0.05270, 0.03622, 0.14722, -0.15393],
                            [0.09808, 0.06995, 0.05582, -0.14697],
                            [-0.12372, -0.05560, -0.17603, 0.09015],
                            [0.11957, -0.01917, -0.09204, 0.18726],
                            [0.20032, -0.16455, -0.13855, 0.08411],
                            [-0.13660, -0.10291, -0.14563, 0.07312],
                            [0.18372, -0.19434, -0.11871, -0.17969],
                            [0.13892, -0.21265, -0.10748, -0.07727],
                            [0.14343, 0.01645, 0.13916, -0.18958],
                            [-0.05872, 0.09705, 0.12286, -0.02374],
                            [0.19238, 0.06372, -0.12598, -0.06082],
                            [-0.08539, -0.13770, -0.09790, -0.17725],
                            [-0.07581, 0.18494, -0.04684, -0.01625],
                            [0.12854, -0.06915, 0.09705, -0.01479],
                            [0.08685, 0.21240, 0.09039, 0.11932],
                            [-0.09369, 0.06812, -0.12390, 0.03229],
                            [0.02271, -0.15552, 0.07416, -0.10162],
                            [-0.05103, -0.17847, -0.17175, -0.20081],
                            [0.21204, 0.20508, -0.11768, 0.09998],
                            [0.20227, -0.06995, -0.17725, 0.07953],
                            [-0.13477, 0.20447, 0.13806, 0.10223],
                            [0.08521, -0.07434, -0.09686, 0.17810],
                            [0.07391, -0.10913, -0.10913, 0.03812],
                            [-0.08038, -0.08521, 0.17639, 0.09454],
                            [-0.08228, 0.21143, 0.13367, 0.14868],
                            [-0.04706, 0.03497, -0.01688, 0.16284],
                            [0.13831, -0.09082, 0.20886, -0.11432],
                            [0.13892, 0.21094, 0.19666, 0.04977],
                            [-0.03333, 0.04373, 0.04498, 0.07562],
                            [-0.14575, 0.18408, 0.16687, 0.15369],
                            [-0.12079, 0.09351, -0.04205, 0.02187],
                            [-0.04562, -0.11914, 0.21094, -0.03894],
                            [0.03082, -0.20300, -0.13354, -0.06604],
                            [0.03958, -0.13245, 0.13098, 0.13281],
                            [-0.03604, -0.08600, -0.09039, 0.13269],
                            [0.16809, -0.11432, -0.03415, -0.13281],
                            [0.06891, -0.21094, 0.09912, -0.17615],
                            [0.05768, 0.16187, -0.05499, -0.20142],
                            [0.11707, 0.03290, 0.03082, -0.12683],
                            [0.08832, -0.06458, 0.18701, 0.00000],
                            [0.09412, -0.05643, -0.00999, 0.15808],
                            [0.17310, 0.01333, -0.18958, -0.07477],
                            [-0.06519, -0.01229, -0.19788, 0.14600],
                            [-0.07349, -0.08789, -0.02020, 0.13574],
                            [-0.11163, 0.10663, 0.10309, -0.21179],
                            [0.00479, 0.05643, 0.08789, -0.16260],
                            [-0.16101, -0.00125, 0.07166, -0.18884],
                            [0.09393, 0.06143, -0.14490, -0.06934],
                            [0.09369, -0.06683, 0.19312, 0.11371],
                            [0.11578, 0.06500, -0.15686, 0.13501],
                            [-0.17456, 0.07269, 0.13635, -0.10541],
                            [-0.02499, -0.21216, -0.03937, -0.07312],
                            [0.16870, 0.06082, -0.18384, 0.12244],
                            [0.21179, 0.01917, -0.10748, -0.13477],
                            [-0.15405, 0.13806, -0.03812, 0.04706],
                            [0.20264, -0.01521, -0.11810, -0.19055],
                            [-0.05872, 0.18677, 0.00125, -0.02936],
                            [0.03748, -0.04205, 0.06750, 0.13220],
                            [-0.14868, 0.11707, 0.12793, 0.03748],
                            [-0.03104, -0.14429, -0.11517, 0.16785],
                            [0.03436, -0.11249, -0.13306, 0.05997],
                            [-0.02998, 0.12512, 0.06354, 0.00500],
                            [-0.14343, -0.17517, 0.02333, 0.18640],
                            [0.05978, -0.18115, 0.01854, -0.21143],
                            [-0.18225, 0.19971, 0.01688, 0.20532],
                            [0.15332, -0.16016, 0.14124, 0.13391],
                            [-0.08685, -0.10455, 0.16370, -0.19177],
                            [-0.18909, -0.09308, -0.13000, -0.15271],
                            [-0.19763, 0.20325, 0.19873, 0.20850],
                            [0.04437, -0.18469, -0.14771, -0.09058],
                            [-0.11328, -0.00396, 0.02541, 0.10327],
                            [-0.00021, -0.19019, -0.08307, -0.15308],
                            [-0.03665, -0.05978, 0.05728, -0.10767],
                            [0.02707, -0.15186, 0.15198, 0.19824],
                            [0.15540, -0.09100, 0.15784, -0.18176],
                            [-0.10431, 0.14563, 0.10327, 0.00437],
                            [-0.12573, -0.16052, 0.02687, -0.08832],
                            [0.01521, -0.14343, -0.09723, -0.19788],
                            [0.19727, -0.18115, -0.18286, 0.00396],
                            [-0.06708, -0.09015, 0.11810, 0.04062],
                            [0.00979, -0.20911, 0.09998, -0.12976],
                            [0.19788, -0.01874, 0.18262, -0.12122],
                            [0.01625, -0.13367, 0.18909, -0.10828],
                            [0.03479, -0.17261, -0.21143, -0.15369],
                            [0.09705, -0.11328, -0.12061, 0.10309],
                            [-0.10327, 0.19556, 0.01250, -0.09308],
                            [-0.14636, -0.20789, -0.06476, 0.17456],
                            [0.05206, 0.17810, -0.00229, -0.16284],
                            [0.20728, -0.12329, -0.11725, 0.02728],
                            [0.19824, 0.19934, -0.06641, 0.18579],
                            [0.09100, 0.05392, 0.17810, -0.17517],
                            [0.15637, 0.13660, -0.15601, -0.18579],
                            [0.05185, 0.13013, -0.04416, 0.09265],
                            [-0.13391, -0.17493, -0.03644, -0.07666],
                            [-0.06934, 0.01645, 0.20056, 0.15845],
                            [-0.19141, -0.02396, -0.07727, 0.09851],
                            [-0.15637, 0.10394, -0.12915, 0.17822],
                            [-0.20325, -0.19824, -0.04291, -0.10101],
                            [-0.01625, 0.00021, 0.20801, 0.07080],
                            [-0.03333, 0.12201, -0.17371, -0.08643],
                            [-0.15222, 0.17102, -0.12769, 0.04352],
                            [0.05853, -0.04602, -0.06082, 0.05914],
                            [0.15784, -0.20618, -0.17957, 0.04187],
                            [-0.06873, -0.08246, -0.03061, 0.16431],
                            [-0.15686, 0.16992, -0.14514, 0.15955],
                            [0.20801, -0.15808, 0.10767, 0.16394],
                            [-0.17493, -0.11603, 0.06476, -0.21155],
                            [-0.13477, 0.16907, 0.02936, -0.09015],
                            [-0.20056, 0.14661, 0.08270, -0.13062],
                            [-0.13391, 0.09851, -0.18616, -0.08850],
                            [0.10120, -0.20239, -0.16931, 0.11206],
                            [0.10913, 0.18958, -0.20789, -0.11890],
                            [-0.17456, 0.03540, -0.02124, -0.10394],
                            [0.14600, 0.06934, -0.06061, -0.04437],
                            [0.18579, -0.10645, -0.20557, 0.09161],
                            [0.12622, -0.15662, -0.12433, 0.09039],
                            [0.07977, -0.03769, 0.02979, 0.09601],
                            [-0.04749, 0.12268, 0.01811, 0.08539],
                            [0.03976, 0.20203, 0.04248, -0.03250],
                            [-0.02353, 0.11517, 0.07538, 0.07062],
                            [0.08911, 0.04437, 0.07642, 0.10785],
                            [-0.12830, -0.00771, 0.06854, -0.11475]])

    B_nf4_1 = torch.Tensor([[110], [175], [254], [31], [86], [109], [126],
                            [242], [27], [9], [30], [47], [169], [225], [203],
                            [161], [36], [28], [214], [63], [241], [28], [18],
                            [27], [240], [33], [224], [35], [232], [224], [76],
                            [214], [251], [36], [49], [33], [63], [86], [212],
                            [198], [207], [205], [59], [41], [129], [178],
                            [65], [16], [255], [44], [244], [27], [31], [236],
                            [195], [62], [178], [41], [51], [236], [63], [222],
                            [89], [110], [227], [242], [239], [250], [90],
                            [171], [31], [238], [44], [88], [82], [245], [144],
                            [20], [145], [221], [83], [61], [226], [81], [176],
                            [193], [174], [64], [217], [146], [196], [247],
                            [196], [110], [232], [3], [70], [14], [51], [109],
                            [45], [192], [122], [193], [23], [176], [202],
                            [20], [196], [253], [219], [29], [27], [226], [96],
                            [83], [234], [13], [248], [33], [30], [90], [246],
                            [32], [79], [117], [149], [189], [29], [217], [81],
                            [46], [146], [26], [93], [183], [17], [143], [160],
                            [128], [15], [143], [225], [237], [50], [224], [3],
                            [17], [15], [255], [160], [19], [39], [140], [112],
                            [49], [84], [162], [145], [239], [227], [224
                                                                     ], [46],
                            [199], [33], [147], [129], [32], [240], [7], [67],
                            [217], [128], [193], [246], [242], [129], [242],
                            [145], [1], [194], [44], [47], [131], [16], [78],
                            [174], [113], [242], [41], [255], [79], [202],
                            [225], [238], [16], [173], [92], [17], [83], [72],
                            [254], [6], [60], [28], [30], [0], [82], [103],
                            [251], [93], [19], [30], [26], [165], [74], [224],
                            [9], [67], [94], [30], [30], [241], [222], [18],
                            [176], [30], [147], [14], [193], [28], [3], [192],
                            [29], [223], [2], [25], [98], [235], [69], [242],
                            [12], [209], [44], [181], [156], [93], [140
                                                                    ], [159],
                            [149], [109], [187], [202], [189], [39], [178]])
    B_nf4_1 = B_nf4_1.to(dtype=torch.uint8)

    absmax = get_absmax(B_1, blocksize)
    out_B = torch.empty_like(B_1, dtype=torch.float16)

    B_dq_1 = torch.empty_like(B_1, dtype=torch.float16)
    B_dq_1 = dequantize_nf4(B_nf4_1, B_dq_1, quant_state_code, absmax,
                            blocksize)

    dequant_nf4_fp16(B_nf4_1, out_B, quant_state_code, absmax, blocksize)

    max_diff = (out_B - B_dq_1)
    assert torch.allclose(
        out_B, B_dq_1, atol=1e-2, rtol=0
    ), f"dequantized weight not close to original, max diff: {max_diff} Second failed"


# mm4_ref()
# exit(0)


# 1, 1, 3584, 512
def mm4(batch=1, seq=1, model=1, hidden=736):
    A = torch.randn(batch, seq, model, device="cpu").half()
    B = torch.empty(hidden, model, dtype=torch.float16, device="cpu")
    torch.nn.init.xavier_uniform_(B)

    quant_state_code = [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ]

    # print("original B: ", B)
    print("original B: ", B.shape)
    B_nf4, state_nf4 = F.quantize_nf4(B)
    # print("quantize B_nf4: ", B_nf4)

    B_dq = F.dequantize_nf4(B_nf4, state_nf4)
    # print ("B - B_dq: ", B - B_dq)
    print("B dq: ", B_dq.dtype, " B orig shape: ", B.dtype)
    print("B dq: ", B_dq.shape, " B orig shape: ", B.shape)

    blocksize = 64
    if blocksize != state_nf4.blocksize:
        raise RuntimeError("Mismatch of quantization blocksizes: ", blocksize,
                           " != ", state_nf4.blocksize)
    absmax = get_absmax(B, blocksize)
    quant_state_code = torch.HalfTensor(quant_state_code)

    # out_B_dq = torch.empty_like(B_dq, dtype=torch.float16)
    # out_B_dq = dequantize_nf4(B_nf4, out_B_dq, quant_state_code, absmax, blocksize)
    out_B_dq = B_dq

    out_B = torch.empty_like(B_dq, dtype=torch.float16)
    out_B = dequant_nf4_fp16(B_nf4, out_B, quant_state_code, absmax, blocksize)

    # grid = (1, )
    # dequant_kernel[grid](B_nf4, out_B, quant_state_code, state_nf4.absmax, B_nf4.shape[0], B_nf4.shape[1], 64, 128)

    out_B = out_B.cpu()
    out_B_dq = out_B_dq.cpu()
    max_diff = (out_B.cpu() - out_B_dq.cpu()).max()
    assert torch.allclose(
        out_B, out_B_dq, atol=1e-2, rtol=0
    ), f"dequantized weight not close to original, max diff: {max_diff} Second failed"
    print("Passed!")


mm4()
