import os
from os.path import join
import shutil
import time
import uuid

from lion_pytorch import Lion
import pytest
import torch

import bitsandbytes as bnb
import bitsandbytes.functional as F
from tests.helpers import describe_dtype, id_formatter

# import apex

k = 20


def assert_most_approx_close(a, b, rtol=1e-3, atol=1e-3, max_error_count=0):
    idx = torch.isclose(a, b, rtol=rtol, atol=atol)
    error_count = (idx == 0).sum().item()
    if error_count > max_error_count:
        print(f"Too many values not close: assert {error_count} < {max_error_count}")
        torch.testing.assert_close(a, b, rtol=rtol, atol=atol)


def get_temp_dir():
    path = f"/tmp/autoswap/{uuid.uuid4()}"
    os.makedirs(path, exist_ok=True)
    return path


def rm_path(path):
    shutil.rmtree(path)


str2optimizers = {}

## TODO: maybe remove these three.
str2optimizers["adam_pytorch"] = (None, torch.optim.Adam, bnb.optim.Adam)
str2optimizers["lion_pytorch"] = (None, Lion, bnb.optim.Lion)
str2optimizers["momentum_pytorch"] = (
    None,
    lambda pxx: torch.optim.SGD(pxx, 0.01, 0.9),
    bnb.optim.Adam,
)

str2optimizers["adam"] = (torch.optim.Adam, bnb.optim.Adam)
str2optimizers["adam8bit"] = (torch.optim.Adam, lambda pxx: bnb.optim.Adam8bit(pxx, block_wise=False))
str2optimizers["adam8bit_blockwise"] = (torch.optim.Adam, lambda pxx: bnb.optim.Adam8bit(pxx, block_wise=True))
str2optimizers["paged_adam"] = (torch.optim.Adam, bnb.optim.PagedAdam)
str2optimizers["paged_adamw"] = (torch.optim.AdamW, bnb.optim.PagedAdamW)
str2optimizers["paged_adam8bit_blockwise"] = (
    torch.optim.Adam,
    lambda pxx: bnb.optim.PagedAdam8bit(pxx, block_wise=True),
)
str2optimizers["paged_adamw8bit_blockwise"] = (
    torch.optim.AdamW,
    lambda pxx: bnb.optim.PagedAdamW8bit(pxx, block_wise=True),
)

str2optimizers["ademamix"] = (bnb.optim.ademamix._ReferenceAdEMAMix, bnb.optim.AdEMAMix)
str2optimizers["ademamix8bit_blockwise"] = (
    bnb.optim.ademamix._ReferenceAdEMAMix,
    lambda pxx: bnb.optim.AdEMAMix8bit(pxx),
)
str2optimizers["paged_ademamix"] = (bnb.optim.ademamix._ReferenceAdEMAMix, bnb.optim.PagedAdEMAMix)
str2optimizers["paged_ademamix8bit_blockwise"] = (
    bnb.optim.ademamix._ReferenceAdEMAMix,
    lambda pxx: bnb.optim.PagedAdEMAMix8bit(pxx),
)
str2optimizers["ademamix_scheduled"] = (
    lambda pxx: bnb.optim.ademamix._ReferenceAdEMAMix(pxx, t_alpha=k, t_beta3=k),
    lambda pxx: bnb.optim.AdEMAMix(pxx, t_alpha=k, t_beta3=k),
)
str2optimizers["paged_ademamix_scheduled"] = (
    lambda pxx: bnb.optim.ademamix._ReferenceAdEMAMix(pxx, t_alpha=k, t_beta3=k),
    lambda pxx: bnb.optim.PagedAdEMAMix(pxx, t_alpha=k, t_beta3=k),
)
str2optimizers["ademamix8bit_blockwise_scheduled"] = (
    lambda pxx: bnb.optim.ademamix._ReferenceAdEMAMix(pxx, t_alpha=100, t_beta3=100),
    lambda pxx: bnb.optim.AdEMAMix8bit(pxx, t_alpha=100, t_beta3=100),
)
str2optimizers["paged_ademamix8bit_blockwise_scheduled"] = (
    lambda pxx: bnb.optim.ademamix._ReferenceAdEMAMix(pxx, t_alpha=100, t_beta3=100),
    lambda pxx: bnb.optim.PagedAdEMAMix8bit(pxx, t_alpha=100, t_beta3=100),
)

str2optimizers["lion"] = (Lion, bnb.optim.Lion)
str2optimizers["lion8bit"] = (Lion, lambda pxx: bnb.optim.Lion8bit(pxx, block_wise=False))
str2optimizers["lion8bit_blockwise"] = (Lion, lambda pxx: bnb.optim.Lion8bit(pxx, block_wise=True))
str2optimizers["paged_lion"] = (Lion, bnb.optim.PagedLion)
str2optimizers["paged_lion8bit_blockwise"] = (Lion, lambda pxx: bnb.optim.PagedLion8bit(pxx, block_wise=True))

str2optimizers["momentum"] = (
    lambda pxx: torch.optim.SGD(pxx, 0.01, 0.9),
    lambda pxx: bnb.optim.SGD(pxx, 0.01, 0.9, block_wise=False),
)
str2optimizers["momentum8bit"] = (
    lambda pxx: torch.optim.SGD(pxx, 0.01, 0.9),
    lambda pxx: bnb.optim.SGD8bit(pxx, 0.01, 0.9, block_wise=False),
)
str2optimizers["momentum8bit_blockwise"] = (
    lambda pxx: torch.optim.SGD(pxx, 0.01, 0.9),
    lambda pxx: bnb.optim.SGD8bit(pxx, 0.01, 0.9, block_wise=True),
)

str2optimizers["rmsprop"] = (
    lambda pxx: torch.optim.RMSprop(pxx, 0.01, 0.9),
    lambda pxx: bnb.optim.RMSprop(pxx, 0.01, 0.9, block_wise=False),
)
str2optimizers["rmsprop8bit"] = (
    lambda pxx: torch.optim.RMSprop(pxx, 0.01, 0.9),
    lambda pxx: bnb.optim.RMSprop8bit(pxx, 0.01, 0.9, block_wise=False),
)
str2optimizers["rmsprop8bit_blockwise"] = (
    lambda pxx: torch.optim.RMSprop(pxx, 0.01, 0.9),
    lambda pxx: bnb.optim.RMSprop8bit(pxx, 0.01, 0.9, block_wise=True),
)

str2statenames = {}
str2statenames["adam"] = [("exp_avg", "state1"), ("exp_avg_sq", "state2")]
str2statenames["paged_adamw"] = [("exp_avg", "state1"), ("exp_avg_sq", "state2")]
str2statenames["paged_adam"] = [("exp_avg", "state1"), ("exp_avg_sq", "state2")]
str2statenames["lion"] = [("exp_avg", "state1")]
str2statenames["paged_lion"] = [("exp_avg", "state1")]
str2statenames["momentum"] = [("momentum_buffer", "state1")]
str2statenames["lamb"] = [("exp_avg", "state1"), ("exp_avg_sq", "state2")]
str2statenames["rmsprop"] = [("square_avg", "state1")]
str2statenames["adam8bit"] = [("exp_avg", "state1", "qmap1", "max1"), ("exp_avg_sq", "state2", "qmap2", "max2")]
str2statenames["lamb8bit"] = [("exp_avg", "state1", "qmap1", "max1"), ("exp_avg_sq", "state2", "qmap2", "max2")]
str2statenames["adam8bit_blockwise"] = [
    ("exp_avg", "state1", "qmap1", "absmax1"),
    ("exp_avg_sq", "state2", "qmap2", "absmax2"),
]
str2statenames["paged_adam8bit_blockwise"] = [
    ("exp_avg", "state1", "qmap1", "absmax1"),
    ("exp_avg_sq", "state2", "qmap2", "absmax2"),
]
str2statenames["paged_adamw8bit_blockwise"] = [
    ("exp_avg", "state1", "qmap1", "absmax1"),
    ("exp_avg_sq", "state2", "qmap2", "absmax2"),
]
str2statenames["momentum8bit"] = [("momentum_buffer", "state1", "qmap1", "max1")]
str2statenames["lion8bit"] = [("exp_avg", "state1", "qmap1", "max1")]
str2statenames["momentum8bit_blockwise"] = [("momentum_buffer", "state1", "qmap1", "absmax1")]
str2statenames["rmsprop8bit"] = [("square_avg", "state1", "qmap1", "max1")]
str2statenames["rmsprop8bit_blockwise"] = [("square_avg", "state1", "qmap1", "absmax1")]
str2statenames["lion8bit_blockwise"] = [("exp_avg", "state1", "qmap1", "absmax1")]
str2statenames["paged_lion8bit_blockwise"] = [("exp_avg", "state1", "qmap1", "absmax1")]

str2statenames["ademamix"] = str2statenames["ademamix_scheduled"] = [("m1_m2", "state1"), ("nu", "state2")]
str2statenames["paged_ademamix"] = str2statenames["paged_ademamix_scheduled"] = [("m1_m2", "state1"), ("nu", "state2")]
str2statenames["ademamix8bit_blockwise"] = str2statenames["ademamix8bit_blockwise_scheduled"] = [
    ("m1_m2", "state1", "qmap1", "absmax1"),
    ("nu", "state2", "qmap2", "absmax2"),
]
str2statenames["paged_ademamix8bit_blockwise"] = [
    ("m1_m2", "state1", "qmap1", "absmax1"),
    ("nu", "state2", "qmap2", "absmax2"),
]

optimizer_names_32bit = [
    "adam",
    "paged_adamw",
    "paged_adam",
    "momentum",
    "rmsprop",
    "lion",
    "paged_lion",
    "ademamix",
    "ademamix_scheduled",
    "paged_ademamix",
    "paged_ademamix_scheduled",
]


@pytest.mark.parametrize("optim_name", optimizer_names_32bit, ids=id_formatter("opt"))
@pytest.mark.parametrize("gtype", [torch.float32, torch.float16, torch.bfloat16], ids=describe_dtype)
@pytest.mark.parametrize("dim1", [1024], ids=id_formatter("dim1"))
@pytest.mark.parametrize("dim2", [32, 1024, 4097, 1], ids=id_formatter("dim2"))
def test_optimizer32bit(dim1, dim2, gtype, optim_name):
    if gtype == torch.bfloat16 and optim_name in ["momentum", "rmsprop"]:
        pytest.skip()
    if dim1 == 1 and dim2 == 1:
        return
    p1 = torch.randn(dim1, dim2, device="cuda", dtype=gtype) * 0.1
    p2 = p1.clone()
    p1 = p1.float()

    torch_optimizer = str2optimizers[optim_name][0]([p1])
    bnb_optimizer = str2optimizers[optim_name][1]([p2])

    if gtype == torch.float32:
        atol, rtol = 1e-6, 1e-5
    elif gtype == torch.bfloat16:
        atol, rtol = 1e-3, 1e-2
    else:
        atol, rtol = 1e-4, 1e-3

    for i in range(k):
        g = torch.randn(dim1, dim2, device="cuda", dtype=gtype) * 0.01
        p1.grad = g.clone().float()
        p2.grad = g.clone()

        bnb_optimizer.step()
        torch_optimizer.step()

        for name1, name2 in str2statenames[optim_name]:
            torch.testing.assert_close(
                torch_optimizer.state[p1][name1],
                bnb_optimizer.state[p2][name2].cuda(),
                atol=atol,
                rtol=rtol,
            )

        # since Lion can have pretty noisy updates where things lie at the boundary
        # allow up to 10 errors for Lion
        assert_most_approx_close(p1, p2.float(), atol=atol, rtol=rtol, max_error_count=10)

        if i % (k // 5) == 0 and i > 0:
            path = get_temp_dir()
            torch.save(bnb_optimizer.state_dict(), join(path, "opt.pt"))
            del bnb_optimizer
            bnb_optimizer = None
            bnb_optimizer = str2optimizers[optim_name][1]([p2])
            bnb_optimizer.load_state_dict(torch.load(join(path, "opt.pt")))
            rm_path(path)
            # since Lion can have pretty noisy updates where things lie at the boundary
            # allow up to 10 errors for Lion
            assert_most_approx_close(p1, p2.float(), atol=atol, rtol=rtol, max_error_count=10)
            for name1, name2 in str2statenames[optim_name]:
                # since Lion can have pretty noisy updates where things lie at the boundary
                # allow up to 10 errors for Lion
                assert_most_approx_close(
                    torch_optimizer.state[p1][name1],
                    bnb_optimizer.state[p2][name2],
                    atol=atol,
                    rtol=rtol,
                    max_error_count=10,
                )

        if gtype != torch.float32:
            # the adam buffers should also be close because they are 32-bit
            # but the parameters can diverge because they are 16-bit
            # the difference grow larger and larger with each update
            # --> copy the state to keep weights close
            p1.data = p1.data.to(p2.dtype).float()
            p2.copy_(p1.data)
            torch.testing.assert_close(p1.to(p2.dtype), p2)
        if optim_name in ["lars", "lamb"]:
            assert bnb_optimizer.state[p2]["unorm_vec"] > 0.0


@pytest.mark.parametrize("dim1", [1024], ids=id_formatter("dim1"))
@pytest.mark.parametrize("dim2", [32, 1024, 4097], ids=id_formatter("dim2"))
@pytest.mark.parametrize("gtype", [torch.float32, torch.float16], ids=describe_dtype)
def test_global_config(dim1, dim2, gtype):
    if dim1 == 1 and dim2 == 1:
        return
    p1 = torch.randn(dim1, dim2, device="cpu", dtype=gtype) * 0.1
    p2 = torch.randn(dim1, dim2, device="cpu", dtype=gtype) * 0.1
    p3 = torch.randn(dim1, dim2, device="cpu", dtype=gtype) * 0.1
    mask = torch.rand_like(p2) < 0.1
    beta1 = 0.9
    beta2 = 0.999
    lr = 0.001
    eps = 1e-8

    bnb.optim.GlobalOptimManager.get_instance().initialize()
    bnb.optim.GlobalOptimManager.get_instance().override_config(p3, "optim_bits", 8)

    bnb.optim.GlobalOptimManager.get_instance().register_parameters([p1, p2, p3])
    p1 = p1.cuda()
    p2 = p2.cuda()
    p3 = p3.cuda()

    adam2 = bnb.optim.Adam([p1, p2, p3], lr, (beta1, beta2), eps)

    if gtype == torch.float32:
        atol, rtol = 1e-6, 1e-5
    else:
        atol, rtol = 1e-4, 1e-3

    for i in range(50):
        g1 = torch.randn(dim1, dim2, device="cuda", dtype=gtype) * 0.1 + 0.001
        g2 = torch.randn(dim1, dim2, device="cuda", dtype=gtype) * 0.1 + 0.001
        g3 = torch.randn(dim1, dim2, device="cuda", dtype=gtype) * 0.1 + 0.001
        p1.grad = g1
        p2.grad = g2
        p3.grad = g3

        adam2.step()

        assert adam2.state[p3]["state1"].dtype == torch.uint8
        assert adam2.state[p3]["state2"].dtype == torch.uint8


optimizer_names_8bit = [
    "adam8bit",
    "lion8bit",
    "momentum8bit",
    "rmsprop8bit",
    "adam8bit_blockwise",
    "lion8bit_blockwise",
    "momentum8bit_blockwise",
    "rmsprop8bit_blockwise",
    "ademamix8bit_blockwise",
    "ademamix8bit_blockwise_scheduled",
]


@pytest.mark.parametrize("optim_name", optimizer_names_8bit, ids=id_formatter("opt"))
@pytest.mark.parametrize("gtype", [torch.float32, torch.float16, torch.bfloat16], ids=describe_dtype)
@pytest.mark.parametrize("dim2", [32, 1024, 4097], ids=id_formatter("dim2"))
@pytest.mark.parametrize("dim1", [1024], ids=id_formatter("dim1"))
def test_optimizer8bit(dim1, dim2, gtype, optim_name):
    torch.set_printoptions(precision=6)

    if gtype == torch.bfloat16 and "blockwise" not in optim_name:
        pytest.skip()

    if dim1 == 1 and dim2 == 1:
        return
    p1 = torch.randn(dim1, dim2, device="cuda", dtype=gtype) * 0.1
    p2 = p1.clone()
    p1 = p1.float()
    blocksize = 256

    torch_optimizer = str2optimizers[optim_name][0]([p1])
    bnb_optimizer = str2optimizers[optim_name][1]([p2])

    if gtype == torch.float32:
        atol, rtol = 3e-3, 1e-3
        patol, prtol = 1e-5, 1e-3
    elif gtype == torch.bfloat16:
        atol, rtol = 3e-3, 1e-3
        patol, prtol = 1e-4, 1e-2
    else:
        atol, rtol = 3e-3, 1e-3
        patol, prtol = 1e-5, 1e-3

    errors = []
    relerrors = []

    for i in range(50):
        g = torch.randn(dim1, dim2, device="cuda", dtype=gtype) * 0.01
        p1.grad = g.clone().float()
        p2.grad = g.clone()

        bnb_optimizer.step()
        torch_optimizer.step()

        # since Lion can have pretty noisy updates where things lie at the boundary
        assert_most_approx_close(p1, p2.float(), patol, prtol, max_error_count=0)

        dequant_states = []
        for name1, name2, qmap, max_val in str2statenames[optim_name]:
            # print(bnb_optimizer.state[p2][max_val], name1)
            if "blockwise" in optim_name:
                ## For AdEMAMix, we need to dequantize [p2][name2][0] and [p2][name2][1]
                ## separately and then stack them. The qmap is shared, but absmax is also stacked.
                if optim_name == "ademamix8bit_blockwise" and name1 == "m1_m2":
                    m1 = F.dequantize_blockwise(
                        code=bnb_optimizer.state[p2][qmap],
                        absmax=bnb_optimizer.state[p2][max_val][0],
                        A=bnb_optimizer.state[p2][name2][0],
                        blocksize=blocksize,
                    )
                    m2 = F.dequantize_blockwise(
                        code=bnb_optimizer.state[p2][qmap],
                        absmax=bnb_optimizer.state[p2][max_val][1],
                        A=bnb_optimizer.state[p2][name2][1],
                        blocksize=blocksize,
                    )

                    s1 = torch.stack((m1, m2))

                else:
                    s1 = F.dequantize_blockwise(
                        code=bnb_optimizer.state[p2][qmap],
                        absmax=bnb_optimizer.state[p2][max_val],
                        A=bnb_optimizer.state[p2][name2],
                        blocksize=blocksize,
                    )
            else:
                s1 = F.dequantize(
                    code=bnb_optimizer.state[p2][qmap],
                    absmax=bnb_optimizer.state[p2][max_val],
                    A=bnb_optimizer.state[p2][name2],
                )
            num_not_close = torch.isclose(torch_optimizer.state[p1][name1], s1, atol=atol, rtol=rtol) == 0
            # assert num_not_close.sum().item() < 20
            dequant_states.append(s1.clone())

        err = torch.abs(p1 - p2)
        relerr = err / (torch.abs(p1) + 1e-9)
        if g.dtype == torch.bfloat16:
            assert err.mean() <= 0.00017
            assert relerr.mean() <= 0.0016
        else:
            assert err.mean() < 0.00006
            assert relerr.mean() < 0.0006

        errors.append(err.mean().item())
        relerrors.append(relerr.mean().item())

        if i % 10 == 0 and i > 0:
            for (name1, name2, qmap, max_val), s in zip(str2statenames[optim_name], dequant_states):
                s1cpy = s.clone()
                raws1cpy = bnb_optimizer.state[p2][name2].clone()
                qmap1 = bnb_optimizer.state[p2][qmap].clone()

                path = get_temp_dir()
                torch.save(bnb_optimizer.state_dict(), join(path, "opt.pt"))
                del bnb_optimizer
                bnb_optimizer = None
                bnb_optimizer = str2optimizers[optim_name][1]([p2])
                bnb_optimizer.load_state_dict(torch.load(join(path, "opt.pt")))
                rm_path(path)
                torch.testing.assert_close(raws1cpy, bnb_optimizer.state[p2][name2])
                torch.testing.assert_close(qmap1, bnb_optimizer.state[p2][qmap])

                if "blockwise" in optim_name:
                    ## For AdEMAMix, we need to dequantize [p2][name2][0] and [p2][name2][1]
                    ## separately and then stack them. The qmap is shared, but absmax is also stacked.
                    if optim_name == "ademamix8bit_blockwise" and name1 == "m1_m2":
                        s1 = torch.stack(
                            (
                                F.dequantize_blockwise(
                                    code=bnb_optimizer.state[p2][qmap],
                                    absmax=bnb_optimizer.state[p2][max_val][0],
                                    A=bnb_optimizer.state[p2][name2][0],
                                    blocksize=blocksize,
                                ),
                                F.dequantize_blockwise(
                                    code=bnb_optimizer.state[p2][qmap],
                                    absmax=bnb_optimizer.state[p2][max_val][1],
                                    A=bnb_optimizer.state[p2][name2][1],
                                    blocksize=blocksize,
                                ),
                            )
                        )
                    else:
                        s1 = F.dequantize_blockwise(
                            code=bnb_optimizer.state[p2][qmap],
                            absmax=bnb_optimizer.state[p2][max_val],
                            A=bnb_optimizer.state[p2][name2],
                            blocksize=blocksize,
                        )
                else:
                    s1 = F.dequantize(
                        code=bnb_optimizer.state[p2][qmap],
                        absmax=bnb_optimizer.state[p2][max_val],
                        A=bnb_optimizer.state[p2][name2],
                    )
                torch.testing.assert_close(s1cpy, s1)

                num_not_close = torch.isclose(torch_optimizer.state[p1][name1], s1, atol=atol, rtol=rtol) == 0
                assert num_not_close.sum().item() < 20

            # Lion can have pretty noisy updates where things lie at the boundary
            assert_most_approx_close(p1, p2.float(), patol, prtol, max_error_count=0)

        # the parameters diverge quickly. Here we keep them close
        # together so we can test against the Adam error
        p1.data = p1.data.to(gtype).float()
        p2.copy_(p1.data)
        torch.testing.assert_close(p1.to(gtype), p2)
        for (name1, name2, qmap, max_val), s in zip(str2statenames[optim_name], dequant_states):
            torch_optimizer.state[p1][name1].copy_(s.data)

    # print(sum(errors)/len(errors))
    # print(sum(relerrors)/len(relerrors))


@pytest.mark.parametrize("optim_bits", [32, 8], ids=id_formatter("optim_bits"))
@pytest.mark.parametrize("gtype", [torch.float32], ids=describe_dtype)
@pytest.mark.parametrize("dim2", [32, 1024, 4097], ids=id_formatter("dim2"))
@pytest.mark.parametrize("dim1", [1024], ids=id_formatter("dim1"))
def test_adam_percentile_clipping(dim1, dim2, gtype, optim_bits):
    if dim1 == 1 and dim2 == 1:
        return
    p1 = torch.randn(dim1, dim2, device="cpu", dtype=gtype) * 0.1
    beta1 = 0.9
    beta2 = 0.999
    lr = 0.001
    eps = 1e-8
    p1 = p1.cuda()
    p2 = p1.clone()
    adam1 = bnb.optim.Adam([p1], lr, (beta1, beta2), eps, optim_bits=optim_bits)
    adam2 = bnb.optim.Adam(
        [p2],
        lr,
        (beta1, beta2),
        eps,
        optim_bits=optim_bits,
        percentile_clipping=5,
    )

    gnorm_vec = torch.zeros(100).cuda()
    step = 0

    for i in range(50):
        step += 1
        g1 = torch.randn(dim1, dim2, device="cuda", dtype=gtype) * 0.1 + (0.01 * i)
        g2 = g1.clone()
        p2.grad = g2

        current_gnorm, clip_val, gnorm_scale = F.percentile_clipping(g1, gnorm_vec, step, 5)
        g1 = (g1.float() * gnorm_scale).to(gtype)
        p1.grad = g1

        adam1.step()
        adam2.step()

        # gnorm_scale is not deterministic (warp reductions), as such there can be slight differences in state
        if optim_bits == 32:
            torch.testing.assert_close(p1, p2)
            torch.testing.assert_close(
                adam1.state[p1]["state1"],
                adam2.state[p2]["state1"],
                atol=5e-5,
                rtol=1e-4,
            )
            torch.testing.assert_close(
                adam1.state[p1]["state2"],
                adam2.state[p2]["state2"],
                atol=5e-5,
                rtol=1e-4,
            )
        elif optim_bits == 8:
            torch.testing.assert_close(p1, p2, atol=1e-4, rtol=1e-3)
            torch.testing.assert_close(
                adam1.state[p1]["state1"],
                adam2.state[p2]["state1"],
                atol=2,
                rtol=1e-3,
            )
            torch.testing.assert_close(
                adam1.state[p1]["state2"],
                adam2.state[p2]["state2"],
                atol=2,
                rtol=1e-3,
            )
            adam1.state[p1]["state1"].copy_(adam2.state[p2]["state1"])
            adam1.state[p1]["state2"].copy_(adam2.state[p2]["state2"])
        if i % 10 == 0 and i > 0:
            path = get_temp_dir()
            torch.save(adam2.state_dict(), join(path, "opt.pt"))
            del adam2
            adam2 = None
            adam2 = bnb.optim.Adam(
                [p2],
                lr,
                (beta1, beta2),
                eps,
                optim_bits=optim_bits,
                percentile_clipping=5,
            )
            adam2.load_state_dict(torch.load(join(path, "opt.pt")))


optimizer_names_benchmark = [
    "adam8bit_blockwise",
    "paged_adam8bit_blockwise",
    "ademamix8bit_blockwise",
    "paged_ademamix8bit_blockwise",
    "ademamix8bit_blockwise_scheduled",
    "paged_ademamix8bit_blockwise_scheduled",
    "lion8bit_blockwise",
    "paged_lion8bit_blockwise",
    "paged_ademamix8bit_blockwise",
]


@pytest.mark.parametrize("dim1", [4096], ids=id_formatter("dim1"))
@pytest.mark.parametrize("dim2", [4096], ids=id_formatter("dim2"))
@pytest.mark.parametrize("gtype", [torch.float32, torch.bfloat16, torch.float16], ids=describe_dtype)
@pytest.mark.parametrize("optim_name", optimizer_names_benchmark, ids=id_formatter("opt"))
@pytest.mark.benchmark
def test_benchmark_blockwise(dim1, dim2, gtype, optim_name):
    if dim1 == 1 and dim2 == 1:
        return
    p1 = torch.randn(dim1, dim2, device="cuda", dtype=gtype) * 0.1

    bnb_optimizer = str2optimizers[optim_name][1]([p1])

    g = torch.randn(dim1, dim2, device="cuda", dtype=gtype) * 0.01
    p1.grad = g
    total_steps = 500
    for i in range(total_steps):
        if i == total_steps // 5:
            # 100 iterations for burn-in
            torch.cuda.synchronize()
            t0 = time.time()

        bnb_optimizer.step()

    torch.cuda.synchronize()
    s = time.time() - t0
    print("")
    params = (total_steps - total_steps // 5) * dim1 * dim2
    print(optim_name, gtype, s, params, s / params)
    # assert s < 3.9


@pytest.mark.parametrize("dim1", [2 * 1024], ids=id_formatter("dim1"))
@pytest.mark.parametrize("gtype", [torch.float16], ids=describe_dtype)
@pytest.mark.parametrize("optim_name", ["paged_adamw"], ids=id_formatter("optim_name"))
@pytest.mark.parametrize("mode", ["bnb"], ids=id_formatter("mode"))
@pytest.mark.benchmark
def test_stream_optimizer_bench(dim1, gtype, optim_name, mode):
    layers1 = torch.nn.Sequential(*torch.nn.ModuleList([torch.nn.Linear(dim1, dim1) for i in range(10)]))
    layers1 = layers1.to(gtype)
    layers1 = layers1.cuda()

    large_tensor = None
    if mode == "torch":
        optim = str2optimizers[optim_name][0](layers1.parameters())
    else:
        optim = str2optimizers[optim_name][1](layers1.parameters())
        # 12 GB
        large_tensor = torch.empty((int(4.5e9),), device="cuda")

    torch.cuda.synchronize()
    time.sleep(5)

    num_batches = 5
    batches = torch.randn(num_batches, 128, dim1, device="cuda").to(gtype)
    lbls = torch.randint(0, 10, size=(num_batches, 128)).cuda()

    for i in range(num_batches):
        print(i)
        b = batches[i]
        if i == 2:
            torch.cuda.synchronize()
            t0 = time.time()

        out1 = layers1(b)

        loss1 = torch.nn.functional.cross_entropy(out1, lbls[i]).mean()
        loss1.backward()
        optim.step()
    torch.cuda.synchronize()
    print(mode, time.time() - t0)
