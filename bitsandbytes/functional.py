# Copyright (c) Facebook, Inc. and its affiliates. 
#   
# This source code is licensed under the MIT license found in the 
# LICENSE file in the root directory of this source tree.
import os
import random
import math
import ctypes as ct
import torch
from torch import Tensor
from typing import Tuple

lib = ct.cdll.LoadLibrary(os.path.dirname(__file__) + '/libbitsandbytes.so')
name2qmap = {}

''' C FUNCTIONS FOR OPTIMIZERS '''

str2optimizer32bit = {}
str2optimizer32bit['adam'] = (lib.cadam32bit_g32, lib.cadam32bit_g16)
str2optimizer32bit['momentum'] = (lib.cmomentum32bit_g32, lib.cmomentum32bit_g16)
str2optimizer32bit['rmsprop'] = (lib.crmsprop32bit_g32, lib.crmsprop32bit_g16)
str2optimizer32bit['adagrad'] = (lib.cadagrad32bit_g32, lib.cadagrad32bit_g16)
str2optimizer32bit['lars'] = (lib.cmomentum32bit_g32, lib.cmomentum32bit_g16)
str2optimizer32bit['lamb'] = (lib.cadam32bit_g32, lib.cadam32bit_g16)

str2optimizer8bit = {}
str2optimizer8bit['adam'] = (lib.cadam_static_8bit_g32, lib.cadam_static_8bit_g16)
str2optimizer8bit['momentum'] = (lib.cmomentum_static_8bit_g32, lib.cmomentum_static_8bit_g16)
str2optimizer8bit['rmsprop'] = (lib.crmsprop_static_8bit_g32, lib.crmsprop_static_8bit_g16)
str2optimizer8bit['lamb'] = (lib.cadam_static_8bit_g32, lib.cadam_static_8bit_g16)
str2optimizer8bit['lars'] = (lib.cmomentum_static_8bit_g32, lib.cmomentum_static_8bit_g16)

str2optimizer8bit_blockwise = {}
str2optimizer8bit_blockwise['adam'] = (lib.cadam_8bit_blockwise_fp32, lib.cadam_8bit_blockwise_fp16)
str2optimizer8bit_blockwise['momentum'] = (lib.cmomentum_8bit_blockwise_fp32, lib.cmomentum_8bit_blockwise_fp16)
str2optimizer8bit_blockwise['rmsprop'] = (lib.crmsprop_8bit_blockwise_fp32, lib.crmsprop_8bit_blockwise_fp16)
str2optimizer8bit_blockwise['adagrad'] = (lib.cadagrad_8bit_blockwise_fp32, lib.cadagrad_8bit_blockwise_fp16)

optimal_normal = [-0.9939730167388916, -0.8727636337280273, -0.8097418546676636, -0.7660024166107178, -0.7318882346153259, -0.6793879270553589, -0.657649040222168, -0.6385974884033203, -0.6211113333702087, -0.5901028513908386, -0.5762918591499329, -0.5630806684494019, -0.5509274005889893, -0.5394591689109802, -0.5283197164535522, -0.517780065536499, -0.5074946284294128, -0.4980469048023224, -0.48867011070251465, -0.48003149032592773, -0.47125306725502014, -0.4629971981048584, -0.4547359049320221, -0.446626216173172, -0.43902668356895447, -0.43158355355262756, -0.4244747757911682, -0.4173796474933624, -0.41038978099823, -0.4055633544921875, -0.4035947024822235, -0.39701032638549805, -0.39057496190071106, -0.38439232110977173, -0.3782760500907898, -0.3721940815448761, -0.3661896586418152, -0.3604033589363098, -0.354605108499527, -0.34892538189888, -0.34320303797721863, -0.3376772701740265, -0.3323028087615967, -0.3269782066345215, -0.32166096568107605, -0.316457599401474, -0.3112771809101105, -0.3061025142669678, -0.30106794834136963, -0.2961243987083435, -0.2912728488445282, -0.28644347190856934, -0.28165507316589355, -0.2769731283187866, -0.2722635865211487, -0.26779335737228394, -0.26314786076545715, -0.2586647868156433, -0.2541804611682892, -0.2496625930070877, -0.24527113139629364, -0.24097171425819397, -0.23659978806972504, -0.23218469321727753, -0.22799566388130188, -0.22380566596984863, -0.21965542435646057, -0.2154538631439209, -0.2113603949546814, -0.20735277235507965, -0.20334717631340027, -0.19932441413402557, -0.19530178606510162, -0.19136647880077362, -0.18736697733402252, -0.18337111175060272, -0.17951400578022003, -0.1757056713104248, -0.17182783782482147, -0.1680615097284317, -0.16431649029254913, -0.16053077578544617, -0.15685945749282837, -0.15298527479171753, -0.1493264138698578, -0.14566898345947266, -0.14188314974308014, -0.13819937407970428, -0.1344561129808426, -0.1306886374950409, -0.1271020770072937, -0.12346585839986801, -0.11981867253780365, -0.11614970862865448, -0.11256207525730133, -0.10889036953449249, -0.10525048524141312, -0.1016591489315033, -0.09824034571647644, -0.09469068050384521, -0.0911419615149498, -0.08773849159479141, -0.08416644483804703, -0.08071305602788925, -0.07720902562141418, -0.07371306419372559, -0.07019119709730148, -0.06673648208379745, -0.06329209357500076, -0.059800852090120316, -0.0564190037548542, -0.05296570807695389, -0.049522045999765396, -0.04609023034572601, -0.04262964054942131, -0.039246633648872375, -0.03577171266078949, -0.03236335143446922, -0.028855687007308006, -0.02542758360505104, -0.022069433704018593, -0.018754752352833748, -0.015386369079351425, -0.01194947212934494, -0.008439815603196621, -0.004995611496269703, -0.0016682245768606663, 0.0, 0.0015510577941313386, 0.005062474869191647, 0.008417150937020779, 0.011741090565919876, 0.015184164978563786, 0.018582714721560478, 0.02204744517803192, 0.025471193715929985, 0.02889077737927437, 0.0323684960603714, 0.03579240292310715, 0.039281025528907776, 0.0427563451230526, 0.04619763046503067, 0.04968220740556717, 0.05326594039797783, 0.05679265409708023, 0.060245808213949203, 0.06372645497322083, 0.06721872836351395, 0.0706876739859581, 0.0742349922657013, 0.07774098962545395, 0.08123527467250824, 0.08468879014253616, 0.08810535818338394, 0.09155989438295364, 0.09498448669910431, 0.0985206812620163, 0.10206405073404312, 0.10563778132200241, 0.10921968519687653, 0.11284469068050385, 0.11653254181146622, 0.12008969485759735, 0.12368203699588776, 0.1272617131471634, 0.13089501857757568, 0.134552001953125, 0.1382799744606018, 0.14194637537002563, 0.14563234150409698, 0.14930322766304016, 0.15303383767604828, 0.1567956507205963, 0.16050070524215698, 0.16431072354316711, 0.16813558340072632, 0.17204202711582184, 0.1758781224489212, 0.17973239719867706, 0.1836014688014984, 0.18753431737422943, 0.19138391315937042, 0.19535475969314575, 0.19931404292583466, 0.20333819091320038, 0.20738255977630615, 0.21152682602405548, 0.21568812429904938, 0.21978361904621124, 0.22393859922885895, 0.22814159095287323, 0.23241068422794342, 0.23675410449504852, 0.24123944342136383, 0.24569889903068542, 0.2500703036785126, 0.25904011726379395, 0.26349544525146484, 0.2682226300239563, 0.272907555103302, 0.2774306833744049, 0.28220856189727783, 0.2869136929512024, 0.2916390895843506, 0.29649388790130615, 0.30142995715141296, 0.3065022826194763, 0.3114383816719055, 0.31648796796798706, 0.3216581642627716, 0.32700115442276, 0.3322487473487854, 0.33778008818626404, 0.3431521952152252, 0.3487405776977539, 0.3543166518211365, 0.3601346015930176, 0.36605337262153625, 0.37217751145362854, 0.378179669380188, 0.3843980133533478, 0.3906566798686981, 0.39714935421943665, 0.40357843041419983, 0.4104187488555908, 0.4171563684940338, 0.42418959736824036, 0.43136918544769287, 0.4389212429523468, 0.44673123955726624, 0.45457619428634644, 0.4627031683921814, 0.47130417823791504, 0.4798591434955597, 0.48897242546081543, 0.4979848861694336, 0.5, 0.5076631307601929, 0.5177803635597229, 0.5282770991325378, 0.5392990112304688, 0.5506287813186646, 0.5632893443107605, 0.5764452815055847, 0.5903191566467285, 0.6051878333091736, 0.6209936141967773, 0.6382884979248047, 0.6573970913887024, 0.6795773506164551, 0.7037051916122437, 0.7327037453651428, 0.7677436470985413, 0.8111193776130676, 0.875165581703186, 1.0]

optimal_half_normal = [0.0025565922260284424, 0.005811259150505066, 0.00961565226316452, 0.010822802782058716, 0.013123787939548492, 0.014242202043533325, 0.0143156498670578, 0.016469404101371765, 0.017666727304458618, 0.01773911714553833, 0.0199756920337677, 0.0210941880941391, 0.021161124110221863, 0.02451971173286438, 0.024580076336860657, 0.02685210108757019, 0.028012827038764954, 0.030198264867067337, 0.0302925705909729, 0.03136435151100159, 0.03374280035495758, 0.03487399220466614, 0.035243816673755646, 0.037192340940237045, 0.03822284936904907, 0.04164902865886688, 0.04173608124256134, 0.04401407018303871, 0.04508155584335327, 0.047482021152973175, 0.04756556823849678, 0.050963032990694046, 0.05196474492549896, 0.055417388677597046, 0.05793146416544914, 0.05799369141459465, 0.05887940526008606, 0.05895659327507019, 0.062420234084129333, 0.06493274495005608, 0.06499008461833, 0.06935599446296692, 0.07197384163737297, 0.07201516255736351, 0.07276943325996399, 0.07283210754394531, 0.07550075277686119, 0.07975354790687561, 0.07980883121490479, 0.08257630094885826, 0.0867777168750763, 0.08682405948638916, 0.08967285975813866, 0.09323835000395775, 0.09386616945266724, 0.09735457599163055, 0.09739077091217041, 0.10092401504516602, 0.10444298386573792, 0.10447832942008972, 0.10770941898226738, 0.10803905129432678, 0.11161200702190399, 0.1151546835899353, 0.11520349979400635, 0.11875157058238983, 0.11879390478134155, 0.1222602017223835, 0.122351735830307, 0.12240418791770935, 0.12594850733876228, 0.12597402930259705, 0.12602100148797035, 0.12960633635520935, 0.1296597123146057, 0.12966342642903328, 0.13227657973766327, 0.13325360417366028, 0.1333133578300476, 0.13691483438014984, 0.1371927298605442, 0.14066261053085327, 0.14088113978505135, 0.1447291411459446, 0.14805573225021362, 0.148526418954134, 0.15170684456825256, 0.15178103744983673, 0.15225710347294807, 0.1554398238658905, 0.15609459951519966, 0.15618794038891792, 0.1592724472284317, 0.1629735231399536, 0.16382690146565437, 0.16676269471645355, 0.16873238794505596, 0.17066434025764465, 0.17068277299404144, 0.1717144437134266, 0.17558929696679115, 0.17827065289020538, 0.17835864424705505, 0.18222273886203766, 0.18353315070271492, 0.18604370951652527, 0.18611834943294525, 0.1876586265861988, 0.18996606767177582, 0.19170701876282692, 0.19398853182792664, 0.19786442816257477, 0.19795633852481842, 0.20195159316062927, 0.2058800607919693, 0.2099103182554245, 0.2122517265379429, 0.21410366892814636, 0.21819619834423065, 0.22221362590789795, 0.22233009338378906, 0.22500130906701088, 0.2251257635653019, 0.22638091444969177, 0.23067741096019745, 0.23368822410702705, 0.2348879873752594, 0.2382080741226673, 0.2390350103378296, 0.2391497790813446, 0.24253453686833382, 0.24265171959996223, 0.2470107562839985, 0.24764248728752136, 0.24777774512767792, 0.2516774423420429, 0.256104726344347, 0.2564055472612381, 0.2607169933617115, 0.265461727976799, 0.26985861361026764, 0.2701106257736683, 0.2702729292213917, 0.274574413895607, 0.2750340588390827, 0.27919672429561615, 0.283704474568367, 0.28386808931827545, 0.28953738883137703, 0.2896753139793873, 0.29320384562015533, 0.29451676085591316, 0.295327290892601, 0.29802779853343964, 0.29818175733089447, 0.29972871020436287, 0.30290623009204865, 0.30305664241313934, 0.30486901476979256, 0.31299956142902374, 0.31518544629216194, 0.31790371239185333, 0.3205283172428608, 0.3230419009923935, 0.32595496252179146, 0.32612212374806404, 0.3282426446676254, 0.3283906430006027, 0.33146094158291817, 0.3316439874470234, 0.33365286886692047, 0.33723779395222664, 0.3390095978975296, 0.3427443392574787, 0.34853987768292427, 0.34869300201535225, 0.35457711294293404, 0.35537679493427277, 0.3604113645851612, 0.36124424636363983, 0.3665340431034565, 0.36667295172810555, 0.3727492541074753, 0.3729033060371876, 0.37888188660144806, 0.37907837703824043, 0.3792510814964771, 0.38557394221425056, 0.38573457673192024, 0.39108292758464813, 0.39911722019314766, 0.40589402988553047, 0.40604450181126595, 0.410498782992363, 0.4106704741716385, 0.4129834659397602, 0.4131447561085224, 0.4172855168581009, 0.4202354736626148, 0.4204071946442127, 0.43538858368992805, 0.4355536885559559, 0.4432900734245777, 0.44603554904460907, 0.4461968094110489, 0.451409537345171, 0.4598204083740711, 0.46002377942204475, 0.46178819239139557, 0.46868549659848213, 0.46995367109775543, 0.4868385046720505, 0.48702501133084297, 0.4958047419786453, 0.4960057884454727, 0.5051481872797012, 0.506847757846117, 0.5148334950208664, 0.5150565356016159, 0.5174009390175343, 0.5249751061201096, 0.5283288545906544, 0.5355450958013535, 0.539984006434679, 0.5467876642942429, 0.5522958822548389, 0.5584012717008591, 0.5706631988286972, 0.5836620181798935, 0.5836880058050156, 0.5942088551819324, 0.5975865572690964, 0.6102624125778675, 0.6124880760908127, 0.6286389082670212, 0.646102175116539, 0.6471664495766163, 0.665437325835228, 0.6687244363129139, 0.687017485499382, 0.6932839937508106, 0.7115348428487778, 0.7218200154602528, 0.7219699807465076, 0.7747527211904526, 0.7749756425619125, 0.8192005604505539, 0.8194110840559006, 0.8830635994672775, 0.9217727445065975, 0.9245667457580566, 0.947742685675621, 0.9674464613199234, 0.9890814647078514, 0.9891453236341476, 0.9925699159502983]

def create_linear_map(signed=True):
    if signed:
        return torch.linspace(-1.0, 1.0, 256)
    else:
        return torch.linspace(0.0, 1.0, 256)

def create_dynamic_map(signed=True, n=7):
    '''
    Creates the dynamic quantiztion map.

    The dynamic data type is made up of a dynamic exponent and
    fraction. As the exponent increase from 0 to -7 the number
    of bits available for the fraction shrinks.

    This is a generalization of the dynamic type where a certain
    number of the bits and be reserved for the linear quantization
    region (the fraction). n determines the maximum number of
    exponent bits.

    For more details see
    (8-Bit Approximations for Parallelism in Deep Learning)[https://arxiv.org/abs/1511.04561]
    '''

    data = []
    # these are additional items that come from the case
    # where all the exponent bits are zero and no
    # indicator bit is present
    additional_items = 2**(7-n)-1
    if not signed: additional_items = 2*additional_items
    for i in range(n):
        fraction_items = 2**(i+7-n)+1 if signed else 2**(i+7-n+1)+1
        boundaries = torch.linspace(0.1, 1, fraction_items)
        means = (boundaries[:-1]+boundaries[1:])/2.0
        data += ((10**(-(n-1)+i))*means).tolist()
        if signed:
            data += (-(10**(-(n-1)+i))*means).tolist()

    if additional_items > 0:
        boundaries = torch.linspace(0.1, 1, additional_items+1)
        means = (boundaries[:-1]+boundaries[1:])/2.0
        data += ((10**(-(n-1)+i))*means).tolist()
        if signed:
            data += (-(10**(-(n-1)+i))*means).tolist()

    data.append(0)
    data.append(1.0)
    data.sort()
    return Tensor(data)

def get_ptr(A: Tensor) -> ct.c_void_p:
    '''
    Get the ctypes pointer from a PyTorch Tensor.

    Parameters
    ----------
    A : torch.tensor
        The PyTorch tensor.

    Returns
    -------
    ctypes.c_void_p
    '''
    if A is None: return None
    else: return ct.c_void_p(A.data.storage().data_ptr())

def estimate_quantiles(A: Tensor, out: Tensor=None, offset: float=1/512) -> Tensor:
    '''
    Estimates 256 equidistant quantiles on the input tensor eCDF.

    Uses SRAM-Quantiles algorithm to quickly estimate 256 equidistant quantiles
    via the eCDF of the input tensor `A`. This is a fast but approximate algorithm
    and the extreme quantiles close to 0 and 1 have high variance / large estimation
    errors. These large errors can be avoided by using the offset variable which trims
    the distribution. The default offset value of 1/512 ensures minimum entropy encoding -- it
    trims 1/512 = 0.2% from each side of the distrivution. An offset value of 0.01 to 0.02
    usually has a much lower error but is not a minimum entropy encoding. Given an offset
    of 0.02 equidistance points in the range [0.02, 0.98] are used for the quantiles.

    Parameters
    ----------
    A : torch.Tensor
        The input tensor. Any shape.
    out : torch.Tensor
        Tensor with the 256 estimated quantiles.
    offset : float
        The offset for the first and last quantile from 0 and 1. Default: 1/512

    Returns
    -------
    torch.Tensor:
        The 256 quantiles in float32 datatype.
    '''
    if out is None: out = torch.zeros((256,), dtype=torch.float32, device=A.device)
    if A.dtype == torch.float32:
        lib.cestimate_quantiles_fp32(get_ptr(A), get_ptr(out), ct.c_float(offset), ct.c_int(A.numel()))
    elif A.dtype == torch.float16:
        lib.cestimate_quantiles_fp16(get_ptr(A), get_ptr(out), ct.c_float(offset), ct.c_int(A.numel()))
    else:
        raise NotImplementError(f'Not supported data type {A.dtype}')
    return out

def quantize_blockwise(A: Tensor, code: Tensor=None, absmax: Tensor=None, rand=None, out: Tensor=None) -> Tensor:
    '''
    Quantize tensor A in blocks of size 4096 values.

    Quantizes tensor A by dividing it into blocks of 4096 values.
    Then the absolute maximum value within these blocks is calculated
    for the non-linear quantization.

    Parameters
    ----------
    A : torch.Tensor
        The input tensor.
    code : torch.Tensor
        The quantization map.
    absmax : torch.Tensor
        The absmax values.
    rand : torch.Tensor
        The tensor for stochastic rounding.
    out : torch.Tensor
        The output tensor (8-bit).

    Returns
    -------
    torch.Tensor:
        The 8-bit tensor.
    tuple(torch.Tensor, torch.Tensor):
        The quantization state to undo the quantization.
    '''

    if code is None:
        if 'dynamic' not in name2qmap: name2qmap['dynamic'] = create_dynamic_map().to(A.device)
        code = name2qmap['dynamic']
        code = code.to(A.device)

    if absmax is None:
        n = A.numel()
        num_blocks = 4096
        blocks = n//num_blocks
        blocks += 1 if n % num_blocks > 0 else 0
        absmax = torch.zeros((blocks,), device=A.device)

    if out is None: out = torch.zeros_like(A, dtype=torch.uint8)


    if A.device.type != 'cpu':
        if rand is not None:
            assert rand.numel() >= 1024
            rand_offset = random.randint(0, 1023)
            if A.dtype == torch.float32:
                lib.cquantize_blockwise_stochastic_fp32(get_ptr(code), get_ptr(A), get_ptr(absmax), get_ptr(out), get_ptr(rand), ct.c_int32(rand_offset), ct.c_int(A.numel()))
            elif A.dtype == torch.float16:
                lib.cquantize_blockwise_stochastic_fp16(get_ptr(code), get_ptr(A), get_ptr(absmax), get_ptr(out), get_ptr(rand), ct.c_int32(rand_offset), ct.c_int(A.numel()))
            else:
                raise ValueError(f'Blockwise quantization only supports 16/32-bit floats, but got {A.dtype}')
        else:
            if A.dtype == torch.float32:
                lib.cquantize_blockwise_fp32(get_ptr(code), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int(A.numel()))
            elif A.dtype == torch.float16:
                lib.cquantize_blockwise_fp16(get_ptr(code), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int(A.numel()))
            else:
                raise ValueError(f'Blockwise quantization only supports 16/32-bit floats, but got {A.dtype}')
    else:
        # cpu
        assert rand is None
        lib.cquantize_blockwise_cpu_fp32(get_ptr(code), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int(A.numel()))

    return out, (absmax, code)

def dequantize_blockwise(A: Tensor, quant_state: Tuple[Tensor, Tensor]=None,
                         absmax: Tensor=None, code: Tensor=None, out: Tensor=None,
                         blocksize: int=4096) -> Tensor:
    '''
    Dequantizes blockwise quantized values.

    Dequantizes the tensor A with maximum absolute values absmax in
    blocks of size 4096.

    Parameters
    ----------
    A : torch.Tensor
        The input 8-bit tensor.
    quant_state : tuple(torch.Tensor, torch.Tensor)
        Tuple of code and absmax values. 
    absmax : torch.Tensor
        The absmax values.
    code : torch.Tensor
        The quantization map.
    out : torch.Tensor
        Dequantized output tensor (default: float32)


    Returns
    -------
    torch.Tensor:
        Dequantized tensor (default: float32)
    '''
    assert quant_state is not None or absmax is not None
    if code is None and quant_state is None:
        if 'dynamic' not in name2qmap: name2qmap['dynamic'] = create_dynamic_map().to(A.device)
        code = name2qmap['dynamic']
        code = code.to(A.device)

    if out is None: out = torch.zeros_like(A, dtype=torch.float32)
    if quant_state is None: quant_state = (absmax, code)

    if blocksize not in [2048, 4096]:
        raise ValueError(f'The blockwise of {blocksize} is not supported. Supported values: [2048 4096]')

    if A.device.type != 'cpu':
        if out.dtype == torch.float32:
            lib.cdequantize_blockwise_fp32(get_ptr(quant_state[1]), get_ptr(A), get_ptr(quant_state[0]), get_ptr(out), ct.c_int(blocksize), ct.c_int(A.numel()))
        elif out.dtype == torch.float16:
            lib.cdequantize_blockwise_fp16(get_ptr(quant_state[1]), get_ptr(A), get_ptr(quant_state[0]), get_ptr(out), ct.c_int(blocksize), ct.c_int(A.numel()))
        else:
            raise ValueError(f'Blockwise quantization only supports 16/32-bit floats, but got {A.dtype}')
    else:
        lib.cdequantize_blockwise_cpu_fp32(get_ptr(quant_state[1]), get_ptr(A), get_ptr(quant_state[0]), get_ptr(out), ct.c_int(A.numel()))


    return out


def quantize(A: Tensor, code: Tensor=None, out: Tensor=None) -> Tensor:
    if code is None:
        if 'dynamic' not in name2qmap: name2qmap['dynamic'] = create_dynamic_map().to(A.device)
        code = name2qmap['dynamic']
        code = code.to(A.device)

    absmax = torch.abs(A).max()
    inp = A/absmax
    out = quantize_no_absmax(inp, code, out)
    return out, (absmax, code)

def dequantize(A: Tensor, quant_state: Tuple[Tensor, Tensor]=None, absmax: Tensor=None, code: Tensor=None, out: Tensor=None) -> Tensor:
    assert quant_state is not None or absmax is not None
    if code is None and quant_state is None:
        if 'dynamic' not in name2qmap: name2qmap['dynamic'] = create_dynamic_map().to(A.device)
        code = name2qmap['dynamic']
        code = code.to(A.device)

    if quant_state is None: quant_state = (absmax, code)
    out = dequantize_no_absmax(A, quant_state[1], out)
    return out*quant_state[0]

def quantize_no_absmax(A: Tensor, code: Tensor, out: Tensor=None) -> Tensor:
    '''
    Quantizes input tensor to 8-bit.

    Quantizes the 32-bit input tensor `A` to the 8-bit output tensor
    `out` using the quantization map `code`.

    Parameters
    ----------
    A : torch.Tensor
        The input tensor.
    code : torch.Tensor
        The quantization map.
    out : torch.Tensor, optional
        The output tensor. Needs to be of type byte.

    Returns
    -------
    torch.Tensor:
        Quantized 8-bit tensor.
    '''
    if out is None: out = torch.zeros_like(A, dtype=torch.uint8)
    lib.cquantize(get_ptr(code), get_ptr(A), get_ptr(out), ct.c_int(A.numel()))
    return out

def dequantize_no_absmax(A: Tensor, code: Tensor, out: Tensor=None) -> Tensor:
    '''
    Dequantizes the 8-bit tensor to 32-bit.

    Dequantizes the 8-bit tensor `A` to the 32-bit tensor `out` via
    the quantization map `code`.

    Parameters
    ----------
    A : torch.Tensor
        The 8-bit input tensor.
    code : torch.Tensor
        The quantization map.
    out : torch.Tensor
        The 32-bit output tensor.

    Returns
    -------
    torch.Tensor:
        32-bit output tensor.
    '''
    if out is None: out = torch.zeros_like(A, dtype=torch.float32)
    lib.cdequantize(get_ptr(code), get_ptr(A), get_ptr(out), ct.c_int(A.numel()))
    return out

def optimizer_update_32bit(optimizer_name:str, g: Tensor, p: Tensor, state1: Tensor,
                beta1: float, eps: float, step: int, lr: float,
                state2: Tensor=None, beta2: float=0.0,
                weight_decay: float=0.0, gnorm_scale: float=1.0,
                unorm_vec: Tensor=None, max_unorm: float=0.0, skip_zeros=False) -> None:
    '''
    Performs an inplace optimizer update with one or two optimizer states.

    Universal optimizer update for 32-bit state and 32/16-bit gradients/weights.

    Parameters
    ----------
    optimizer_name : str
        The name of the optimizer: {adam}.
    g : torch.Tensor
        Gradient tensor.
    p : torch.Tensor
        Parameter tensor.
    state1 : torch.Tensor
        Optimizer state 1.
    beta1 : float
        Optimizer beta1.
    eps : float
        Optimizer epsilon.
    weight_decay : float
        Weight decay.
    step : int
        Current optimizer step.
    lr : float
        The learning rate.
    state2 : torch.Tensor
        Optimizer state 2.
    beta2 : float
        Optimizer beta2.
    gnorm_scale : float
        The factor to rescale the gradient to the max clip value.
    unorm_vec : torch.Tensor
        The tensor for the update norm.
    max_unorm : float
        The maximum update norm relative to the weight norm.
    skip_zeros : bool
        Whether to skip zero-valued gradients or not (default: False).
    '''

    param_norm = 0.0
    if max_unorm > 0.0:
        param_norm = torch.norm(p.data.float())

    if optimizer_name not in str2optimizer32bit:
        raise NotImplementError(f'Optimizer not implemented: {optimizer_name}. Choices: {",".join(str2optimizer32bit.keys())}')

    if g.dtype == torch.float32 and state1.dtype == torch.float32:
        str2optimizer32bit[optimizer_name][0](get_ptr(g), get_ptr(p), get_ptr(state1), get_ptr(state2), get_ptr(unorm_vec), ct.c_float(max_unorm),
                    ct.c_float(param_norm), ct.c_float(beta1), ct.c_float(beta2), ct.c_float(eps), ct.c_float(weight_decay),
                    ct.c_int32(step), ct.c_float(lr), ct.c_float(gnorm_scale), ct.c_bool(skip_zeros), ct.c_int32(g.numel()))
    elif g.dtype == torch.float16 and state1.dtype == torch.float32:
        str2optimizer32bit[optimizer_name][1](get_ptr(g), get_ptr(p), get_ptr(state1), get_ptr(state2), get_ptr(unorm_vec), ct.c_float(max_unorm),
                    ct.c_float(param_norm), ct.c_float(beta1), ct.c_float(beta2), ct.c_float(eps), ct.c_float(weight_decay),
                    ct.c_int32(step), ct.c_float(lr), ct.c_float(gnorm_scale), ct.c_bool(skip_zeros), ct.c_int32(g.numel()))
    else:
        raise ValueError(f'Gradient+optimizer bit data type combination not supported: grad {g.dtype}, optimizer {state1.dtype}')

def optimizer_update_8bit(optimizer_name: str, g: Tensor, p: Tensor, state1: Tensor, state2: Tensor,
                beta1: float, beta2: float, eps: float,
                step: int, lr: float, qmap1: Tensor, qmap2: Tensor,
                max1: Tensor, max2: Tensor, new_max1: Tensor, new_max2: Tensor,
                weight_decay: float=0.0, gnorm_scale: float=1.0,
                unorm_vec: Tensor=None, max_unorm: float=0.0) -> None:
    '''
    Performs an inplace Adam update.

    Universal Adam update for 32/8-bit state and 32/16-bit gradients/weights.
    Uses AdamW formulation if weight decay > 0.0.

    Parameters
    ----------
    optimizer_name : str
        The name of the optimizer. Choices {adam, momentum}
    g : torch.Tensor
        Gradient tensor.
    p : torch.Tensor
        Parameter tensor.
    state1 : torch.Tensor
        Adam state 1.
    state2 : torch.Tensor
        Adam state 2.
    beta1 : float
        Adam beta1.
    beta2 : float
        Adam beta2.
    eps : float
        Adam epsilon.
    weight_decay : float
        Weight decay.
    step : int
        Current optimizer step.
    lr : float
        The learning rate.
    qmap1 : torch.Tensor
        Quantization map for first Adam state.
    qmap2 : torch.Tensor
        Quantization map for second Adam state.
    max1 : torch.Tensor
        Max value for first Adam state update.
    max2 : torch.Tensor
        Max value for second Adam state update.
    new_max1 : torch.Tensor
        Max value for the next Adam update of the first state.
    new_max2 : torch.Tensor
        Max value for the next Adam update of the second state.
    gnorm_scale : float
        The factor to rescale the gradient to the max clip value.
    unorm_vec : torch.Tensor
        The tensor for the update norm.
    max_unorm : float
        The maximum update norm relative to the weight norm.
    '''

    param_norm = 0.0
    if max_unorm > 0.0:
        param_norm = torch.norm(p.data.float())

    if g.dtype == torch.float32 and state1.dtype == torch.uint8:
        str2optimizer8bit[optimizer_name][0](get_ptr(p), get_ptr(g), get_ptr(state1), get_ptr(state2),
                    get_ptr(unorm_vec), ct.c_float(max_unorm), ct.c_float(param_norm),
                    ct.c_float(beta1), ct.c_float(beta2), ct.c_float(eps),
                    ct.c_int32(step), ct.c_float(lr),
                    get_ptr(qmap1), get_ptr(qmap2),
                    get_ptr(max1), get_ptr(max2), get_ptr(new_max1), get_ptr(new_max2),
                    ct.c_float(weight_decay),ct.c_float(gnorm_scale), ct.c_int32(g.numel()))
    elif g.dtype == torch.float16 and state1.dtype == torch.uint8:
        str2optimizer8bit[optimizer_name][1](get_ptr(p), get_ptr(g), get_ptr(state1), get_ptr(state2),
                    get_ptr(unorm_vec), ct.c_float(max_unorm), ct.c_float(param_norm),
                    ct.c_float(beta1), ct.c_float(beta2), ct.c_float(eps),
                    ct.c_int32(step), ct.c_float(lr),
                    get_ptr(qmap1), get_ptr(qmap2),
                    get_ptr(max1), get_ptr(max2), get_ptr(new_max1), get_ptr(new_max2),
                    ct.c_float(weight_decay),ct.c_float(gnorm_scale), ct.c_int32(g.numel()))
    else:
        raise ValueError(f'Gradient+optimizer bit data type combination not supported: grad {g.dtype}, optimizer {state1.dtype}')


def optimizer_update_8bit_blockwise(optimizer_name: str, g: Tensor, p: Tensor, state1: Tensor, state2: Tensor,
                beta1: float, beta2: float, eps: float,
                step: int, lr: float, qmap1: Tensor, qmap2: Tensor,
                absmax1: Tensor, absmax2: Tensor, weight_decay: float=0.0, gnorm_scale: float=1.0,
                skip_zeros=False) -> None:


    if g.dtype == torch.float32 and state1.dtype == torch.uint8:
        str2optimizer8bit_blockwise[optimizer_name][0](get_ptr(p), get_ptr(g), get_ptr(state1), get_ptr(state2),
                    ct.c_float(beta1), ct.c_float(beta2), ct.c_float(eps),
                    ct.c_int32(step), ct.c_float(lr), get_ptr(qmap1), get_ptr(qmap2),
                    get_ptr(absmax1), get_ptr(absmax2), ct.c_float(weight_decay), ct.c_float(gnorm_scale),
                    ct.c_bool(skip_zeros), ct.c_int32(g.numel()))
    elif g.dtype == torch.float16 and state1.dtype == torch.uint8:
        str2optimizer8bit_blockwise[optimizer_name][1](get_ptr(p), get_ptr(g), get_ptr(state1), get_ptr(state2),
                    ct.c_float(beta1), ct.c_float(beta2), ct.c_float(eps),
                    ct.c_int32(step), ct.c_float(lr), get_ptr(qmap1), get_ptr(qmap2),
                    get_ptr(absmax1), get_ptr(absmax2), ct.c_float(weight_decay), ct.c_float(gnorm_scale),
                    ct.c_bool(skip_zeros), ct.c_int32(g.numel()))
    else:
        raise ValueError(f'Gradient+optimizer bit data type combination not supported: grad {g.dtype}, optimizer {state1.dtype}')


def percentile_clipping(grad: Tensor, gnorm_vec: Tensor, step: int, percentile: int=5):
    """Applies percentile clipping

    grad: torch.Tensor
        The gradient tensor.
    gnorm_vec: torch.Tensor
        Vector of gradient norms. 100 elements expected.
    step: int
        The current optimiation steps (number of past gradient norms).

    """
    if grad.dtype == torch.float32:
        lib.cpercentile_clipping_g32(get_ptr(grad), get_ptr(gnorm_vec), ct.c_int32(step), ct.c_int32(grad.numel()))
    elif grad.dtype == torch.float16:
        lib.cpercentile_clipping_g16(get_ptr(grad), get_ptr(gnorm_vec), ct.c_int32(step), ct.c_int32(grad.numel()))
    else:
        raise ValueError(f'Gradient type {grad.dtype} not supported!')

    current_gnorm = torch.sqrt(gnorm_vec[step % 100])
    vals, idx = torch.sort(gnorm_vec)
    clip_value = torch.sqrt(vals[percentile])
    gnorm_scale = 1.0

    if current_gnorm > clip_value:
        gnorm_scale = clip_value/current_gnorm

    return current_gnorm, clip_value, gnorm_scale


def histogram_scatter_add_2d(histogram: Tensor, index1: Tensor, index2: Tensor, source: Tensor):
    assert len(histogram.shape) == 2
    assert histogram.dtype == torch.float32
    assert source.dtype == torch.float32
    assert index1.dtype == torch.int32
    assert index2.dtype == torch.int32

    assert histogram.device.type == 'cuda'
    assert index1.device.type == 'cuda'
    assert index2.device.type == 'cuda'
    assert source.device.type == 'cuda'

    maxdim1 = ct.c_int32(histogram.shape[0])
    n = ct.c_int32(index1.numel())
    lib.chistogram_scatter_add_2d(get_ptr(histogram), get_ptr(index1), get_ptr(index2), get_ptr(source), maxdim1, n)
