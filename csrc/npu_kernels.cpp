#include "kernel_operator.h"
#include "npu_ops.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;

constexpr half Q_COFF_0 = -0.377685546875;
constexpr half Q_COFF_1 = -3.193359375;
constexpr half Q_COFF_2 = 0.583984375;
constexpr half Q_COFF_3 = 6.02734375;
constexpr half Q_COFF_4 = 1.9560546875;
constexpr half Q_COFF_5 = 7.08984375;

#define CEIL32(num) (((num) + 32 - 1) / 32 * 32)
#define CEIL_BASE(num, base) (((num) + (base) - 1) / (base) * (base))


template <uint32_t TypeMode>
class KernelQuantizeBlockwiseNf4 {
public:
    __aicore__ inline KernelQuantizeBlockwiseNf4() {}

    __aicore__ inline void Init(GM_ADDR A, GM_ADDR absmax, GM_ADDR out, GM_ADDR tilingDevice, TPipe &pipe)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        auto *tiling_data = reinterpret_cast<__gm__ BlockwiseNf4TilingData *>(tilingDevice);
        uint32_t coreNum = tiling_data->coreNum;
        this->blocksize = tiling_data->blocksize;
        uint32_t numel = tiling_data->numel;
        uint32_t singleCoreNumel = tiling_data->singleCoreNumel;
        uint32_t singleCoreNumelTail = tiling_data->singleCoreNumelTail;
        uint32_t ubSize = tiling_data->ubSize;
        uint32_t blockIdx = (uint32_t)GetBlockIdx();
        constexpr uint32_t ELEMENT_BYTES = (TypeMode == 1) ? 4 : 2;  // FP32: 4bytes, FP16/BF16: 2bytes
        if (coreNum - blockIdx == 1) {
            this->curCoreCaclNum = singleCoreNumelTail;
        } else {
            this->curCoreCaclNum = singleCoreNumel;
        }
        uint32_t eachBatchPkgNum = (ubSize - 16 * ELEMENT_BYTES) /
            (this->blocksize  * BUFFER_NUM * ELEMENT_BYTES + ELEMENT_BYTES * BUFFER_NUM +
             this->blocksize / 2 * BUFFER_NUM + ELEMENT_BYTES * this->blocksize * 5 + this->blocksize);
        if (eachBatchPkgNum >= 32 / ELEMENT_BYTES) {
            eachBatchPkgNum = (eachBatchPkgNum / (32 / ELEMENT_BYTES)) * (32 / ELEMENT_BYTES);
        } else {
            eachBatchPkgNum = (eachBatchPkgNum / 2) * 2;
        }
        this->eachBatchCaclNum = this->blocksize * eachBatchPkgNum;

        // gm, 32-byte alignment
        uint32_t AOffset = singleCoreNumel * blockIdx;
        uint32_t ABufferSize = singleCoreNumel;
        AGm.SetGlobalBuffer((__gm__ half*)A + AOffset, ABufferSize);

        uint32_t absmaxOffset = singleCoreNumel / this->blocksize * blockIdx;
        uint32_t absmaxBufferSize = singleCoreNumel / this->blocksize;
        absmaxGm.SetGlobalBuffer((__gm__ half*)absmax + absmaxOffset, absmaxBufferSize);

        uint32_t outOffset = singleCoreNumel / 2 * blockIdx;
        uint32_t outBufferSize = singleCoreNumel / 2;
        outGm.SetGlobalBuffer((__gm__ int8_t*)out + outOffset, outBufferSize);

        // TQue, 32-byte alignment
        pipe.InitBuffer(inQueueA, BUFFER_NUM, this->eachBatchCaclNum * ELEMENT_BYTES);
        pipe.InitBuffer(outQueueAbsmax, BUFFER_NUM, CEIL32(eachBatchPkgNum  * ELEMENT_BYTES));
        pipe.InitBuffer(outQueueOut, BUFFER_NUM, this->eachBatchCaclNum / 2);

        // TBuf, 32-byte alignment
        pipe.InitBuffer(calcAbs, this->eachBatchCaclNum * ELEMENT_BYTES);
        pipe.InitBuffer(calcInt8Tmp, this->eachBatchCaclNum * sizeof(int8_t));
        pipe.InitBuffer(calcAbsmaxBuf, this->eachBatchCaclNum * ELEMENT_BYTES);
        pipe.InitBuffer(calcAbsmaxTmp, this->eachBatchCaclNum * ELEMENT_BYTES);
        pipe.InitBuffer(calcNorm, this->eachBatchCaclNum * ELEMENT_BYTES);
        pipe.InitBuffer(calcQuant, this->eachBatchCaclNum * ELEMENT_BYTES);
    }

    __aicore__ inline void Process()
    {
        Compute();
    }

private:
    __aicore__ inline void Compute()
    {
        constexpr uint32_t ELEMENT_BYTES = (TypeMode == 1) ? 4 : 2;  // FP32: 4bytes, FP16/BF16: 2bytes
        LocalTensor<half> aLocal = inQueueA.AllocTensor<half>();
        LocalTensor<half> absmaxLocal = outQueueAbsmax.AllocTensor<half>();
        LocalTensor<int8_t> outLocal = outQueueOut.AllocTensor<int8_t>();

        LocalTensor<half> aAbs = calcAbs.Get<half>();
        LocalTensor<int8_t> int8Tmp = calcInt8Tmp.Get<int8_t>();
        LocalTensor<half> absmaxBuf = calcAbsmaxBuf.Get<half>();
        LocalTensor<half> absmaxTmp = calcAbsmaxTmp.Get<half>();
        LocalTensor<half> aNorm = calcNorm.Get<half>();
        LocalTensor<half> aQuant = calcQuant.Get<half>();

        // blockCount blockLen srcStride dstStride
        DataCopyParams dataCopyParams = {1, 0, 0, 0};
        uint32_t curBatchNumel = this->eachBatchCaclNum;
        uint32_t curBatchPkgNum = curBatchNumel / this->blocksize;

        uint32_t batchCount = (this->curCoreCaclNum + this->eachBatchCaclNum - 1) / this->eachBatchCaclNum;
        for (uint32_t batchIdx = 0; batchIdx < batchCount; batchIdx++) {
            if (batchCount - batchIdx == 1) {
                curBatchNumel = this->curCoreCaclNum - this->eachBatchCaclNum * batchIdx;
                curBatchPkgNum = (curBatchNumel + this->blocksize - 1) / this->blocksize;
            }

            dataCopyParams.blockLen = ELEMENT_BYTES * curBatchNumel;
            DataCopyPad(aLocal, AGm[this->eachBatchCaclNum * batchIdx], dataCopyParams, {true, 0, 0, 0});
            pipe_barrier(PIPE_ALL);

            // calc absmax
            Abs(aAbs, aLocal, curBatchNumel);
            pipe_barrier(PIPE_ALL);

            uint32_t mask = this->blocksize;
            uint32_t repeat = (curBatchNumel + mask) / mask;
            repeat = repeat > 255 ? 255 : repeat;
            WholeReduceMax<half>(absmaxLocal, aAbs, mask, repeat, 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
            pipe_barrier(PIPE_ALL);

            uint32_t dstShape[] = {curBatchPkgNum, this->blocksize};
            uint32_t srcShape[] = {curBatchPkgNum, 1};
            BroadCast<half, 2, 1>(absmaxBuf, absmaxLocal, dstShape, srcShape);
            pipe_barrier(PIPE_ALL);

            // div absmax
            Div(aNorm, aLocal, absmaxBuf, curBatchNumel);
            pipe_barrier(PIPE_ALL);

            // quant
            Adds(aQuant, aNorm, Q_COFF_0, curBatchNumel);
            pipe_barrier(PIPE_ALL);

            Mul(aQuant, aQuant, aNorm, curBatchNumel);
            pipe_barrier(PIPE_ALL);


            Adds(aQuant, aQuant, Q_COFF_1, curBatchNumel);
            pipe_barrier(PIPE_ALL);

            Mul(aQuant, aQuant, aNorm, curBatchNumel);
            pipe_barrier(PIPE_ALL);


            Adds(aQuant, aQuant, Q_COFF_2, curBatchNumel);
            pipe_barrier(PIPE_ALL);

            Mul(aQuant, aQuant, aNorm, curBatchNumel);
            pipe_barrier(PIPE_ALL);


            Adds(aQuant, aQuant, Q_COFF_3, curBatchNumel);
            pipe_barrier(PIPE_ALL);

            Mul(aQuant, aQuant, aNorm, curBatchNumel);
            pipe_barrier(PIPE_ALL);

            Muls(aQuant, aQuant, Q_COFF_4, curBatchNumel);
            pipe_barrier(PIPE_ALL);

            Adds(aQuant, aQuant, Q_COFF_5, curBatchNumel);
            pipe_barrier(PIPE_ALL);

            // round
            Round(aQuant, aQuant, curBatchNumel);
            pipe_barrier(PIPE_ALL);

            Adds(aQuant, aQuant, static_cast<half>(-8), curBatchNumel);
            pipe_barrier(PIPE_ALL);

            LocalTensor<int4b_t> outLocalTmp = int8Tmp.ReinterpretCast<int4b_t>();
            Cast(outLocalTmp, aQuant, RoundMode::CAST_NONE, curBatchNumel);
            pipe_barrier(PIPE_ALL);

            // Convert two adjacent int4 to one uint8
            LocalTensor<int8_t> tmp = outLocalTmp.ReinterpretCast<int8_t>();
            pipe_barrier(PIPE_ALL);

           // copy absmax to gm
            uint32_t gmOffset = this->eachBatchCaclNum / this->blocksize * batchIdx;
            dataCopyParams.blockLen = ELEMENT_BYTES * curBatchPkgNum; // Byte
            DataCopyPad(absmaxGm[gmOffset], absmaxLocal, dataCopyParams);

            // copy out to gm
            dataCopyParams.blockLen = curBatchNumel / 2; // Byte
            DataCopyPad(outGm[batchIdx * this->eachBatchCaclNum / 2], tmp, dataCopyParams);
            pipe_barrier(PIPE_ALL);
        }

        inQueueA.FreeTensor(aLocal);
        outQueueAbsmax.FreeTensor(absmaxLocal);
        outQueueOut.FreeTensor(outLocal);
    }

private:

    GlobalTensor<half> AGm;
    GlobalTensor<half> absmaxGm;
    GlobalTensor<int8_t> outGm;

    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueA;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueAbsmax;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueOut;

    uint32_t blocksize;
    uint32_t curCoreCaclNum;
    uint32_t eachBatchCaclNum;

    TBuf<TPosition::VECCALC> calcAbs;
    TBuf<TPosition::VECCALC> calcInt8Tmp;
    TBuf<TPosition::VECCALC> calcAbsmaxBuf;
    TBuf<TPosition::VECCALC> calcAbsmaxTmp;
    TBuf<TPosition::VECCALC> calcNorm;
    TBuf<TPosition::VECCALC> calcQuant;
};


template <typename T, uint32_t TypeMode>
class KernelDequantizeBlockwiseNf4 {
public:
    __aicore__ inline KernelDequantizeBlockwiseNf4() {}

    __aicore__ inline void Init(GM_ADDR A, GM_ADDR absmax, GM_ADDR out, GM_ADDR tilingDevice, TPipe &pipe)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        auto *tiling_data = reinterpret_cast<__gm__ BlockwiseNf4TilingData *>(tilingDevice);
        this->blocksize = tiling_data->blocksize;
        uint32_t coreNum = tiling_data->coreNum;
        uint32_t singleCoreNumel = tiling_data->singleCoreNumel;
        uint32_t singleCoreNumelTail = tiling_data->singleCoreNumelTail;
        uint32_t numel = tiling_data->numel;
        uint32_t ubSize = tiling_data->ubSize;
        uint32_t blockIdx = (uint32_t)GetBlockIdx();
        if (coreNum - blockIdx == 1) {
            this->CurCoreFP16Num = singleCoreNumelTail;
        } else {
            this->CurCoreFP16Num = singleCoreNumel;
        }
        constexpr uint32_t ELEMENT_BYTES = (TypeMode == 1) ? 4 : 2;  // FP32: 4bytes, FP16/BF16: 2bytes
        uint32_t eachBatchPkgNum = (ubSize - 16 * ELEMENT_BYTES) /
            (this->blocksize / 2 * BUFFER_NUM + ELEMENT_BYTES * BUFFER_NUM + this->blocksize *
            (ELEMENT_BYTES * BUFFER_NUM + sizeof(half) + sizeof(uint32_t) + ELEMENT_BYTES));
        if (eachBatchPkgNum >= 32 / ELEMENT_BYTES) {
            eachBatchPkgNum = (eachBatchPkgNum / (32 / ELEMENT_BYTES)) * (32 / ELEMENT_BYTES);
        } else {
            eachBatchPkgNum = (eachBatchPkgNum / 2) * 2;
        }
        this->eachBatchFP16Num = this->blocksize * eachBatchPkgNum; // 64 * 288

        // gm, 32-byte alignment
        uint32_t AOffset = singleCoreNumel / 2 * blockIdx;
        uint32_t ABufferSize = singleCoreNumel / 2;
        AGm.SetGlobalBuffer((__gm__ int8_t*)A + AOffset, ABufferSize);
        uint32_t absmaxOffset = singleCoreNumel / this->blocksize * blockIdx;
        uint32_t absmaxBufferSize = singleCoreNumel / this->blocksize;
        absmaxGm.SetGlobalBuffer((__gm__ T*)absmax + absmaxOffset, absmaxBufferSize);
        uint32_t outOffset = singleCoreNumel * blockIdx;
        uint32_t outBufferSize = singleCoreNumel;
        outGm.SetGlobalBuffer((__gm__ T*)out + outOffset, outBufferSize);

        // TQue, 32-byte alignment
        pipe.InitBuffer(inQueueA, BUFFER_NUM, this->eachBatchFP16Num / 2);
        pipe.InitBuffer(inQueueAbsmax, BUFFER_NUM, CEIL32(eachBatchPkgNum * ELEMENT_BYTES));
        pipe.InitBuffer(outQueueOut, BUFFER_NUM, this->eachBatchFP16Num * ELEMENT_BYTES);

        // TBuf, 32-byte alignment
        pipe.InitBuffer(calcNf4ToFloat, 16 * ELEMENT_BYTES);
        pipe.InitBuffer(calcAFP16, this->eachBatchFP16Num * sizeof(half));
        pipe.InitBuffer(calcAUint32, this->eachBatchFP16Num * sizeof(uint32_t));
        pipe.InitBuffer(calcAbsmaxBuf, this->eachBatchFP16Num * ELEMENT_BYTES);
    }

    __aicore__ inline void Process(void)
    {
        Compute();
    }

private:
    __aicore__ inline void initNf4ToFloat(LocalTensor<T> &nf4ToFloat)
    {
        if constexpr (TypeMode == 1) {
            nf4ToFloat(0) = static_cast<float32_t>(-1.0);
            nf4ToFloat(1) = static_cast<float32_t>(-0.6961928009986877);
            nf4ToFloat(2) = static_cast<float32_t>(-0.5250730514526367);
            nf4ToFloat(3) = static_cast<float32_t>(-0.39491748809814453);
            nf4ToFloat(4) = static_cast<float32_t>(-0.28444138169288635);
            nf4ToFloat(5) = static_cast<float32_t>(-0.18477343022823334);
            nf4ToFloat(6) = static_cast<float32_t>(-0.09105003625154495);
            nf4ToFloat(7) = static_cast<float32_t>(0.0);
            nf4ToFloat(8) = static_cast<float32_t>(0.07958029955625534);
            nf4ToFloat(9) = static_cast<float32_t>(0.16093020141124725);
            nf4ToFloat(10) = static_cast<float32_t>(0.24611230194568634);
            nf4ToFloat(11) = static_cast<float32_t>(0.33791524171829224);
            nf4ToFloat(12) = static_cast<float32_t>(0.44070982933044434);
            nf4ToFloat(13) = static_cast<float32_t>(0.5626170039176941);
            nf4ToFloat(14) = static_cast<float32_t>(0.7229568362236023);
            nf4ToFloat(15) = static_cast<float32_t>(1.0);
        } else if constexpr (TypeMode == 2) {
            nf4ToFloat(0) = static_cast<half>(-1.0);
            nf4ToFloat(1) = static_cast<half>(-0.6962890625);
            nf4ToFloat(2) = static_cast<half>(-0.52490234375);
            nf4ToFloat(3) = static_cast<half>(-0.39501953125);
            nf4ToFloat(4) = static_cast<half>(-0.284423828125);
            nf4ToFloat(5) = static_cast<half>(-0.184814453125);
            nf4ToFloat(6) = static_cast<half>(-0.091064453125);
            nf4ToFloat(7) = static_cast<half>(0.0);
            nf4ToFloat(8) = static_cast<half>(0.07958984375);
            nf4ToFloat(9) = static_cast<half>(0.160888671875);
            nf4ToFloat(10) = static_cast<half>(0.24609375);
            nf4ToFloat(11) = static_cast<half>(0.337890625);
            nf4ToFloat(12) = static_cast<half>(0.440673828125);
            nf4ToFloat(13) = static_cast<half>(0.5625);
            nf4ToFloat(14) = static_cast<half>(0.72314453125);
            nf4ToFloat(15) = static_cast<half>(1.0);
        }
    }

    __aicore__ inline void Compute(void)
    {
        constexpr uint32_t ELEMENT_BYTES = (TypeMode == 1) ? 4 : 2;  // FP32: 4bytes, FP16/BF16: 2bytes
        LocalTensor<int8_t> ALocal = inQueueA.AllocTensor<int8_t>();
        LocalTensor<T> absmaxLocal = inQueueAbsmax.AllocTensor<T>();
        LocalTensor<T> outLocal = outQueueOut.AllocTensor<T>();

        LocalTensor<half> AFP16 = calcAFP16.Get<half>();
        LocalTensor<int32_t> AInt32 = calcAUint32.Get<int32_t>();
        LocalTensor<T> absmaxBuf = calcAbsmaxBuf.Get<T>();
        LocalTensor<T> nf4ToFloat = calcNf4ToFloat.Get<T>();
        initNf4ToFloat(nf4ToFloat);

        DataCopyParams dataCopyParams = {1, 0, 0, 0};
        uint32_t curBatchNumel = this->eachBatchFP16Num;
        uint32_t curBatchPkgNum = curBatchNumel / this->blocksize;

        uint32_t batchCount = (this->CurCoreFP16Num + this->eachBatchFP16Num - 1) / this->eachBatchFP16Num;
        for (uint32_t batchIdx = 0; batchIdx < batchCount; batchIdx++) {
            if (batchCount - batchIdx == 1) {
                curBatchNumel = this->CurCoreFP16Num - this->eachBatchFP16Num * batchIdx;
                curBatchPkgNum = (curBatchNumel + this->blocksize - 1) / this->blocksize;
            }

            dataCopyParams.blockLen = curBatchNumel / 2;  // Byte
            DataCopyPad(ALocal, AGm[this->eachBatchFP16Num / 2 * batchIdx], dataCopyParams, {true, 0, 0, 0});
            dataCopyParams.blockLen = ELEMENT_BYTES * curBatchPkgNum;  // Byte
            uint32_t gmOffset = this->eachBatchFP16Num / this->blocksize * batchIdx;
            DataCopyPad(absmaxLocal, absmaxGm[gmOffset], dataCopyParams, {true, 0, 0, 0});
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            pipe_barrier(PIPE_ALL);

            LocalTensor<int4b_t> AInt4 = ALocal.ReinterpretCast<int4b_t>();
            Cast(AFP16, AInt4, RoundMode::CAST_NONE, curBatchNumel);
            pipe_barrier(PIPE_V);
            Adds(AFP16, AFP16, static_cast<half>(8), curBatchNumel);
            pipe_barrier(PIPE_V);
            if constexpr (TypeMode == 1) {
                Muls(AFP16, AFP16, static_cast<half>(4), curBatchNumel);
            } else {
                Muls(AFP16, AFP16, static_cast<half>(2), curBatchNumel);
            }
            pipe_barrier(PIPE_V);
            Cast(AInt32, AFP16, RoundMode::CAST_ROUND, curBatchNumel);
            pipe_barrier(PIPE_V);
            LocalTensor<uint32_t> AUint32 = AInt32.ReinterpretCast<uint32_t>();
            Gather<T>(outLocal, nf4ToFloat, AUint32, 0, curBatchNumel);
            pipe_barrier(PIPE_V);
            uint32_t dstShape[] = {curBatchPkgNum, this->blocksize};
            uint32_t srcShape[] = {curBatchPkgNum, 1};
            BroadCast<T, 2, 1>(absmaxBuf, absmaxLocal, dstShape, srcShape);
            pipe_barrier(PIPE_ALL);
            Mul(outLocal, outLocal, absmaxBuf, curBatchNumel);
            pipe_barrier(PIPE_ALL);

            dataCopyParams.blockLen = ELEMENT_BYTES * curBatchNumel;  // Byte
            DataCopyPad(outGm[batchIdx * this->eachBatchFP16Num], outLocal, dataCopyParams);
            pipe_barrier(PIPE_MTE3);
        }
        pipe_barrier(PIPE_ALL);

        inQueueA.FreeTensor(ALocal);
        inQueueAbsmax.FreeTensor(absmaxLocal);
        outQueueOut.FreeTensor(outLocal);
    }

private:
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueA;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueAbsmax;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueOut;
    TBuf<TPosition::VECCALC> calcAFP16;
    TBuf<TPosition::VECCALC> calcAUint32;
    TBuf<TPosition::VECCALC> calcNf4ToFloat;
    TBuf<TPosition::VECCALC> calcAbsmaxBuf;
    GlobalTensor<int8_t> AGm;
    GlobalTensor<T> absmaxGm;
    GlobalTensor<T> outGm;
    uint32_t blocksize;
    uint32_t CurCoreFP16Num;
    uint32_t eachBatchFP16Num;
};



extern "C" {

__global__ __aicore__ void quantize_blockwise_fp16_nf4(GM_ADDR A, GM_ADDR absmax, GM_ADDR out, GM_ADDR tiling)
{
    TPipe pipe;
    KernelQuantizeBlockwiseNf4<2> op;
    op.Init(A, absmax, out, tiling, pipe);
    op.Process();
}

__global__ __aicore__ void dequantize_blockwise_fp32_nf4(GM_ADDR A, GM_ADDR absmax, GM_ADDR out, GM_ADDR tiling)
{
    TPipe pipe;
    KernelDequantizeBlockwiseNf4<float32_t, 1> op;
    op.Init(A, absmax, out, tiling, pipe);
    op.Process();
}

__global__ __aicore__ void dequantize_blockwise_fp16_nf4(GM_ADDR A, GM_ADDR absmax, GM_ADDR out, GM_ADDR tiling)
{
    TPipe pipe;
    KernelDequantizeBlockwiseNf4<half, 2> op;
    op.Init(A, absmax, out, tiling, pipe);
    op.Process();
}

}
