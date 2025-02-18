#include "kernel_operator.h"
#include "npu_ops.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;


#define CEIL32(num) (((num) + 32 - 1) / 32 * 32)
#define CEIL_BASE(num, base) (((num) + (base) - 1) / (base) * (base))


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


namespace row_col_quant_kernel {
    constexpr uint32_t DEFAULT_MIN_BLOCK_SIZE = 32;

    struct CurrentTileOffset {
        uint32_t rowIndex = 0;  // Uint: element
        uint32_t colIndex = 0;  // Uint: element
    };

    struct CopyParam {
        uint16_t blockCount;
        uint16_t blockLen;
        uint16_t blockLen_int8;
        uint16_t stride;
        uint16_t stride_int8;
    };

    // tiling for RowColQuant Vector on one VectorCore
    struct RowColQuantTilingKernel {
        uint32_t coreIdx = 0;        // vector core idx
        uint32_t is32BAligned = 1;
        uint32_t usedCoreNum = 1;
        uint64_t totalBlockLen = 0;
        uint32_t inputDataType = 1;
        uint64_t colLen = 0;
        uint64_t rowLen = 0;

        uint32_t baseRowLen = 0;  // for one tile in one core, Unit:element
        uint32_t baseColLen = 0;  // for one tile in one core, Unit:element
        uint32_t tailRowLen = 0;  // number of tail row in one core, Unit:element
        uint32_t tailColLen = 0;  // number of column in one core, Unit:element

        uint32_t rowAlignedLen = 0;

        uint32_t tileLength = 0;  // baseRowLen * baseColLen

        uint64_t rowTileNum = 0;
        uint64_t colTileNum = 0;
        uint64_t totalTileNum = 0;

        uint64_t baseRowTileNum = 0;
        uint64_t baseColTileNum = 0;

        uint64_t baseRowBaseColCalLen = 0;
        uint64_t baseRowTailColCalLen = 0;
        uint64_t tailRowBaseColCalLen = 0;
        uint64_t tailRowTailColCalLen = 0;
        CopyParam baseRowBaseColCopyParam;
        CopyParam baseRowTailColCopyParam;
        CopyParam tailRowBaseColCopyParam;
        CopyParam tailRowTailColCopyParam;

        float threshold = 0.0f;
        uint32_t outliersNum = 0;
        uint32_t isColQuant = 0;
        uint32_t isOutlierIndex = 0;

        uint32_t curCalLen = 0;     // curCalRowLen * curCalColLen
        uint64_t curCalAlignedRowLen = 0;
        uint32_t curCalRowLen = 0;  // row length of current tile
        uint32_t curCalColLen = 0;  // aligned col length of current tile. ALIGNUP(curColLen, alignedLen) Uint: element
        uint32_t curColLen = 0;     // col length of current tile. Uint: element
        uint64_t gmOffset = 0;
        float curCalRowLen_float = 0.0;
        CopyParam *curTileCopyParam = nullptr;
        CurrentTileOffset curTileOffset;

        uint32_t usedCoreNumForOutlier = 0;
        uint32_t baseCoreNumForOutlier = 0;
        OutlierTilingParam baseCoreParam;
        OutlierTilingParam tailCoreParam;
        OutlierTilingParam curCoreParam;
        uint32_t copyInOffset;

        // calc tiling data
        __aicore__ void GetTilingAndOffset(GM_ADDR tilingGm_, uint32_t inputDTypeLen_)
        {
            auto tempTilingGm = (__gm__ RowColQuantTilingData *)tilingGm_;
            inputDataType = 2;

            // input scalar parameters
            outliersNum = tempTilingGm->outliersNum;
            usedCoreNum = tempTilingGm->usedCoreNum;
            is32BAligned = tempTilingGm->is32BAligned;
            rowLen = tempTilingGm->rowLen;
            colLen = tempTilingGm->colLen;
            totalBlockLen = rowLen * colLen;

            baseRowLen = tempTilingGm->baseRowLen;
            baseColLen = tempTilingGm->baseColLen;

            // input scalar parameters
            threshold = tempTilingGm->threshold;
            outliersNum = tempTilingGm->outliersNum;
            isColQuant = tempTilingGm->isColQuant;
            isOutlierIndex = tempTilingGm->isOutlierIndex;

            usedCoreNumForOutlier = tempTilingGm->usedCoreNumForOutlier;
            baseCoreNumForOutlier = tempTilingGm->baseCoreNumForOutlier;

            baseCoreParam.colLen = tempTilingGm->baseCoreParam.colLen;
            baseCoreParam.loopNum = tempTilingGm->baseCoreParam.loopNum;
            baseCoreParam.tileCol = tempTilingGm->baseCoreParam.tileCol;
            baseCoreParam.isTailExist = tempTilingGm->baseCoreParam.isTailExist;
            baseCoreParam.tailCol = tempTilingGm->baseCoreParam.tailCol;

            tailCoreParam.colLen = tempTilingGm->tailCoreParam.colLen;
            tailCoreParam.loopNum = tempTilingGm->tailCoreParam.loopNum;
            tailCoreParam.tileCol = tempTilingGm->tailCoreParam.tileCol;
            tailCoreParam.isTailExist = tempTilingGm->tailCoreParam.isTailExist;
            tailCoreParam.tailCol = tempTilingGm->tailCoreParam.tailCol;

            auto alignedLen = DEFAULT_MIN_BLOCK_SIZE / 2;
            tileLength = (is32BAligned == 1) ? (baseRowLen * baseColLen) : baseRowLen * ALIGNUP(baseColLen, alignedLen);

            rowAlignedLen = ALIGNUP(baseRowLen, alignedLen);

            baseRowTileNum = rowLen / baseRowLen;
            baseColTileNum = colLen / baseColLen;
            tailRowLen = rowLen % baseRowLen;
            tailColLen = colLen % baseColLen;
            rowTileNum = (tailRowLen > 0) ? (baseRowTileNum + 1) : baseRowTileNum;
            colTileNum = (tailColLen > 0) ? (baseColTileNum + 1) : baseColTileNum;
            totalTileNum = rowTileNum * colTileNum;

            coreIdx = AscendC::GetBlockIdx();
            if (coreIdx < usedCoreNum) {
                CalcTileCopyParams(inputDataType);
            }
            if (coreIdx < usedCoreNumForOutlier) {
                CalcOutlierParam();
            }
        }

        __aicore__ inline void CalcOutlierParam()
        {
            if (coreIdx < baseCoreNumForOutlier) {
                curCoreParam = baseCoreParam;
                copyInOffset = coreIdx * baseCoreParam.colLen;
            } else {
                curCoreParam = tailCoreParam;
                copyInOffset = colLen - tailCoreParam.colLen;
            }
        }

        __aicore__ inline void CalcOneTileCopyParam(
                uint64_t calRowLen, uint64_t calColLen, uint32_t inputDTypeLen, CopyParam &copyParam)
        {
            uint16_t blockUnit = (is32BAligned == 1) ? DEFAULT_MIN_BLOCK_SIZE : 1;
            copyParam.blockCount = calRowLen;
            copyParam.blockLen = calColLen * inputDTypeLen / blockUnit;
            copyParam.blockLen_int8 = calColLen / blockUnit;
            copyParam.stride = (calRowLen == 1) ? 0 : ((colLen - calColLen) * inputDTypeLen / blockUnit);
            copyParam.stride_int8 = (calRowLen == 1) ? 0 : ((colLen - calColLen) / blockUnit);
        }

        __aicore__ inline void CalcTileCopyParams(uint32_t inputDTypeLen)
        {
            // zone1:baseRow-baseCol  zone2:baseRow-tailCol  zone3:tailRow-baseCol  zone4:tailRow-tailCol
            // base row , base col
            bool aligned = (is32BAligned == 1);
            auto alignedLen = DEFAULT_MIN_BLOCK_SIZE / inputDTypeLen;
            baseRowBaseColCalLen = aligned ? (baseRowLen * baseColLen) : (baseRowLen * ALIGNUP(baseColLen, alignedLen));
            CalcOneTileCopyParam(baseRowLen, baseColLen, inputDTypeLen, baseRowBaseColCopyParam);

            // base row , tail col
            baseRowTailColCalLen = aligned ? (baseRowLen * tailColLen) : baseRowLen * ALIGNUP(tailColLen, alignedLen);
            CalcOneTileCopyParam(baseRowLen, tailColLen, inputDTypeLen, baseRowTailColCopyParam);

            // tail row , base col
            tailRowBaseColCalLen = aligned ? (tailRowLen * baseColLen) : tailRowLen * ALIGNUP(baseColLen, alignedLen);
            CalcOneTileCopyParam(tailRowLen, baseColLen, inputDTypeLen, tailRowBaseColCopyParam);

            // tail row , tail col
            tailRowTailColCalLen = aligned ? (tailRowLen * tailColLen) : tailRowLen * ALIGNUP(tailColLen, alignedLen);
            CalcOneTileCopyParam(tailRowLen, tailColLen, inputDTypeLen, tailRowTailColCopyParam);
        }

        __aicore__ inline void CalcOneTileOffsetParam(uint64_t gmRowOffset, uint64_t rowIdx, uint64_t colIdx)
        {
            curTileOffset.rowIndex = rowIdx * baseRowLen;
            curTileOffset.colIndex = colIdx * baseColLen;
            gmOffset = gmRowOffset * colLen + colIdx * baseColLen;
        }

        __aicore__ inline void SetCurTileParam(
                uint64_t calTileLen_, uint64_t calRowLen_, uint64_t calColLen_, CopyParam *copyParam)
        {
            bool aligned = (is32BAligned == 1);
            auto alignedLen = DEFAULT_MIN_BLOCK_SIZE / inputDataType;
            curCalLen = calTileLen_;
            curCalRowLen = calRowLen_;
            curCalColLen = calColLen_;
            curTileCopyParam = copyParam;
        }

        __aicore__ inline void CalcOneTileParam(uint64_t tileIdx)
        {
            uint64_t rowTileIdx = tileIdx / colTileNum;
            uint64_t colTileIdx = tileIdx % colTileNum;
            CalcOneTileOffsetParam(rowTileIdx * baseRowLen, rowTileIdx, colTileIdx);
            if (rowTileIdx < baseRowTileNum) {
                if (colTileIdx < baseColTileNum) {
                    // base row, base col
                    SetCurTileParam(baseRowBaseColCalLen, baseRowLen, baseColLen, &baseRowBaseColCopyParam);
                } else {
                    // base row, tail col
                    SetCurTileParam(baseRowTailColCalLen, baseRowLen, tailColLen, &baseRowTailColCopyParam);
                }
            } else {
                if (colTileIdx < baseColTileNum) {
                    // tail row, base col
                    SetCurTileParam(tailRowBaseColCalLen, tailRowLen, baseColLen, &tailRowBaseColCopyParam);
                } else {
                    // tail row, tail col
                    SetCurTileParam(tailRowTailColCalLen, tailRowLen, tailColLen, &tailRowTailColCopyParam);
                }
            }
        }
    };

#define ROW_COL_QUANT_PROCESS_TILE(gmOffset, copyParam, calLen) \
    CopyIn(gmOffset, copyParam);                                \
    this->Compute(calLen);                                      \
    CopyOut(gmOffset, copyParam);

#define ROW_COL_QUANT_PROCESS(kernelTiling)                                                       \
    do {                                                                                          \
        uint64_t blockNum = GetBlockNum();                                                        \
        uint64_t baseTileNum = kernelTiling.totalTileNum / blockNum;                              \
        uint64_t oneMoreTileCoreNum = kernelTiling.totalTileNum % blockNum;                       \
        uint64_t startTileIdx, endTileIdx;                                                        \
        if (kernelTiling.coreIdx < oneMoreTileCoreNum) {                                          \
            startTileIdx = kernelTiling.coreIdx * (baseTileNum + 1);                              \
            endTileIdx = startTileIdx + baseTileNum + 1;                                          \
        } else {                                                                                  \
            startTileIdx = kernelTiling.coreIdx * baseTileNum + oneMoreTileCoreNum;               \
            endTileIdx = startTileIdx + baseTileNum;                                              \
        }                                                                                         \
        for (uint64_t tileIdx = startTileIdx; tileIdx < endTileIdx; tileIdx++) {                  \
            kernelTiling.CalcOneTileParam(tileIdx);                                               \
            ROW_COL_QUANT_PROCESS_TILE(                                                           \
                kernelTiling.gmOffset, *(kernelTiling.curTileCopyParam), kernelTiling.curCalLen); \
        }                                                                                         \
    } while (0)

    constexpr uint32_t BUFFER_NUM = 1;
    static constexpr float FLOAT_127 = 127.0f;
    static constexpr float FRACTION_127 = 1.0 / FLOAT_127;

    template<typename InType, typename CalType, typename OutType>
    class RowColQuantKernel {
    public:
        __aicore__ inline RowColQuantKernel() {}
        __aicore__ inline ~RowColQuantKernel() {}
        __aicore__ inline void Init(GM_ADDR xGm, GM_ADDR rowAbsGm, GM_ADDR colAbsGm, GM_ADDR rowNormedGm,
                                    GM_ADDR colNormedGm, GM_ADDR rowIdxGm, GM_ADDR colIdxGm, GM_ADDR valueGm,
                                    GM_ADDR tilingGm);

        __aicore__ inline void Process();
    protected:
        __aicore__ inline void InitGmBuffer(GM_ADDR xGm_, GM_ADDR rowAbsGm_, GM_ADDR colAbsGm_, GM_ADDR rowNormedGm_,
                                            GM_ADDR colNormedGm_, GM_ADDR rowIdx_, GM_ADDR colIdx_, GM_ADDR value_);
        __aicore__ inline void InitUbBuffer();
        __aicore__ inline void CopyIn(uint64_t tileOffset, CopyParam &copyParam);
        __aicore__ inline void Compute(uint32_t curTileLen);
        __aicore__ inline void CopyOut(uint64_t tileOffset, CopyParam &copyParam);
        __aicore__ inline void CopyInForOutlier(uint32_t offset, uint32_t calcLen);
        __aicore__ inline void ComputeForOutlier(uint32_t offset, uint32_t calcLen, LocalTensor<int32_t>& outlierIdx, uint32_t& outlierNum);
        __aicore__ inline void CopyOutForColIdx();

    private:
        TPipe pipe;

        GlobalTensor<InType> xGm;
        GlobalTensor<CalType> colAbsMaxGm, rowAbsMaxGm;
        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueColMax, inQueueRowMax, inQueueForOutlier, tmpQueueForOutlier;
        TBuf<TPosition::VECCALC> xFloatBuffer;
        TBuf<TPosition::VECCALC> thresholdDuplicateBuffer;
        TBuf<TPosition::VECCALC> bitmapBuffer;
        TBuf<TPosition::VECCALC> rowNormedSelectBuffer;
        TBuf<TPosition::VECCALC> repeatFloat127Buffer;

        GlobalTensor<OutType> colNormedGm, rowNormedGm;
        GlobalTensor<int32_t> rowIdxGm, colIdxGm;
        GlobalTensor<InType> valGm;
        TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueColNormed, outQueueRowNormed, outQueueRowIdx, outQueueColIdx, outQueueValue;
        uint32_t outlierNum = 0;
        LocalTensor<int32_t> colIdxLocal;
        TQue<QuePosition::VECIN, BUFFER_NUM> *tempQue;

    protected:
        RowColQuantTilingKernel tiling;
    };

    template<typename InType, typename CalType, typename OutType>
    __aicore__ inline void RowColQuantKernel<InType, CalType, OutType>::Init(GM_ADDR xGm, GM_ADDR rowAbsGm,
                                                                             GM_ADDR colAbsGm,
                                                                             GM_ADDR rowNormedGm, GM_ADDR colNormedGm,
                                                                             GM_ADDR rowIdxGm, GM_ADDR colIdxGm,
                                                                             GM_ADDR valueGm, GM_ADDR tilingGm)
    {
        tiling.GetTilingAndOffset(tilingGm, sizeof(InType));
        InitGmBuffer(xGm, rowAbsGm, colAbsGm, rowNormedGm, colNormedGm, rowIdxGm, colIdxGm, valueGm);
        InitUbBuffer();
    }

    template<typename InType, typename CalType, typename OutType>
    __aicore__ inline void RowColQuantKernel<InType, CalType, OutType>::Process()
    {
        if (tiling.coreIdx < tiling.usedCoreNum) {
            if (tiling.is32BAligned == 1) {
                ROW_COL_QUANT_PROCESS(tiling);
            }
        }
        if (tiling.coreIdx < tiling.usedCoreNumForOutlier) {
            if (tiling.isOutlierIndex == 0 && tiling.threshold > 0) {
                for (uint32_t idx = 0; idx < tiling.curCoreParam.loopNum; idx++) {
                    uint32_t offset = idx * tiling.curCoreParam.tileCol + tiling.copyInOffset;
                    CopyInForOutlier(offset, tiling.curCoreParam.tileCol);
                    ComputeForOutlier(offset, tiling.curCoreParam.tileCol, colIdxLocal, outlierNum);
                }
                if (tiling.curCoreParam.isTailExist == 1) {
                    uint32_t offset = tiling.curCoreParam.loopNum * tiling.curCoreParam.tileCol + tiling.copyInOffset;
                    CopyInForOutlier(offset, tiling.curCoreParam.tailCol);
                    ComputeForOutlier(offset, tiling.curCoreParam.tailCol, colIdxLocal, outlierNum);
                }
            }
            if (outlierNum > 0) {
                CopyOutForColIdx();
            }
        }
    }

    template<typename InType, typename CalType, typename OutType>
    __aicore__ inline void RowColQuantKernel<InType, CalType, OutType>::InitGmBuffer(GM_ADDR xGm_, GM_ADDR rowAbsGm_,
                                                                                     GM_ADDR colAbsGm_, GM_ADDR rowNormedGm_,
                                                                                     GM_ADDR colNormedGm_, GM_ADDR rowIdx_,
                                                                                     GM_ADDR colIdx_, GM_ADDR value_)
    {
        this->xGm.SetGlobalBuffer((__gm__ InType*)xGm_, tiling.totalBlockLen);
        this->rowAbsMaxGm.SetGlobalBuffer((__gm__ CalType*)rowAbsGm_, tiling.rowLen);
        this->colAbsMaxGm.SetGlobalBuffer((__gm__ CalType*)colAbsGm_, tiling.colLen);

        this->rowNormedGm.SetGlobalBuffer((__gm__ OutType*)rowNormedGm_, tiling.totalBlockLen);
        this->colNormedGm.SetGlobalBuffer((__gm__ OutType*)colNormedGm_, tiling.totalBlockLen);

        // col index
        this->rowIdxGm.SetGlobalBuffer((__gm__ int32_t*)rowIdx_, tiling.outliersNum);
        this->colIdxGm.SetGlobalBuffer((__gm__ int32_t*)colIdx_, tiling.outliersNum);
        this->valGm.SetGlobalBuffer((__gm__ InType*)value_, tiling.outliersNum);
    }

    template<typename InType, typename CalType, typename OutType>
    __aicore__ inline void RowColQuantKernel<InType, CalType, OutType>::InitUbBuffer()
    {
        if (tiling.coreIdx < tiling.usedCoreNum) {
            pipe.InitBuffer(inQueueX, BUFFER_NUM, tiling.tileLength * sizeof(InType));
            pipe.InitBuffer(inQueueRowMax, BUFFER_NUM, tiling.rowAlignedLen * sizeof(CalType));
            pipe.InitBuffer(thresholdDuplicateBuffer, tiling.tileLength * sizeof(CalType));
            pipe.InitBuffer(bitmapBuffer, tiling.tileLength * sizeof(int8_t));
            pipe.InitBuffer(xFloatBuffer, tiling.tileLength * sizeof(CalType));
            pipe.InitBuffer(rowNormedSelectBuffer, tiling.tileLength * sizeof(CalType));
            pipe.InitBuffer(outQueueRowNormed, BUFFER_NUM, tiling.tileLength * sizeof(OutType));

            if (tiling.isColQuant == 1){
                pipe.InitBuffer(inQueueColMax, BUFFER_NUM, tiling.baseColLen * sizeof(CalType));
                pipe.InitBuffer(repeatFloat127Buffer, tiling.baseColLen * sizeof(CalType));
                pipe.InitBuffer(outQueueColNormed, BUFFER_NUM, tiling.tileLength * sizeof(OutType));
            }
        }
        if (tiling.coreIdx < tiling.usedCoreNumForOutlier) {
            outlierNum = 0;
            if (tiling.isOutlierIndex == 1){
                pipe.InitBuffer(outQueueRowIdx, BUFFER_NUM, tiling.tileLength * sizeof(int32_t));
                pipe.InitBuffer(outQueueColIdx, BUFFER_NUM, tiling.tileLength * sizeof(int32_t));
                pipe.InitBuffer(outQueueValue, BUFFER_NUM, tiling.tileLength * sizeof(InType));
                tempQue = &inQueueX;
            } else{
                pipe.InitBuffer(inQueueForOutlier, BUFFER_NUM, tiling.curCoreParam.colLen * sizeof(CalType));
                pipe.InitBuffer(outQueueColIdx, BUFFER_NUM, tiling.curCoreParam.colLen * sizeof(int32_t));
                colIdxLocal = outQueueColIdx.AllocTensor<int32_t>();
                pipe.InitBuffer(tmpQueueForOutlier, BUFFER_NUM, tiling.curCoreParam.colLen * sizeof(CalType));
                tempQue = &tmpQueueForOutlier;
            }
        }
    }

    template<typename InType, typename CalType, typename OutType>
    __aicore__ inline void RowColQuantKernel<InType, CalType, OutType>::CopyIn(uint64_t tileOffset, CopyParam &copyParam)
    {
        DataCopyParams copyInParams = {copyParam.blockCount, copyParam.blockLen, copyParam.stride, 0};
        LocalTensor<InType> xLocal = inQueueX.AllocTensor<InType>();
        ::DataCopy(xLocal, xGm[tileOffset], copyInParams);
        inQueueX.EnQue(xLocal);

        LocalTensor<CalType> rowMaxLocal = inQueueRowMax.AllocTensor<CalType>();
        ::DataCopy(rowMaxLocal, rowAbsMaxGm[tiling.curTileOffset.rowIndex], tiling.rowAlignedLen);
        inQueueRowMax.EnQue(rowMaxLocal);

        if (tiling.isColQuant == 1) {
            LocalTensor <CalType> colMaxLocal = inQueueColMax.AllocTensor<CalType>();
            ::DataCopy(colMaxLocal, colAbsMaxGm[tiling.curTileOffset.colIndex], tiling.baseColLen);
            inQueueColMax.EnQue(colMaxLocal);
        }
    }

    template<typename InType, typename CalType, typename OutType>
    __aicore__ inline void RowColQuantKernel<InType, CalType, OutType>::Compute(uint32_t curCalLen)
    {
        LocalTensor<InType> xLocal = inQueueX.DeQue<InType>();
        LocalTensor<CalType> xFloatLocal = xFloatBuffer.Get<CalType>();
        ::Cast(xFloatLocal, xLocal, RoundMode::CAST_NONE, curCalLen);
        pipe_barrier(PIPE_V);

        LocalTensor<CalType> rowNormedSelectLocal = rowNormedSelectBuffer.Get<CalType>();
        if (tiling.threshold > 0){
            ::Abs(rowNormedSelectLocal, xFloatLocal, curCalLen);

            pipe_barrier(PIPE_V);
            LocalTensor thresholdDuplicateLocal = thresholdDuplicateBuffer.Get<CalType>();
            ::Duplicate(thresholdDuplicateLocal, tiling.threshold, curCalLen);

            pipe_barrier(PIPE_V);
            LocalTensor<OutType> bitmapLocal = bitmapBuffer.Get<OutType>();
            ::Compare(bitmapLocal, rowNormedSelectLocal, thresholdDuplicateLocal, CMPMODE::LT, (curCalLen + 63) / 64 * 64);
            pipe_barrier(PIPE_V);

            ::Select(rowNormedSelectLocal, bitmapLocal, xFloatLocal, 0.f, SELMODE::VSEL_TENSOR_SCALAR_MODE, curCalLen);

            if (tiling.isOutlierIndex == 1){
                LocalTensor<uint16_t> bitmap16Buf = bitmapLocal.template ReinterpretCast<uint16_t>();
                Not(bitmap16Buf, bitmap16Buf, curCalLen / 16);
                pipe_barrier(PIPE_V);
                uint64_t resv_cnt = 1;
                GatherMask(xLocal, xLocal, bitmap16Buf, true, curCalLen, {1, 1, 8, 8}, resv_cnt);
                pipe_barrier(PIPE_V);
                set_flag(PIPE_V, PIPE_S, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
                outlierNum = outlierNum + static_cast<int32_t>(resv_cnt);
                set_flag(PIPE_S, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
            }
        } else {
            ::Adds(rowNormedSelectLocal, xFloatLocal, 0.f, curCalLen);
        }

        pipe_barrier(PIPE_V);
        LocalTensor<CalType> rowMaxLocal = inQueueRowMax.DeQue<CalType>();
        uint32_t rowBeginOffset = 0;

        for (uint32_t r = 0; r < tiling.curCalRowLen; r++) {
            set_flag(PIPE_V, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
            auto rowAbsMax = rowMaxLocal.GetValue(r);
            rowBeginOffset = r * tiling.curCalColLen;
            CalType factor = (rowAbsMax == 0 ? 0.f : FLOAT_127 / rowAbsMax);
            set_flag(PIPE_S, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
            ::Muls(rowNormedSelectLocal[rowBeginOffset], rowNormedSelectLocal[rowBeginOffset], factor, tiling.curCalColLen);
        }
        pipe_barrier(PIPE_V);

        inQueueRowMax.FreeTensor(rowMaxLocal);

        LocalTensor<int16_t> tempInt16Local = xLocal.template ReinterpretCast<int16_t>();
        ::Cast(tempInt16Local, rowNormedSelectLocal, RoundMode::CAST_RINT, curCalLen);
        pipe_barrier(PIPE_V);

        LocalTensor<half> temphalfLocal = rowNormedSelectLocal.template ReinterpretCast<half>();
        ::Cast(temphalfLocal, tempInt16Local, RoundMode::CAST_NONE, curCalLen);
        pipe_barrier(PIPE_V);

        LocalTensor<OutType> rowNormedLocal = outQueueRowNormed.AllocTensor<OutType>();
        ::Cast(rowNormedLocal, temphalfLocal, RoundMode::CAST_NONE, curCalLen);
        pipe_barrier(PIPE_V);

        outQueueRowNormed.EnQue(rowNormedLocal);

        if (tiling.isColQuant == 1) {
            LocalTensor<CalType> colMaxLocal = inQueueColMax.DeQue<CalType>();
            LocalTensor<CalType> repeatFloat127Local = repeatFloat127Buffer.Get<CalType>();
            ::Duplicate(repeatFloat127Local, FLOAT_127, tiling.curCalColLen);
            pipe_barrier(PIPE_V);
            ::Div(colMaxLocal, repeatFloat127Local, colMaxLocal, tiling.curCalColLen);
            pipe_barrier(PIPE_V);

            for (auto r = 0; r < tiling.curCalRowLen; ++r) {
                set_flag(PIPE_V, PIPE_S, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
                rowBeginOffset = r * tiling.curCalColLen;
                set_flag(PIPE_S, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
                ::Mul(xFloatLocal[rowBeginOffset], xFloatLocal[rowBeginOffset], colMaxLocal, tiling.curCalColLen);
            }

            LocalTensor<OutType> colNormedLocal = outQueueColNormed.AllocTensor<OutType>();

            ::Cast(tempInt16Local, xFloatLocal, RoundMode::CAST_RINT, curCalLen);
            pipe_barrier(PIPE_V);
            ::Cast(temphalfLocal, tempInt16Local, RoundMode::CAST_NONE, curCalLen);
            pipe_barrier(PIPE_V);
            ::Cast(colNormedLocal, temphalfLocal, RoundMode::CAST_NONE, curCalLen);
            pipe_barrier(PIPE_V);

            outQueueColNormed.EnQue<OutType>(colNormedLocal);
            inQueueColMax.FreeTensor(colMaxLocal);
        }
        inQueueX.FreeTensor(xLocal);
    }

    template<typename InType, typename CalType, typename OutType>
    __aicore__ inline void RowColQuantKernel<InType, CalType, OutType>::CopyOut(uint64_t tileOffset, CopyParam &copyParam)
    {
        DataCopyParams copyOutParams = {copyParam.blockCount, copyParam.blockLen_int8, 0, copyParam.stride_int8};
        LocalTensor<OutType> rowNormedLocal = outQueueRowNormed.DeQue<OutType>();
        ::DataCopy(rowNormedGm[tileOffset], rowNormedLocal, copyOutParams);
        outQueueRowNormed.FreeTensor(rowNormedLocal);

        if (tiling.isColQuant == 1) {
            LocalTensor <OutType> colNormedLocal = outQueueColNormed.DeQue<OutType>();
            ::DataCopy(colNormedGm[tileOffset], colNormedLocal, copyOutParams);
            outQueueColNormed.FreeTensor(colNormedLocal);
        }
    }

    template<typename InType, typename CalType, typename OutType>
    __aicore__ inline void RowColQuantKernel<InType, CalType, OutType>::CopyInForOutlier(uint32_t offset, uint32_t calcLen)
    {
        LocalTensor<CalType> colMaxLocal = inQueueForOutlier.AllocTensor<CalType>();
        ::DataCopy(colMaxLocal, colAbsMaxGm[offset], calcLen);
        inQueueForOutlier.EnQue(colMaxLocal);
    }

    template<typename InType, typename CalType, typename OutType>
    __aicore__ inline void RowColQuantKernel<InType, CalType, OutType>::ComputeForOutlier(uint32_t offset,
                                                                                          uint32_t calcLen,
                                                                                          LocalTensor<int32_t>& outlierIdx,
                                                                                          uint32_t& outlierNum)
    {
        LocalTensor<CalType> colMaxLocal = inQueueForOutlier.DeQue<CalType>();
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        for (uint32_t c = 0; c < calcLen; c++) {
            auto curVal = colMaxLocal.GetValue(c);
            if (curVal >= tiling.threshold) {
                outlierIdx.SetValue(outlierNum, offset + c);
                outlierNum += 1;
            }
        }
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
        inQueueForOutlier.FreeTensor(colMaxLocal);
    }

    template<typename InType, typename CalType, typename OutType>
    __aicore__ inline void RowColQuantKernel<InType, CalType, OutType>::CopyOutForColIdx()
    {
        DataCopyParams copyParams{1, static_cast<uint16_t>(outlierNum * sizeof(int32_t)), 0, 0};
        ::DataCopyPad(colIdxGm[tiling.coreIdx * tiling.colLen], colIdxLocal, copyParams);
        outQueueColIdx.FreeTensor(colIdxLocal);
    }
}  // namespace row_col_quant_kernel


namespace row_col_stats_fp16_kernel {
    template<typename InType, typename OutType, uint16_t BUFFER_NUM = 1>
    class RowColStatsKernelFp16 {
    public:
        __aicore__ inline RowColStatsKernelFp16() {}
        __aicore__ inline void Init(GM_ADDR x, GM_ADDR rmaxbuf, GM_ADDR cmaxbuf, GM_ADDR cnt, GM_ADDR tiling_gm)
        {
            auto tiling_host = (__gm__ RowColStatsTiling*)tiling_gm;
            tiling.M = tiling_host->M;
            tiling.K = tiling_host->K;
            tiling.threshold = tiling_host->threshold;
            tiling.is_outlier_index = tiling_host->is_outlier_index;
            tiling.use_gather_mask = tiling_host->use_gather_mask;
            uint32_t blkid = get_block_idx();
            tiling.core_m = tiling_host->core_rows[blkid];
            tiling.core_k = tiling_host->core_cols[blkid];
            uint64_t start_off = tiling_host->start_offs[blkid];
            tiling.align_k = AlignTo16(tiling.core_k);
            tiling.align_m = AlignTo16(tiling.core_m);
            tiling.align_K = AlignTo16(tiling.K);
            tiling.ub_sizes = tiling_host-> ub_sizes;
            uint32_t max_elements_per_ub = tiling_host->max_elements_per_ub;
            tiling.tile_lines = CalcTileLines(tiling.ub_sizes, max_elements_per_ub, InTypeSize, tiling.align_k, BUFFER_NUM);
            tiling.start_off = start_off;

            // number of tile(tileLength per tile) on this core, don't include tail tile
            tiling.tile_num = (tiling.tile_lines == 0) ? 0 : tiling.core_m / tiling.tile_lines;
            tiling.tail_tile_lines = tiling.core_m - tiling.tile_lines * tiling.tile_num;
            tiling.last_tile_idx = (tiling.tail_tile_lines > 0) ? tiling.tile_num : (tiling.tile_num - 1);

            xGm.SetGlobalBuffer((__gm__ InType*)x + start_off, (tiling.M * tiling.K - start_off) * InTypeSize);
            rmaxGm.SetGlobalBuffer((__gm__ OutType*)rmaxbuf + (start_off / tiling.K), tiling.core_m * sizeof(OutType));
            cmaxGm.SetGlobalBuffer((__gm__ OutType*)cmaxbuf + (start_off % tiling.K), tiling.core_k * sizeof(OutType));
            cntGm.SetGlobalBuffer((__gm__ int32_t*)cnt, sizeof(int32_t));

            uint32_t max_lines_per_tile =
                    tiling.tile_lines > tiling.tail_tile_lines ? tiling.tile_lines : tiling.tail_tile_lines;
            pipe.InitBuffer(inQueue, BUFFER_NUM,
                            AlignToN(max_lines_per_tile * tiling.align_k * InTypeSize, ONE_REPEAT_BYTE_SIZE));
            pipe.InitBuffer(calcTBuf, AlignToN(max_lines_per_tile * tiling.align_k * InTypeSize, ONE_REPEAT_BYTE_SIZE));
            pipe.InitBuffer(rmaxQueue, BUFFER_NUM, AlignToN(max_lines_per_tile * sizeof(OutType), ONE_BLK_SIZE));
            pipe.InitBuffer(cmaxQueue, BUFFER_NUM, tiling.align_k * sizeof(OutType));
            pipe.InitBuffer(cntQueue, 1, 32);
            pipe.InitBuffer(bitmapTBuf, AlignToN(max_lines_per_tile * tiling.align_k / UINT8_BITS, ONE_BLK_SIZE));
            calcBuf = calcTBuf.Get<InType>();
            bitmapBuf = bitmapTBuf.Get<uint8_t>();
            outlier_cnt = 0;

            cntsBuf = cntQueue.template AllocTensor<InType>();
            cmaxCalcBuf = cmaxQueue.template AllocTensor<InType>();
            LocalTensor<int32_t> cnts32Buf = cntsBuf.template ReinterpretCast<int32_t>();
            Duplicate(cnts32Buf, (int32_t)0, 1);
            cntQueue.EnQue(cnts32Buf);
            DataCopyExtParams copyParams{1, (uint32_t)sizeof(int32_t), 0, 0, 0};
            cnts32Buf = cntQueue.template DeQue<int32_t>();

            LocalTensor<OutType> cmaxFloatBuf = cmaxCalcBuf.template ReinterpretCast<OutType>();
            Duplicate(cmaxFloatBuf, (float)0.0, tiling.core_k);
            cmaxQueue.EnQue(cmaxFloatBuf);
            cmaxFloatBuf = cmaxQueue.template DeQue<float>();

            rmaxCalcBuf = rmaxQueue.template AllocTensor<InType>();
            LocalTensor<OutType> rmaxFloatBuf = rmaxCalcBuf.template ReinterpretCast<OutType>();

            Duplicate(rmaxFloatBuf, (float)0.0,
                      AlignToN(max_lines_per_tile * sizeof(OutType), ONE_BLK_SIZE) / sizeof(OutType));

            rmaxQueue.EnQue(rmaxFloatBuf);
            rmaxFloatBuf = rmaxQueue.template DeQue<float>();
            rmaxQueue.template FreeTensor<OutType>(rmaxFloatBuf);
        }

        __aicore__ inline void Process()
        {
            do {
                for (uint64_t i = 0; i < tiling.tile_num; i++) {
                    CopyIn(i, tiling.tile_lines);
                    Compute(i, tiling.tile_lines);
                    CopyOut(i, tiling.tile_lines);
                }
                if (tiling.tail_tile_lines > 0) {
                    CopyIn(tiling.tile_num, tiling.tail_tile_lines);
                    Compute(tiling.tile_num, tiling.tail_tile_lines);
                    CopyOut(tiling.tile_num, tiling.tail_tile_lines);
                }
            } while(0);
        }

    private:
        __aicore__ inline void CopyIn(int32_t progress, uint32_t cur_lines)
        {
            LocalTensor<InType> xLocal = inQueue.template AllocTensor<InType>();
            if (tiling.K == tiling.align_K) {
                DataCopyParams copyParams{(uint16_t)cur_lines, (uint16_t)(tiling.core_k / 16),
                                          (uint16_t)((tiling.K - tiling.core_k) / 16), 0};
                DataCopy(xLocal, xGm[progress * tiling.tile_lines * tiling.K], copyParams);
            } else {
                DataCopyExtParams copyParams{(uint16_t)cur_lines, (uint32_t)(tiling.core_k * InTypeSize),
                                             (uint32_t)(tiling.K - tiling.core_k) * InTypeSize, 0, 0};
                DataCopyPadExtParams<InType> padParams{(tiling.core_k != tiling.align_k), 0,
                                                       (uint8_t)(tiling.align_k - tiling.core_k), 0};
                DataCopyPad(xLocal, xGm[progress * tiling.tile_lines * tiling.K], copyParams, padParams);
            }
            inQueue.EnQue(xLocal);
        }

        __aicore__ inline void ComputeColMaxs(int32_t progress, uint32_t cur_lines, LocalTensor<InType>& xLocal)
        {
            if (progress == 0) {
                Duplicate(cmaxCalcBuf, (InType)0, tiling.align_k);
            }
            if (cur_lines > 1) {
                uint32_t left_num = cur_lines;
                uint32_t half_num = left_num >> 1;
                Max(calcBuf, xLocal, xLocal[half_num * tiling.align_k], half_num * tiling.align_k);
                if (left_num % 2) {
                    Max(cmaxCalcBuf, cmaxCalcBuf, xLocal[(left_num - 1) * tiling.align_k], tiling.align_k);
                }
                left_num = half_num;
                uint32_t off = 0;
                while(left_num > 1) {
                    half_num = left_num >> 1;
                    Max(calcBuf[(off + half_num) * tiling.align_k], calcBuf[off * tiling.align_k],
                        calcBuf[(off + half_num) * tiling.align_k], half_num * tiling.align_k);
                    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
                    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
                    off += half_num;
                    left_num -= half_num;
                    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
                    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
                }
                Max(cmaxCalcBuf, cmaxCalcBuf, calcBuf[off * tiling.align_k], tiling.align_k);
            } else {
                Max(cmaxCalcBuf, cmaxCalcBuf, xLocal, tiling.align_k);
            }
        }

        __aicore__ inline void ComputeRowMaxs(int32_t progress, uint32_t cur_lines, LocalTensor<InType>& xLocal)
        {
            uint32_t total_calc_cnt = cur_lines * tiling.align_k;
            half threshold = static_cast<half>(tiling.threshold);
            ComplexCompareScalar(bitmapBuf, xLocal, threshold, CMPMODE::LT, total_calc_cnt);
            ComplexSelectScalar(calcBuf, bitmapBuf, xLocal, (InType)0, total_calc_cnt);

            LocalTensor<InType> rmaxBuf = rmaxQueue.template AllocTensor<InType>();
            for (uint32_t i = 0; i < cur_lines; i++) {
                AscendC::ReduceMax<InType>(calcBuf[i], calcBuf[i * tiling.align_k], xLocal, (int32_t)tiling.core_k, false);
            }
            LocalTensor<OutType> rmaxFloatBuf = rmaxBuf.template ReinterpretCast<OutType>();
            Cast(rmaxFloatBuf, calcBuf, RoundMode::CAST_NONE, cur_lines);
            rmaxQueue.EnQue(rmaxFloatBuf);

            if (tiling.is_outlier_index) {
                CalcTilingOutlierCnts(progress, cur_lines, xLocal);
            }
        }

        __aicore__ inline void CalcTilingOutlierCnts(int32_t progress, uint32_t cur_lines, LocalTensor<InType>& xLocal)
        {
            uint32_t total_calc_cnt = cur_lines * tiling.align_k;
            LocalTensor<uint16_t> bitmap16Buf = bitmapBuf.template ReinterpretCast<uint16_t>();
            Not(bitmap16Buf, bitmap16Buf, total_calc_cnt / 16);
            uint64_t resv_cnt = 1;
            GatherMask(calcBuf, calcBuf, bitmap16Buf, true, total_calc_cnt, {1, 1, 8, 8}, resv_cnt);
            set_flag(PIPE_V, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
            outlier_cnt += (int32_t)resv_cnt;
            set_flag(PIPE_S, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
        }

        __aicore__ inline void CalcAllOutlierCnts(int32_t progress)
        {
            LocalTensor<int32_t> cnts32Buf = cntsBuf.template ReinterpretCast<int32_t>();
            Duplicate(cnts32Buf, (int32_t)outlier_cnt, 1);
            cntQueue.EnQue(cnts32Buf);
        }

        template<typename T>
        __aicore__ inline void CalcAllOutlierCols(LocalTensor<T>& cmaxBuf, LocalTensor<T>& xLocal)
        {
            T threshold = static_cast<T>(tiling.threshold);
            ComplexCompareScalar(bitmapBuf, cmaxBuf, threshold, CMPMODE::GE, tiling.core_k);
            LocalTensor<int32_t> cnts32Buf = cntsBuf.template ReinterpretCast<int32_t>();
            LocalTensor<T> cntsTBuf = cntsBuf.template ReinterpretCast<T>();
            LocalTensor<T> calcTBuf = calcBuf.template ReinterpretCast<T>();
            uint64_t resv_cnt = 1;
            LocalTensor<uint16_t> bitmap16Buf = bitmapBuf.template ReinterpretCast<uint16_t>();
            GatherMask(calcTBuf, calcTBuf, bitmap16Buf, true, tiling.core_k, {1, 1, 8, 8}, resv_cnt);
            set_flag(PIPE_V, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
            outlier_cnt += (int32_t)resv_cnt;
            set_flag(PIPE_S, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
            Duplicate(cnts32Buf, (int32_t)outlier_cnt, 1);
            cntQueue.EnQue(cnts32Buf);
        }

        __aicore__ inline void Compute(int32_t progress, uint32_t cur_lines)
        {
            LocalTensor<InType> xLocal = inQueue.template DeQue<InType>();
            Abs(xLocal, xLocal, cur_lines * tiling.align_k);

            ComputeColMaxs(progress, cur_lines, xLocal);

            if (tiling.threshold > 0) {
                ComputeRowMaxs(progress, cur_lines, xLocal);
                if (progress == tiling.last_tile_idx) {
                    if (tiling.is_outlier_index) {
                        CalcAllOutlierCnts(progress);
                    } else if (tiling.core_m == tiling.M) {
                        CalcAllOutlierCols<InType>(cmaxCalcBuf, xLocal);
                    }
                }
            } else {
                LocalTensor<InType> rmaxBuf = rmaxQueue.template AllocTensor<InType>();
                uint32_t rid = 0;
                for (uint32_t i = 0; i < cur_lines; i++) {
                    ReduceMax<InType>(xLocal[i], xLocal[i * tiling.align_k], calcBuf, (int32_t)tiling.core_k);
                }
                LocalTensor<OutType> rmaxFloatBuf = rmaxBuf.template ReinterpretCast<OutType>();
                Cast(rmaxFloatBuf, xLocal, RoundMode::CAST_NONE, cur_lines);
                rmaxQueue.EnQue(rmaxFloatBuf);

                if (progress == tiling.last_tile_idx) {
                    LocalTensor<int32_t> cnts32Buf = cntsBuf.template ReinterpretCast<int32_t>();
                    Duplicate(cnts32Buf, (int32_t)outlier_cnt, 1);
                    cntQueue.EnQue(cnts32Buf);
                }
            }
            inQueue.FreeTensor(xLocal);

            if (progress == tiling.last_tile_idx) {
                LocalTensor<OutType> cmaxFloatBuf = cmaxCalcBuf.template ReinterpretCast<OutType>();
                ComplexCopy(calcBuf, cmaxCalcBuf, tiling.core_k);
                Cast(cmaxFloatBuf, calcBuf, RoundMode::CAST_NONE, tiling.core_k);
                cmaxQueue.EnQue(cmaxFloatBuf);
            }
        }

        __aicore__ inline void CopyOutRmax(int32_t progress, uint32_t cur_lines) {
            LocalTensor<OutType> rmaxBuf = rmaxQueue.template DeQue<OutType>();
            if (cur_lines % 16) {
                DataCopyExtParams copyParams{1, (uint32_t)(cur_lines * sizeof(OutType)), 0, 0, 0};
                if (tiling.core_k == tiling.K) {
                    DataCopyPad(rmaxGm[progress * tiling.tile_lines], rmaxBuf, copyParams);
                } else {
                    SetAtomicMax<OutType>();
                    DataCopyPad(rmaxGm[progress * tiling.tile_lines], rmaxBuf, copyParams);
                    SetAtomicNone();
                }
            } else {
                if (tiling.core_k == tiling.K) {
                    DataCopy(rmaxGm[progress * tiling.tile_lines], rmaxBuf, cur_lines);
                } else {
                    SetAtomicMax<OutType>();
                    DataCopy(rmaxGm[progress * tiling.tile_lines], rmaxBuf, cur_lines);
                    SetAtomicNone();
                }
            }
            rmaxQueue.FreeTensor(rmaxBuf);
        }

        __aicore__ inline void CopyOutCmax() {
            LocalTensor<OutType> cmaxBuf = cmaxQueue.template DeQue<OutType>();
            if (tiling.core_k == tiling.align_k) {
                if (tiling.core_m == tiling.M) {
                    DataCopy(cmaxGm, cmaxBuf, tiling.core_k);
                } else {
                    SetAtomicMax<OutType>();
                    DataCopy(cmaxGm, cmaxBuf, tiling.core_k);
                    SetAtomicNone();
                }
            } else {
                DataCopyExtParams copyParams{1, (uint32_t)(tiling.core_k * sizeof(OutType)), 0, 0, 0};
                if (tiling.core_m == tiling.M) {
                    DataCopyPad(cmaxGm, cmaxBuf, copyParams);
                } else {
                    SetAtomicMax<OutType>();
                    DataCopyPad(cmaxGm, cmaxBuf, copyParams);
                    SetAtomicNone();
                }
            }
            cmaxQueue.FreeTensor(cmaxBuf);
        }

        __aicore__ inline void CalcFinalOutlierCols() {
            if (tiling.start_off / tiling.K > 0) {
                LocalTensor<int32_t> cnts32Buf = cntsBuf.template ReinterpretCast<int32_t>();
                Duplicate(cnts32Buf, (int32_t)0, 1);
                cntQueue.EnQue(cnts32Buf);
                return;
            }
            LocalTensor<float> cmaxFBuf = cmaxQueue.template AllocTensor<float>();
            if (tiling.K == tiling.align_K) {
                DataCopy(cmaxFBuf, cmaxGm, tiling.core_k);
            } else {
                uint32_t blockLen = (uint32_t)(tiling.core_k * sizeof(float));
                DataCopyExtParams copyParams{(uint16_t)1, blockLen, 0, 0, 0};
                DataCopyPadExtParams<float> padParams{true, 0, (uint8_t)((AlignTo32(blockLen) - blockLen) / sizeof(float)), 0};
                DataCopyPad(cmaxFBuf, cmaxGm, copyParams, padParams);
            }
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            LocalTensor<float> xLocal = inQueue.template AllocTensor<float>();
            CalcAllOutlierCols<float>(cmaxFBuf, xLocal);
            inQueue.FreeTensor(xLocal);
            cmaxQueue.FreeTensor(cmaxFBuf);
        }

        __aicore__ inline void CopyOutOutlierCnt() {
            if ((!tiling.is_outlier_index) && (tiling.threshold > 0) && (tiling.core_m != tiling.M)) {
                CalcFinalOutlierCols();
            }
            DataCopyExtParams copyParams{1, (uint32_t)sizeof(int32_t), 0, 0, 0};
            LocalTensor<int32_t> cnts32Buf = cntQueue.template DeQue<int32_t>();
            SetAtomicAdd<int32_t>();
            DataCopyPad(cntGm, cnts32Buf, copyParams);
            SetAtomicNone();
            cntQueue.FreeTensor(cnts32Buf);
        }

        __aicore__ inline void CopyOut(int32_t progress, uint32_t cur_lines)
        {
            CopyOutRmax(progress, cur_lines);
            if (progress == tiling.last_tile_idx) {
                CopyOutCmax();
                CopyOutOutlierCnt();
            }
        }

        __aicore__ inline uint32_t
        CalcTileLines(uint32_t ub_size, uint64_t max_datas_per_ub, uint32_t dtype_len, uint32_t align_k, uint32_t buffer_num) {
            uint32_t tiling_lines = (ub_size - buffer_num * 320 - 320 - 4 * buffer_num * align_k) * 8
                                    / ((dtype_len * (buffer_num + 1) * 8 + 1) * align_k + 4 * 8 * buffer_num);
            uint32_t align_num = 32 / sizeof(uint16_t);
            uint32_t aligned_tiling_lines = tiling_lines / align_num * align_num;
            tiling_lines = (aligned_tiling_lines == 0) ? tiling_lines : aligned_tiling_lines;

            return tiling_lines;
        }

        static constexpr uint32_t UINT8_BITS = ONE_BYTE_BIT_SIZE;

        template <typename T>
        __aicore__ inline void ComplexCompareScalar(const LocalTensor<uint8_t>& dstLocal, const LocalTensor<T>& src0Local,
                                                    const T src1Scalar, CMPMODE cmpMode, uint32_t calCount)
        {
            UnaryRepeatParams repeatParams{1, 1, 1, 8};
            if constexpr (std::is_same_v<T, float>) {
                uint32_t repeat = (calCount % 64) ? (calCount / 64 + 1) : (calCount / 64);
                uint32_t off = 0;
                while (repeat > 248) {
                    CompareScalar(dstLocal[off * 8], src0Local[off * 64], src1Scalar, cmpMode, 64, 248, repeatParams);
                    repeat -= 248;
                    off += 248;
                }
                if (repeat > 0) {
                    CompareScalar(dstLocal[off * 8], src0Local[off * 64], src1Scalar, cmpMode, 64, repeat, repeatParams);
                    off += repeat;
                }
            } else if constexpr (std::is_same_v<T, half>)  {
                uint32_t repeat = (calCount % 128) ? (calCount / 128 + 1) : (calCount / 128);
                uint32_t off = 0;

                while (repeat > 254) {
                    CompareScalar(dstLocal[off * 16], src0Local[off * 128], src1Scalar, cmpMode, 128, 254, repeatParams);
                    repeat -= 254;
                    off += 254;
                }
                if (repeat > 0) {
                    CompareScalar(dstLocal[off * 16], src0Local[off * 128], src1Scalar, cmpMode, 128, repeat, repeatParams);
                    off += repeat;
                }
            }
        }

        template <typename T>
        __aicore__ inline void ComplexSelectScalar(const LocalTensor<T>& dstLocal, const LocalTensor<uint8_t>& selMask,
                                                   const LocalTensor<T>& src0Local, T src1Scalar, uint32_t calCount)
        {
            BinaryRepeatParams repeatParams{1, 1, 1, 8, 8, 1};
            if constexpr (std::is_same_v<T, float>) {
                uint32_t repeat = (calCount % 64) ? (calCount / 64 + 1) : (calCount / 64);
                uint32_t off = 0;
                while (repeat > 248) {
                    Select(dstLocal[off * 64], selMask[off * 8], src0Local[off * 64], src1Scalar,
                           SELMODE::VSEL_TENSOR_SCALAR_MODE, 64, 248, repeatParams);
                    repeat -= 248;
                    off += 248;
                }
                if (repeat > 0) {
                    Select(dstLocal[off * 64], selMask[off * 8], src0Local[off * 64], src1Scalar,
                           SELMODE::VSEL_TENSOR_SCALAR_MODE, 64, (uint8_t)repeat, repeatParams);
                    off += repeat;
                }
            } else if constexpr (std::is_same_v<T, half>)  {
                uint32_t repeat = (calCount % 128) ? (calCount / 128 + 1) : (calCount / 128);
                uint32_t off = 0;
                while (repeat > 254) {
                    Select(dstLocal[off * 128], selMask[off * 16], src0Local[off * 128], src1Scalar,
                           SELMODE::VSEL_TENSOR_SCALAR_MODE, 128, 254, repeatParams);
                    repeat -= 254;
                    off += 254;
                }
                if (repeat > 0) {
                    Select(dstLocal[off * 128], selMask[off * 16], src0Local[off * 128], src1Scalar,
                           SELMODE::VSEL_TENSOR_SCALAR_MODE, 128, (uint8_t)repeat, repeatParams);
                    off += repeat;
                }
            }
        }

        template <typename T>
        __aicore__ inline void ComplexSelectScalar2(const LocalTensor<T>& dstLocal, const LocalTensor<uint8_t>& selMask,
                                                    const LocalTensor<T>& src0Local, T src1Scalar, uint32_t calCount)
        {
            BinaryRepeatParams repeatParams{1, 0, 1, 8, 0, 8};
            if constexpr (std::is_same_v<T, float>) {
                uint32_t repeat = (calCount % 64) ? (calCount / 64 + 1) : (calCount / 64);
                uint32_t off = 0;
                while (repeat > 248) {
                    Select(dstLocal[off * 64], selMask[off * 8], src0Local, src1Scalar,
                           SELMODE::VSEL_TENSOR_SCALAR_MODE, 64, 248, repeatParams);
                    repeat -= 248;
                    off += 248;
                }
                if (repeat > 0) {
                    Select(dstLocal[off * 64], selMask[off * 8], src0Local, src1Scalar,
                           SELMODE::VSEL_TENSOR_SCALAR_MODE, 64, (uint8_t)repeat, repeatParams);
                    off += repeat;
                }
            } else if constexpr (std::is_same_v<T, half>)  {
                uint32_t repeat = (calCount % 128) ? (calCount / 128 + 1) : (calCount / 128);
                uint32_t off = 0;
                while (repeat > 254) {
                    Select(dstLocal[off * 128], selMask[off * 16], src0Local, src1Scalar,
                           SELMODE::VSEL_TENSOR_SCALAR_MODE, 128, 254, repeatParams);
                    repeat -= 254;
                    off += 254;
                }
                if (repeat > 0) {
                    Select(dstLocal[off * 128], selMask[off * 16], src0Local, src1Scalar,
                           SELMODE::VSEL_TENSOR_SCALAR_MODE, 128, (uint8_t)repeat, repeatParams);
                }
            }
        }

        template <typename T>
        __aicore__ inline void ComplexCopy(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal, uint32_t calCount)
        {
            uint32_t repeat_eles = 256 / sizeof(T);
            uint32_t repeat = calCount / repeat_eles;
            CopyRepeatParams repeatParams{1, 1, 8, 8};
            uint32_t off = 0;
            while (repeat > 255) {
                Copy(dstLocal[off * repeat_eles], srcLocal[off * repeat_eles], repeat_eles, 255, repeatParams);
                repeat -= 255;
                off += 255;
            }
            if (repeat > 0) {
                Copy(dstLocal[off * repeat_eles], srcLocal[off * repeat_eles], repeat_eles, (uint8_t)repeat, repeatParams);
                off += repeat;
            }
            repeat = calCount % 128;
            if (repeat) {
                Copy(dstLocal[off * repeat_eles], srcLocal[off * repeat_eles], repeat, (uint8_t)1, repeatParams);
            }
        }

        __aicore__ inline uint32_t AlignTo16(uint32_t n)
        {
            return (n + 15) / 16 * 16;
        }

        __aicore__ inline uint32_t AlignTo32(uint32_t n)
        {
            return (n + 31) / 32 * 32;
        }

        __aicore__ inline uint32_t AlignToN(uint32_t m, uint32_t n)
        {
            return (m + n - 1) / n * n;
        }

    private:
        TPipe pipe;
        TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
        TQue<QuePosition::VECOUT, BUFFER_NUM> rmaxQueue;
        TQue<QuePosition::VECOUT, BUFFER_NUM> cmaxQueue;
        TQue<QuePosition::VECOUT, 1> cntQueue;
        GlobalTensor<InType> xGm;
        GlobalTensor<OutType> rmaxGm;
        GlobalTensor<OutType> cmaxGm;
        GlobalTensor<int32_t> cntGm;
        TBuf<TPosition::VECCALC> calcTBuf;
        TBuf<TPosition::VECCALC> bitmapTBuf;
        LocalTensor<InType> calcBuf;
        LocalTensor<uint8_t> bitmapBuf;
        LocalTensor<InType> cmaxCalcBuf;
        LocalTensor<InType> rmaxCalcBuf;
        LocalTensor<InType> cntsBuf;
        RowColStatsTilingKernel tiling;
        int32_t outlier_cnt;
        static constexpr uint32_t InTypeSize = sizeof(InType);
        static constexpr int32_t InTypeStripe = 32 / sizeof(InType);
        static constexpr uint32_t VEC_REPEAT_SIZE = DEFAULT_BLOCK_SIZE / sizeof(InType);
    };
}

extern "C" {

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

__global__ __aicore__ void row_col_quant(GM_ADDR x, GM_ADDR rowAbsMax, GM_ADDR colAbsMax, GM_ADDR outRowNormed,
                                         GM_ADDR outColNormed, GM_ADDR outliersRowIdx, GM_ADDR outliersColIdx,
                                         GM_ADDR outliersValue, GM_ADDR tiling)
{
    row_col_quant_kernel::RowColQuantKernel<half, float, int8_t> op;
    op.Init(x, rowAbsMax, colAbsMax, outRowNormed, outColNormed, outliersRowIdx, outliersColIdx, outliersValue, tiling);
    op.Process();
}

__global__ __aicore__ void row_col_stats(GM_ADDR x, GM_ADDR rmax, GM_ADDR cmax, GM_ADDR cnt, GM_ADDR tiling)
{
    row_col_stats_fp16_kernel::RowColStatsKernelFp16<half, float, 1> op;
    op.Init(x, rmax, cmax, cnt, tiling);
    op.Process();
}

}