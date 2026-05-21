/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file silu_grad.cpp
 */
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

// 输入dy和x必须是相同类型
static_assert(std::is_same_v<DTYPE_DY, DTYPE_X>, "Inputs dy and x must be the same dtype.");
// 输入类型必须是float16、float32或bfloat16
static_assert(std::is_same_v<DTYPE_DY, half> || std::is_same_v<DTYPE_DY, float> || std::is_same_v<DTYPE_DY, bfloat16_t>, 
              "Input dtype must be float16, float32 or bfloat16");

// 定义输出类型与输入相同
using DTYPE_DX = DTYPE_DY;

class KernelSiluGrad {
public:
    __aicore__ inline KernelSiluGrad() {}
    
    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR x, GM_ADDR dx,
                                uint64_t smallCoreDataNum, uint64_t bigCoreDataNum,
                                uint64_t finalBigTileNum, uint64_t finalSmallTileNum,
                                uint64_t tileDataNum,
                                uint64_t smallTailDataNum, uint64_t bigTailDataNum,
                                uint64_t tailBlockNum)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        uint64_t coreId = GetBlockIdx();
        uint64_t globalBufferIndex = bigCoreDataNum * coreId;
        this->tileDataNum = tileDataNum;
        if (coreId < tailBlockNum) {
            this->coreDataNum = bigCoreDataNum;
            this->tileNum = finalBigTileNum;
            this->tailDataNum = bigTailDataNum;
        } else {
            this->coreDataNum = smallCoreDataNum;
            this->tileNum = finalSmallTileNum;
            this->tailDataNum = smallTailDataNum;
            globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (coreId - tailBlockNum);
        }
        dyGm.SetGlobalBuffer((__gm__ DTYPE_DY*)dy + globalBufferIndex, this->coreDataNum);
        xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x + globalBufferIndex, this->coreDataNum);
        dxGm.SetGlobalBuffer((__gm__ DTYPE_DX*)dx + globalBufferIndex, this->coreDataNum);

        if constexpr (!std::is_same_v<DTYPE_DY, float>) {
            pipe.InitBuffer(castDYBuf, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(castXBuf, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(castDXBuf, this->tileDataNum * sizeof(float));
        }
        pipe.InitBuffer(inQueueDY, BUFFER_NUM, this->tileDataNum * sizeof(DTYPE_DY));
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueDX, BUFFER_NUM, this->tileDataNum * sizeof(DTYPE_DX));
        pipe.InitBuffer(tmpBuf1, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tmpBuf2, this->tileDataNum * sizeof(float));
    }

    __aicore__ inline void Process() {
        int32_t loopCount = this->tileNum;
        this->processDataNum = this->tileDataNum;
        for (int32_t i = 0; i < loopCount - 1; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
        this->processDataNum = this->tailDataNum;
        CopyIn(loopCount - 1);
        Compute(loopCount - 1);
        CopyOut(loopCount - 1);
    }

private:
    __aicore__ inline void CopyIn(int32_t progress) {
        LocalTensor<DTYPE_DY> dyLocal = inQueueDY.AllocTensor<DTYPE_DY>();
        LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        DataCopy(dyLocal, dyGm[progress * this->tileDataNum], this->processDataNum);
        DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
        inQueueDY.EnQue(dyLocal);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t progress) {
        (void)progress;
        if constexpr (std::is_same_v<DTYPE_DY, float>) {
            LocalTensor<float> dyLocal = inQueueDY.DeQue<float>();
            LocalTensor<float> xLocal = inQueueX.DeQue<float>();
            LocalTensor<float> dxLocal = outQueueDX.AllocTensor<float>();
            LocalTensor<float> tmp1 = tmpBuf1.Get<float>();
            LocalTensor<float> tmp2 = tmpBuf2.Get<float>();

            // SiLU 梯度公式: dx = dy * sigmoid(x) * (1 + x * (1 - sigmoid(x)))
            Muls(tmp1, xLocal, -1.0f, this->processDataNum);
            Exp(tmp1, tmp1, this->processDataNum);
            Adds(tmp1, tmp1, 1.0f, this->processDataNum);
            Duplicate(dxLocal, 1.0f, this->processDataNum);
            Div(tmp2, dxLocal, tmp1, this->processDataNum);
            Duplicate(dxLocal, 1.0f, this->processDataNum);
            Sub(tmp1, dxLocal, tmp2, this->processDataNum);
            Mul(tmp1, xLocal, tmp1, this->processDataNum);
            Adds(tmp1, tmp1, 1.0f, this->processDataNum);
            Mul(tmp1, tmp2, tmp1, this->processDataNum);
            Mul(dxLocal, dyLocal, tmp1, this->processDataNum);

            outQueueDX.EnQue<float>(dxLocal);
            inQueueDY.FreeTensor(dyLocal);
            inQueueX.FreeTensor(xLocal);
        } else {
            LocalTensor<DTYPE_DY> dyLocal = inQueueDY.DeQue<DTYPE_DY>();
            LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
            LocalTensor<DTYPE_DX> dxLocal = outQueueDX.AllocTensor<DTYPE_DX>();
            LocalTensor<float> dyCalc = castDYBuf.Get<float>();
            LocalTensor<float> xCalc = castXBuf.Get<float>();
            LocalTensor<float> dxCalc = castDXBuf.Get<float>();
            LocalTensor<float> tmp1 = tmpBuf1.Get<float>();
            LocalTensor<float> tmp2 = tmpBuf2.Get<float>();

            Cast(dyCalc, dyLocal, RoundMode::CAST_NONE, this->processDataNum);
            Cast(xCalc, xLocal, RoundMode::CAST_NONE, this->processDataNum);

            // SiLU 梯度公式: dx = dy * sigmoid(x) * (1 + x * (1 - sigmoid(x)))
            Muls(tmp1, xCalc, -1.0f, this->processDataNum);
            Exp(tmp1, tmp1, this->processDataNum);
            Adds(tmp1, tmp1, 1.0f, this->processDataNum);
            Duplicate(dxCalc, 1.0f, this->processDataNum);
            Div(tmp2, dxCalc, tmp1, this->processDataNum);
            Duplicate(dxCalc, 1.0f, this->processDataNum);
            Sub(tmp1, dxCalc, tmp2, this->processDataNum);
            Mul(tmp1, xCalc, tmp1, this->processDataNum);
            Adds(tmp1, tmp1, 1.0f, this->processDataNum);
            Mul(tmp1, tmp2, tmp1, this->processDataNum);
            Mul(dxCalc, dyCalc, tmp1, this->processDataNum);

            Cast(dxLocal, dxCalc, RoundMode::CAST_RINT, this->processDataNum);
            outQueueDX.EnQue<DTYPE_DX>(dxLocal);
            inQueueDY.FreeTensor(dyLocal);
            inQueueX.FreeTensor(xLocal);
        }
    }

    __aicore__ inline void CopyOut(int32_t progress) {
        LocalTensor<DTYPE_DX> dxLocal = outQueueDX.DeQue<DTYPE_DX>();
        DataCopy(dxGm[progress * this->tileDataNum], dxLocal, this->processDataNum);
        outQueueDX.FreeTensor(dxLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueDY;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueDX;
    
    // 中间计算缓冲区 - 2个（复用）
    TBuf<QuePosition::VECCALC> tmpBuf1;
    TBuf<QuePosition::VECCALC> tmpBuf2;
    
    // 类型转换缓冲区 - 3个（仅非float32路径）
    TBuf<QuePosition::VECCALC> castDYBuf;
    TBuf<QuePosition::VECCALC> castXBuf;
    TBuf<QuePosition::VECCALC> castDXBuf;

    // 全局张量
    GlobalTensor<DTYPE_DY> dyGm;
    GlobalTensor<DTYPE_X> xGm;
    GlobalTensor<DTYPE_DX> dxGm;

    // 参数
    uint64_t coreDataNum;
    uint64_t tileNum;
    uint64_t tileDataNum;
    uint64_t tailDataNum;
    uint64_t processDataNum;
};

extern "C" __global__ __aicore__ void silu_grad(GM_ADDR dy, GM_ADDR x, GM_ADDR dx, 
                                                GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelSiluGrad op;
    op.Init(dy, x, dx,
            tiling_data.smallCoreDataNum,
            tiling_data.bigCoreDataNum,
            tiling_data.finalBigTileNum,
            tiling_data.finalSmallTileNum,
            tiling_data.tileDataNum,
            tiling_data.smallTailDataNum,
            tiling_data.bigTailDataNum,
            tiling_data.tailBlockNum);
    op.Process();
}
