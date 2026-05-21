/**
 * @file silu_grad.cpp
 */
#include "silu_grad_tiling.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

#define BLOCK_SIZE 256U
#define UB_NUM_FLOAT 8U
#define UB_NUM_FLOAT16_BF16 16U

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    if (context == nullptr || context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    coreNum = ascendcPlatform.GetCoreNum();
    if (coreNum == 0 || ubSize == 0) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeAttrsInfo(
    gert::TilingContext* context, uint64_t ubSize, uint64_t& inputNum, uint64_t& inputBytes, uint64_t& tileBlockNum,
    uint64_t& tileDataNum, uint64_t& inputLengthAlgin)
{
    if (context == nullptr || context->GetInputShape(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t typeLength = 2;
    ge::DataType dt = context->GetInputDesc(0)->GetDataType();
    if (dt == ge::DT_FLOAT) {
        typeLength = 4;
    }
    uint64_t inputLength = inputNum * typeLength;
    if (inputNum == 0) {
        return ge::GRAPH_FAILED;
    }
    inputBytes = inputLength / inputNum;
    if (inputBytes == 0) {
        return ge::GRAPH_FAILED;
    }
    uint64_t ubDataNumber = UB_NUM_FLOAT16_BF16;
    if (context->GetInputDesc(0)->GetDataType() == ge::DT_FLOAT) {
        ubDataNumber = UB_NUM_FLOAT;
    }
    if (BLOCK_SIZE == 0 || ubDataNumber == 0) {
        return ge::GRAPH_FAILED;
    }
    tileBlockNum = (ubSize / BLOCK_SIZE) / ubDataNumber;
    tileDataNum = (tileBlockNum * BLOCK_SIZE) / inputBytes;
    inputLengthAlgin = (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CalculateCoreBlockNums(
    gert::TilingContext* context, uint64_t inputLengthAlgin, int64_t coreNum, uint64_t tileBlockNum, uint64_t inputBytes, uint64_t tileDataNum,
    uint64_t& smallCoreDataNum, uint64_t& bigCoreDataNum, uint64_t& smallTailDataNum, uint64_t& bigTailDataNum, uint64_t& finalSmallTileNum,
    uint64_t& finalBigTileNum, uint64_t& tailBlockNum)
{
    if (0 == BLOCK_SIZE || 0 == coreNum || 0 == tileBlockNum || 0 == inputBytes) {
        return ge::GRAPH_FAILED;
    }
    uint64_t everyCoreInputBlockNum = inputLengthAlgin / BLOCK_SIZE / coreNum;
    tailBlockNum = (inputLengthAlgin / BLOCK_SIZE) % coreNum;
    smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / inputBytes;
    uint64_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;
    finalSmallTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
    smallTailDataNum = smallCoreDataNum - (tileDataNum * smallTileNum);
    smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;

    everyCoreInputBlockNum += 1;
    bigCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / inputBytes;
    uint64_t bigTileNum = everyCoreInputBlockNum / tileBlockNum;
    finalBigTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
    bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
    bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNum;
    ge::graphStatus ret = GetPlatformInfo(context, ubSize, coreNum);
    if (ret != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    uint64_t inputNum, inputBytes, tileBlockNum, tileDataNum, inputLengthAlgin;
    ret = GetShapeAttrsInfo(context, ubSize, inputNum, inputBytes, tileBlockNum, tileDataNum, inputLengthAlgin);
    if (ret != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    SiluGradTilingData tiling;

    if (tileDataNum >= inputNum) {
        coreNum = 1;
    } else {
        coreNum = (static_cast<uint64_t>(coreNum) < inputLengthAlgin / BLOCK_SIZE) ? coreNum : inputLengthAlgin / BLOCK_SIZE;
    }

    uint64_t smallCoreDataNum, bigCoreDataNum, smallTailDataNum, bigTailDataNum;
    uint64_t finalSmallTileNum, finalBigTileNum, tailBlockNum;
    ret = CalculateCoreBlockNums(
        context, inputLengthAlgin, coreNum, tileBlockNum, inputBytes, tileDataNum, smallCoreDataNum, bigCoreDataNum,
        smallTailDataNum, bigTailDataNum, finalSmallTileNum, finalBigTileNum, tailBlockNum);
    if (ret != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    tiling.set_smallCoreDataNum(static_cast<uint64_t>(smallCoreDataNum));
    tiling.set_bigCoreDataNum(static_cast<uint64_t>(bigCoreDataNum));
    tiling.set_tileDataNum(static_cast<uint64_t>(tileDataNum));
    tiling.set_smallTailDataNum(static_cast<uint64_t>(smallTailDataNum));
    tiling.set_bigTailDataNum(static_cast<uint64_t>(bigTailDataNum));
    tiling.set_finalSmallTileNum(static_cast<uint64_t>(finalSmallTileNum));
    tiling.set_finalBigTileNum(static_cast<uint64_t>(finalBigTileNum));
    tiling.set_tailBlockNum(static_cast<uint64_t>(tailBlockNum));

    context->SetBlockDim(coreNum);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* dy_shape = context->GetInputShape(0);
    const gert::Shape* x_shape = context->GetInputShape(1);
    gert::Shape* y_shape = context->GetOutputShape(0);

    int64_t dyRank = dy_shape->GetDimNum();
    int64_t xRank = x_shape->GetDimNum();
    
    if (dyRank != xRank) {
        return ge::GRAPH_FAILED;
    }
    
    for (int64_t i = 0; i < dyRank; i++) {
        if (dy_shape->GetDim(i) != x_shape->GetDim(i)) {
            return ge::GRAPH_FAILED;
        }
    }

    *y_shape = *dy_shape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto dyDataType = context->GetInputDataType(0);
    const auto xDataType = context->GetInputDataType(1);
    
    if (dyDataType != xDataType) {
        return ge::GRAPH_FAILED;
    }
    
    context->SetOutputDataType(0, dyDataType);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class SiluGrad : public OpDef {
public:
    explicit SiluGrad(const char* name) : OpDef(name)
    {
        this->Input("dy")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend310b")
            .AddConfig("ascend910b")
            .AddConfig("ascend910")
            .AddConfig("ascend310p");
    }
};

OP_ADD(SiluGrad);
}