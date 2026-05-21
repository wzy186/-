/**
 * @file silu_grad_tiling.h
 */
#ifndef SILU_GRAD_TILING_H
#define SILU_GRAD_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SiluGradTilingData)
  TILING_DATA_FIELD_DEF(uint64_t, smallCoreDataNum);
  TILING_DATA_FIELD_DEF(uint64_t, bigCoreDataNum);
  TILING_DATA_FIELD_DEF(uint64_t, finalBigTileNum);
  TILING_DATA_FIELD_DEF(uint64_t, finalSmallTileNum);
  TILING_DATA_FIELD_DEF(uint64_t, tileDataNum);
  TILING_DATA_FIELD_DEF(uint64_t, smallTailDataNum);
  TILING_DATA_FIELD_DEF(uint64_t, bigTailDataNum);
  TILING_DATA_FIELD_DEF(uint64_t, tailBlockNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SiluGrad, SiluGradTilingData)
} // namespace optiling
#endif // SILU_GRAD_TILING_H