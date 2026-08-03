// ffi/jasper_ffi_directional_l2_k16.cu
//
// Directional (LSH + PQ) ops for the L2 / k_ranks=16 config bucket. Split out
// from the other directional buckets so nvcc can compile them as independent,
// parallelizable translation units — see jasper_ffi_common.cuh.
#include "jasper_ffi_common.cuh"

namespace jasper_ffi {

JASPER_FOR_EACH_DIRECTIONAL_CONFIG_L2_K16(DEFINE_DIRECTIONAL_OPS)
#undef DEFINE_DIRECTIONAL_OPS

JASPER_FOR_EACH_DIRECTIONAL_CONFIG_L2_K16(EXPORT_DIRECTIONAL_OPS)
#undef EXPORT_DIRECTIONAL_OPS

} // namespace jasper_ffi
