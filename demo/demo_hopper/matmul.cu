#define NUM_GPUS 1
#define USE_NVSHMEM false
#include "runtime.h"
using namespace cute;

__global__ void __launch_bounds__(128) custom_kernel_0(half_t* __restrict__ dtensor10000005_ptr, half_t const* __restrict__ dtensor10000003_ptr, half_t const* __restrict__ dtensor10000004_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor30000013_ptr = (half_t*)(buf + 12416);
  half_t *stensor20000013_ptr = (half_t*)(buf + 4224);
  half_t *stensor20000012_ptr = (half_t*)(buf + 128);
  half_t *stensor20000015_ptr = (half_t*)(buf + 128);
  half_t *stensor30000012_ptr = (half_t*)(buf + 2176);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000003 -> stensor 20000012
  const half_t *dtensor10000003_tile_ptr = dtensor10000003_ptr ;
  using DTensor10000003TileLayout = Layout<Shape<Int<64>, Int<16>>, Stride<Int<1>, Int<4096>>>;
  using STensor20000012InputAtom = tb::InputChunkedAsyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>>, Stride<Int<1>, Int<64>>>{})), DTensor10000003TileLayout, NUM_THREADS>;
  half_t *stensor20000012_async_copy_buf = stensor30000012_ptr;
  // Copy for G->S: dtensor 10000004 -> stensor 20000013
  const half_t *dtensor10000004_tile_ptr = dtensor10000004_ptr  + blockIdx.x*64*1;
  using DTensor10000004TileLayout = Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<4096>>>;
  using STensor20000013InputAtom = tb::InputChunkedAsyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{})), DTensor10000004TileLayout, NUM_THREADS>;
  half_t *stensor20000013_async_copy_buf = stensor30000013_ptr;
  
  
  // S->G copy atoms
  // Copy for S->G: stensor 20000015 -> dtensor 10000005
  half_t *dtensor10000005_tile_ptr = dtensor10000005_ptr  + blockIdx.x*64*1;
  using DTensor10000005TileLayout = Layout<Shape<Int<64>, Int<16>>, Stride<Int<1>, Int<4096>>>;
  using STensor20000015OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000005TileLayout, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>>, Stride<Int<1>, Int<64>>>{})), NUM_THREADS>;
  
  
  using Matmul20000015LayoutA = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000015LayoutB = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000015LayoutC = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000015LayoutAAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000015LayoutBAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000015Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<1>, Int<4>, _1>>, true, false, Matmul20000015LayoutA, Matmul20000015LayoutB, Matmul20000015LayoutC, Matmul20000015LayoutAAligned, Matmul20000015LayoutBAligned,NUM_THREADS, 0, false>;
  auto matmul_20000015_accum = Matmul20000015Kernel::get_mma_rC(thread_idx);
  
  {
    STensor20000013InputAtom::run(stensor20000013_async_copy_buf, dtensor10000004_tile_ptr, thread_idx);
    STensor20000012InputAtom::run(stensor20000012_async_copy_buf, dtensor10000003_tile_ptr, thread_idx);
    cute::cp_async_fence();
  }
  
  // The main loop
  for (int for_idx = 0; for_idx < 64; for_idx++) {
    {
      // Issue async copies for the next round
      if (for_idx+1 != 64) {
        STensor20000013InputAtom::run(stensor20000013_ptr, dtensor10000004_tile_ptr + 262144*(for_idx+1), thread_idx);
        STensor20000012InputAtom::run(stensor20000012_ptr, dtensor10000003_tile_ptr + 64*(for_idx+1), thread_idx);
      }
      cute::cp_async_fence();
      // Wait for the async copies in the last round to finish
      cute::cp_async_wait<1>();
      // Switch buffers
      SWAP(stensor20000013_ptr, stensor20000013_async_copy_buf);
      SWAP(stensor20000012_ptr, stensor20000012_async_copy_buf);
    }
    __syncthreads();
    {
      // OP type: tb_matmul_op
      Matmul20000015Kernel::run(matmul_20000015_accum, stensor20000012_ptr, stensor20000013_ptr, (char*)(buf+0), thread_idx);
    }
  }
  
  // Write back in-register accumulators
  __syncthreads();
  Matmul20000015Kernel::write_back_mma_rC(stensor20000015_ptr, matmul_20000015_accum, thread_idx);
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20000015OutputAtom::run(dtensor10000005_tile_ptr, stensor20000015_ptr, thread_idx);
  }
}


static void _init() {
}


static void _execute_mugraph(std::vector<void const *> input_tensors, std::vector<void*> output_tensors, void* buf, cudaStream_t stream, void * profiler_buffer){
  {
    // OP type: kn_input_op
  }
  {
    // OP type: kn_input_op
  }
  {
    // OP type: kn_customized_op
    half_t *dtensor10000005 = (half_t*)output_tensors.at(0);
    half_t *dtensor10000003 = (half_t*)input_tensors.at(0);
    half_t *dtensor10000004 = (half_t*)input_tensors.at(1);
    dim3 grid_dim(64, 1, 1);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 20608;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_0, cudaFuncAttributeMaxDynamicSharedMemorySize, 20608);
    custom_kernel_0<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000005, dtensor10000003, dtensor10000004);
  }
  {
    // OP type: kn_output_op
  }
}