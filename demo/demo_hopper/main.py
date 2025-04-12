import mirage as mi
import torch

config = {
  "M":  16  ,
  "N":  4096,
  "K":  4096,
}

def matmul_fp8(M, N, K):
  kn_graph = mi.new_kernel_graph()
  X = kn_graph.new_input(dims=(M, K), dtype=mi.float16)
  W = kn_graph.new_input(dims=(K, N), dtype=mi.float16)
  

  # launch 64x1x1 blocks, each running a warp group (128 threads)
  tb_graph = mi.new_threadblock_graph(grid_dim=(64,1,1), block_dim=(128,1,1), forloop_range=64, reduction_dimx=64)
  tX = tb_graph.new_input(dtensor=X, input_map=(-1,-1,-1), forloop_dim=1)
  tW = tb_graph.new_input(dtensor=W, input_map=( 1,-1,-1), forloop_dim=0)
  tM = tb_graph.matmul(tX, tW)
  tO = tb_graph.forloop_accum(tM)
  tb_graph.new_output(stensor=tO, output_map=(1, -1, -1))

  O = kn_graph.customized([X, W], tb_graph)

  kn_graph.mark_output(O[0], mode='q')
  kn_graph.mark_output(O[1], mode='q')
  kn_graph.mark_output(O[2], mode='q')
  
  """
  quantize matmul
    mode: [tensor(default)|channel|block]
  """
  kn_graph.quantize(mode="tensor")
#   kn_graph.visualize("matmul")
  return kn_graph

if __name__ == "__main__":
  mirage_dtype = mi.bfloat16
  torch_dtype = mi.convert_dtype_to_torch_type(mirage_dtype)
  
  M, N, K = config["M"], config["N"], config["K"]

  mm = matmul_fp8(M, N, K)
  
  # real inputs
  input_tensors = [
      torch.randn(M, K, dtype=torch_dtype, device='cuda:0'),
      torch.randn(K, N, dtype=torch_dtype, device='cuda:0'),
  ]
  
  # debug: view transpiled CUDA code
  input_strides = [tensor.stride() for tensor in input_tensors]
  p = mi.generate_cuda_program(mm.cygraph, target_cc=86, input_strides=input_strides)
  print(p["code"])
  
  # run kernel graph
  output = mm(inputs=input_tensors)[0]

  # print(output.shape)
  # print(output.stride(0), output.stride(1))