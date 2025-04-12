def quantize_per_tensor(operators):
    for knode in operators:
        print(f"KNNode: {knode}\n")
        if knode["op_type"] == "kn_matmul_op": 
            assert len(knode["input_tensors"]) == 2, "kn_matmul_op with more than 2 inputs!"
            A, B = knode["input_tensors"]
            # get A, B's raw pointer
            print(A)
        elif knode["op_type"] == "kn_customized_op": 
            for tbnode in knode['bgraph']["operators"]: 
                print(f"TBNode: {tbnode}\n")
                if tbnode['op_type'] == "tb_matmul_op": 
                    A, B = tbnode["input_tensors"]
                    print(A)
            
            
    # raise NotImplementedError("per-tensor quantization not implemented yet.")

      
      
def quantize_per_channel(operators):
    raise NotImplementedError("per-channel quantization not implemented yet.")

def quantize_per_block(operators):
    raise NotImplementedError("per-block quantization not implemented yet.")

if __name__ == "__main__":
    pass