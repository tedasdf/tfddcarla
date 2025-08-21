# import torch
# import time

# def test_fp16():
#     # Check CUDA availability
#     if not torch.cuda.is_available():
#         print("CUDA is not available. Exiting.")
#         return

#     device = torch.device("cuda")
#     print(f"Using device: {torch.cuda.get_device_name(device)}")

#     # Matrix size
#     size = 4096  
#     a = torch.randn(size, size, device=device, dtype=torch.float16)
#     b = torch.randn(size, size, device=device, dtype=torch.float16)

#     # Warm-up
#     for _ in range(5):
#         _ = torch.matmul(a, b)

#     torch.cuda.synchronize()

#     # Timed run
#     start = time.time()
#     c = torch.matmul(a, b)
#     torch.cuda.synchronize()
#     end = time.time()

#     elapsed = end - start
#     print(f"FP16 matmul completed, result dtype: {c.dtype}")
#     print(f"Time taken for {size}x{size} FP16 matmul: {elapsed:.6f} seconds")

#     # Compute TFLOPS
#     flops = 2 * (size ** 3)
#     tflops = flops / elapsed / 1e12
#     print(f"Achieved performance: {tflops:.2f} TFLOPS")

#     # Accuracy check vs FP32
#     a32 = a.float()
#     b32 = b.float()
#     c32 = torch.matmul(a32, b32)
#     max_diff = (c.float() - c32).abs().max().item()
#     print(f"Max difference vs FP32 result: {max_diff:.6f}")

# if __name__ == "__main__":]


a = 'Completed'
if a is 'Completed':
   print('Failed')
else:
   print(True)