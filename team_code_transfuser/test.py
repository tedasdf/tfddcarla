import torch

del tensor  # remove large tensor
torch.cuda.empty_cache()


# Optional: Reset all memory allocations
torch.cuda.reset_peak_memory_stats()
