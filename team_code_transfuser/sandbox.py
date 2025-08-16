import torch
print(torch.version.cuda)   # Should show 10.2
print(torch.cuda.is_available())  # Should be True
x = torch.randn(3, device='cuda')
print(x)  # Should work without "no kernel image" error
