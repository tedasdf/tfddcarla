import torch

modelweight = torch.load(
    "../model_ckpt/models_2022/transfuser/model_seed1_39.pth",
    map_location=torch.device("cpu"),
)

# for model, weight in modelweight.items():
#     print("---------------------")
#     print(model)
#     print(weight.shape)
#     print("---------------------")

from model import LidarCenterNet
from config import GlobalConfig


config = GlobalConfig(root_dir='../data', setting='all')
config.use_target_point_image = bool(1)
config.n_layer = 4
config.use_point_pillars = bool(0)
config.backbone = 'transFuser'

local_rank = 0
device = torch.device('cuda:{}'.format(local_rank))

model = LidarCenterNet(config, device, 'transFuser',"mlp", 'resnet34', 'resnet34', bool(0))
 
# model.load_state_dict(modelweight)
# optimizer.load_state_dict(torch.load(args.load_file.replace("model_", "optimizer_"), map_location=model.device))

# print(model.named_modules())

# for name, module in model.named_modules():

#     print("==============================================")
#     print(f"{name}:")
#     print(f"  in_features  = {module.in_features}")
#     print(f"  out_features = {module.out_features}")
#     print(f"  weight shape = {tuple(module.weight.shape)}")
#     if module.bias is not None:
#         print(f"  bias shape   = {tuple(module.bias.shape)}")


print("done")
# # import torch

# print("GPU:", torch.cuda.get_device_name(0))
# print("PyTorch CUDA build:", torch.version.cuda)
# print("Is CUDA available:", torch.cuda.is_available())

# a = torch.randn(2, 2).cuda()
# b = torch.randn(2, 2).cuda()
# c = torch.cat((a, b), dim=1)
# print(c)
