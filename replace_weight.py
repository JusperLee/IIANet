import torch
from look2hear.models import IIANet
import yaml
from collections import OrderedDict

# Load training config
with open("checkpoints/vox2/conf.yml", "rb") as f:
    train_conf = yaml.safe_load(f)

pretrained_weight = torch.load("/home/likai/ssd/IIANet/checkpoints/lrs2/best_model.pth", map_location="cpu")
# import pdb; pdb.set_trace()
model = IIANet(**train_conf["audionet"]["audionet_config"])


# new_state_dict = OrderedDict()

# for k, v in pretrained_weight["state_dict"].items():
#     if "audio_model." in k:
#         # print(k)
#         if "mlp_avfusion" in k:
#             k = k.replace("audio_model.", "").replace("mlp_avfusion", "InterA_T")
#             new_state_dict[k] = v
#             print(k)
#             continue
#         elif "concat_block" in k:
#             k = k.replace("audio_model.", "").replace("concat_block", "InterA_B_A")
#             new_state_dict[k] = v
#             print(k)
#             continue
#         elif "video_concat" in k:
#             k = k.replace("audio_model.", "").replace("video_concat", "InterA_B_V")
#             new_state_dict[k] = v
#             print(k)
#             continue
#         elif k.replace("audio_model.", "") in model.state_dict().keys():
#             new_state_dict[k.replace("audio_model.", "")] = v
#             print(k)
#             continue

# model.load_state_dict(new_state_dict, strict=True)

# pretrained_weight["state_dict"] = new_state_dict

# delete keys in pretrained_weight
# for k in list(pretrained_weight.keys()):
#     if k not in ['model_name', 'state_dict', 'model_args', 'infos']:
#         del pretrained_weight[k]
pretrained_weight["model_name"] = "IIANet"
pretrained_weight["model_args"] = train_conf["audionet"]["audionet_config"]
pretrained_weight["infos"] = []
torch.save(pretrained_weight, "checkpoints/lrs2/best_model.pth")

# for k, v in model.state_dict().items():
#     print(k)