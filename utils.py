import torch
from safetensors.torch import load_file
from diffusers import AutoencoderKL, AutoPipelineForInpainting
import os
from tqdm import tqdm
import random
import numpy as np


def get_module_kohya_state_dict(module, prefix: str, dtype: torch.dtype, adapter_name: str = "default"):
    kohya_ss_state_dict = {}
    for peft_key, weight in module.items():
        kohya_key = peft_key.replace("unet.base_model.model", prefix)
        kohya_key = kohya_key.replace("lora_A", "lora_down")
        kohya_key = kohya_key.replace("lora_B", "lora_up")
        kohya_key = kohya_key.replace(".", "_", kohya_key.count(".") - 2)
        kohya_ss_state_dict[kohya_key] = weight.to(dtype)
        # Set alpha parameter
        if "lora_down" in kohya_key:
            alpha_key = f'{kohya_key.split(".")[0]}.alpha'
            kohya_ss_state_dict[alpha_key] = torch.tensor(8).to(dtype)
    return kohya_ss_state_dict

def writeFile(path, info):
    f = open(path, 'w')
    f.write(info)
    f.close()


def prepare_pipe(sd, vae, lora=None):
    
    vae = AutoencoderKL.from_pretrained(vae, subfolder="vae", revision=None).to(torch.float16)
    pipe = AutoPipelineForInpainting.from_pretrained(sd, revision=None, torch_dtype=torch.float16,  safety_checker=None)
    pipe.vae = vae
    # from diffusers import StableDiffusionInpaintPipeline
    # pipe = StableDiffusionInpaintPipeline.from_pretrained(
    #     sd,
    #     torch_dtype=torch.float16,
    # )


    if lora is not None:
        lora_weight     = load_file(lora)
        lora_state_dict = get_module_kohya_state_dict(lora_weight, "lora_unet", torch.float16)
        pipe.load_lora_weights(lora_state_dict)
        pipe.fuse_lora()
        print("LOADING LORA FINISHED")
    return pipe

def seed_torch(seed=0):
    random.seed(seed)   # Python的随机性
    os.environ['PYTHONHASHSEED'] = str(seed)    # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)   # numpy的随机性
    torch.manual_seed(seed)   # torch的CPU随机性，为CPU设置随机种子
    torch.cuda.manual_seed(seed)   # torch的GPU随机性，为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.   torch的GPU随机性，为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = False   # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = True   # 选择确定性算法


