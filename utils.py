import nni
import torch
import torch.nn as nn
from einops import rearrange


def merge_from_nni(args):
    params = nni.get_next_parameter()

    if "engine_depth" in params.keys():
        args.engine_depth = params["engine_depth"]
        print(f"NNI: set engine_depth to {args.engine_depth}")
    if "num_type_instruction" in params.keys():
        args.num_type_instruction = params["num_type_instruction"]
        print(f"NNI: set num_type_instruction to {args.num_type_instruction}")
    if "num_instruction_generator_fc_layer" in params.keys():
        args.num_instruction_generator_fc_layer = params["num_instruction_generator_fc_layer"]
        print(f"NNI: set num_instruction_generator_fc_layer to {args.num_instruction_generator_fc_layer}")
    if "num_parameter_generator_fc_layer" in params.keys():
        args.num_parameter_generator_fc_layer = params["num_parameter_generator_fc_layer"]
        print(f"NNI: set num_parameter_generator_fc_layer to {args.num_parameter_generator_fc_layer}")
    if "module_depth" in params.keys():
        args.module_depth = params["module_depth"]
        print(f"NNI: set module_depth to {args.module_depth}")
    return args


def params_require_grad(module, update):
    for idx, param in enumerate(module.parameters()):
        param.requires_grad = update


def l2norm(x):
    """L2-normalize each row of x"""
    norm = torch.pow(x, 2).sum(dim=-1, keepdim=True).sqrt()
    return torch.div(x, norm)


class SimpleModule(nn.Module):

    def __init__(self, input_dim, embed_dim):
        super(SimpleModule, self).__init__()
        self.proj = nn.Conv2d(input_dim, embed_dim, kernel_size=1)
    
    def forward(self, x):
        x = self.proj(x.squeeze(dim=1))
        x = rearrange(x, "B D H W -> B (H W) D")
        return x
