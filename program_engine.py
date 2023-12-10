
import torch
import torch.nn as nn
import torch.nn.functional as F

import networks
from sqa import SequentialQueryAttention
from transformer_layer import TransformerLayer


class InstructionGenerator(nn.Module):
    def __init__(self, embed_dim, depth, num_type_instruction, num_fc_layer):
        super().__init__()
        self.depth = depth
        self.num_type_instruction = num_type_instruction
        self.mlpencoder = networks.MLP(embed_dim, num_type_instruction*depth, embed_dim, num_fc_layer, norm="none", activ="relu")
        self.temperature = 1.0
    
    def forward(self, sentence_feats):
        instructions = self.mlpencoder(sentence_feats.detach()) # [B, 6]
        instructions = instructions.view(-1, self.depth, self.num_type_instruction)
        instructions = F.gumbel_softmax(instructions, self.temperature, hard=False)
        instructions = instructions.view(-1, self.depth, self.num_type_instruction, 1, 1)
        return instructions


class ProgramEngine(nn.Module):
    def __init__(self, embed_dim, engine_depth, num_type_instruction, num_instruction_generator_fc_layer, num_parameter_generator_fc_layer, module_depth, heads, dim_head):
        super(ProgramEngine, self).__init__()
        self.query_attention = SequentialQueryAttention(num_type_instruction, embed_dim)
        self.instruction_generator = InstructionGenerator(embed_dim, engine_depth, num_type_instruction, num_instruction_generator_fc_layer)
        self.out_proj = networks.MLP(embed_dim, embed_dim, embed_dim, 2, norm="none", activ="relu")
        self.T = 1.0
        self.engine_depth = engine_depth
        self.num_type_instruction = num_type_instruction

        executors = nn.ModuleDict()
        for layer in range(engine_depth):
            executors[f"layer{layer}"] = nn.ModuleDict()
            for instruction_type in range(num_type_instruction):
                module = TransformerLayer(dim=embed_dim, depth=module_depth, heads=heads, dim_head=dim_head, mlp_dim=4*embed_dim, dropout=0.1)
                params_generator = networks.MLP(embed_dim, self.get_num_params(module), 4*embed_dim, num_parameter_generator_fc_layer, norm='none', activ='relu')
                executors[f"layer{layer}"][f"instruction_type{instruction_type}"] = nn.ModuleDict({
                    "module": module,
                    "params_generator": params_generator,
                })
        self.executors = executors
        
    def forward(self, r, m, word_feats, word_mask, concepts, concept_mask):
        instructions = self.instruction_generator(m)
        if word_mask.dim() == 3:
            word_mask = word_mask.squeeze(dim=2)
        assert word_mask.dim() == 2
        att_entity_feats, _ = self.query_attention(m, word_feats, word_mask) 

        for layer in range(self.engine_depth):
            if layer == 0:
                inputs = torch.cat((r, concepts), dim=1)
                inputs_mask = torch.ones((r.size(0), r.size(1))).cuda()
                inputs_mask = torch.cat((inputs_mask, concept_mask), dim=1)
            else:
                inputs = outputs
            outputs = None
            for instruction_type in range(self.num_type_instruction):
                att_entity_feat = att_entity_feats[:, instruction_type, :]
                params = self.executors[f"layer{layer}"][f"instruction_type{instruction_type}"]["params_generator"](att_entity_feat)
                self.assign_params(params, self.executors[f"layer{layer}"][f"instruction_type{instruction_type}"]["module"])
                if outputs is not None:
                    outputs += self.executors[f"layer{layer}"][f"instruction_type{instruction_type}"]["module"](inputs, inputs_mask) * instructions[:, layer, instruction_type]
                else:
                    outputs = self.executors[f"layer{layer}"][f"instruction_type{instruction_type}"]["module"](inputs, inputs_mask) * instructions[:, layer, instruction_type]

        output = self.out_proj(outputs[:, 1:r.size(1), :].mean(dim=1))
        return output

    def assign_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
          if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            mean = adain_params[:, :m.num_features]
            std = adain_params[:, m.num_features:2*m.num_features]
            m.bias = mean.contiguous().view(-1)
            m.weight = std.contiguous().view(-1)
            if adain_params.size(1) > 2*m.num_features:
              adain_params = adain_params[:, 2*m.num_features:]

    def get_num_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
          if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            num_adain_params += 2*m.num_features
        return num_adain_params
