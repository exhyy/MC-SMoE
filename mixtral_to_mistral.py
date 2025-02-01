import torch
from transformers import MixtralForCausalLM, MixtralConfig, MistralForCausalLM, MistralConfig, AutoTokenizer
from safetensors import safe_open

import json
import os
from copy import deepcopy
from typing import Dict
from tqdm import tqdm

def load_weight_map(model_path: str) -> Dict[str, str]:
    json_path = os.path.join(model_path, 'model.safetensors.index.json')
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['weight_map']

def load_weight(model_path: str, weight_map: Dict[str, str], weight_name: str) -> torch.Tensor:
    filename = weight_map[weight_name]
    path = os.path.join(model_path, filename)
    with safe_open(path, framework='pt', device='cpu') as f:
        return f.get_tensor(weight_name)

def main():
    mixtral_path = '/opt/data/private/hyy/hf_models/Mixtral-8x7B-Instruct-v0.1_merged-group1_cpu_fp32'
    output_path = '/opt/data/private/hyy/hf_models/Mixtral-8x7B-Instruct-v0.1_merged-group1_cpu_fp32_mistral'

    config_mixtral = MixtralConfig.from_pretrained(mixtral_path)
    assert isinstance(config_mixtral, MixtralConfig)

    config_mistral = MistralConfig()
    # attr_names = dir(config_mistral)
    attr_names = ['vocab_size', 'max_position_embeddings', 'hidden_size', 'intermediate_size', 'num_hidden_layers', 'num_attention_heads',
                  'sliding_window', 'head_dim', 'num_key_value_heads', 'hidden_act', 'initializer_range', 'rms_norm_eps', 'use_cache',
                  'rope_theta', 'attention_dropout']
    for attr_name in attr_names:
        if hasattr(config_mixtral, attr_name):
            setattr(config_mistral, attr_name, getattr(config_mixtral, attr_name))
    if isinstance(config_mistral.intermediate_size, (list, tuple)):
        config_mistral.intermediate_size = config_mistral.intermediate_size[0]

    print('buiding model')
    model_mistral = MistralForCausalLM(config_mistral)
    model_mistral = model_mistral.to(torch.bfloat16)

    weight_map = load_weight_map(mixtral_path)
    with torch.no_grad():
        for name, parameter in tqdm(model_mistral.named_parameters(), desc='copy data'):
            if name.startswith('model.layers') and 'mlp' in name:
                layer_id = int(name.split('.')[2])
                if 'gate_proj' in name:
                    name_mixtral = f'model.layers.{layer_id}.block_sparse_moe.experts.0.w1.weight'
                if 'up_proj' in name:
                    name_mixtral = f'model.layers.{layer_id}.block_sparse_moe.experts.0.w3.weight'
                if 'down_proj' in name:
                    name_mixtral = f'model.layers.{layer_id}.block_sparse_moe.experts.0.w2.weight'
                weight = load_weight(mixtral_path, weight_map, name_mixtral)
            else:
                weight = load_weight(mixtral_path, weight_map, name)
            assert weight.dtype == parameter.dtype and weight.device == parameter.device
            parameter.copy_(weight)

    print(f'saving model to {output_path}')
    model_mistral.save_pretrained(output_path)
    tokenizer = AutoTokenizer.from_pretrained(mixtral_path)
    tokenizer.save_pretrained(output_path)

if __name__ == '__main__':
    main()
