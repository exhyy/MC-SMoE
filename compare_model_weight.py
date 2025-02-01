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
    model_1_path = '/opt/data/private/hyy/hf_models/Mixtral-8x7B-Instruct-v0.1'
    # model_2_path = '/opt/data/private/hyy/hf_models/Mixtral-8x7B-Instruct-v0.1_merged-group4'
    model_2_path = '/opt/data/private/hyy/hf_models/Mixtral-8x7B-Instruct-v0.1_merged-group8_cpu_fp32_fix'

    weight_map_1 = load_weight_map(model_1_path)
    weight_map_2 = load_weight_map(model_2_path)
    assert set(weight_map_1.keys()) == set(weight_map_2.keys())

    for key in weight_map_1.keys():
        weight_1 = load_weight(model_1_path, weight_map_1, key)
        weight_2 = load_weight(model_2_path, weight_map_2, key)
        max_diff = (weight_1 - weight_2).abs().max().item()
        print(f'{key}: {max_diff}')

if __name__ == '__main__':
    main()
