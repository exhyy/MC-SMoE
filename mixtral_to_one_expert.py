import torch
from transformers import MixtralForCausalLM, MixtralConfig, AutoTokenizer
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
    output_path = '/opt/data/private/hyy/hf_models/Mixtral-8x7B-Instruct-v0.1_merged-group1_cpu_fp32_one_expert'

    config_mixtral = MixtralConfig.from_pretrained(mixtral_path)
    assert isinstance(config_mixtral, MixtralConfig)

    config_one_expert = deepcopy(config_mixtral)
    config_one_expert.num_local_experts = 1
    config_one_expert.num_experts_per_tok = 1

    print('buiding model')
    model_one_expert = MixtralForCausalLM(config_one_expert)
    model_one_expert = model_one_expert.to(torch.bfloat16)

    weight_map = load_weight_map(mixtral_path)
    with torch.no_grad():
        for name, parameter in tqdm(list(model_one_expert.named_parameters()), desc='copy data'):
            # if not name.endswith('.block_sparse_moe.gate.weight'):
            #     print('YES:', name)
            # else:
            #     print('NO: ', name)
            if not name.endswith('.block_sparse_moe.gate.weight'):
                weight = load_weight(mixtral_path, weight_map, name)
                assert weight.dtype == parameter.dtype and weight.device == parameter.device
                parameter.copy_(weight)

    print(f'saving model to {output_path}')
    model_one_expert.save_pretrained(output_path)
    tokenizer = AutoTokenizer.from_pretrained(mixtral_path)
    tokenizer.save_pretrained(output_path)

if __name__ == '__main__':
    main()
