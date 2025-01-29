# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2024/2/18
import sys
sys.path.extend(['.', '..'])

import os
import pickle
import functools
from typing import Optional
from collections import OrderedDict
from copy import deepcopy
from types import MethodType

import torch
from fire import Fire
from transformers import MixtralForCausalLM, AutoTokenizer, MixtralConfig
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock, MixtralDecoderLayer, MixtralBlockSparseTop2MLP
from accelerate import infer_auto_device_map, dispatch_model

from mcsmoe.evaluation import get_minipile_dataloder, evaluate_minipile_perplexity, evaluate_fewshot
from mcsmoe.merging.grouping_mixtral import ExpertsGrouperForMixtral, merge_by_groups_with_usage_weighted


MIXTRAL_MODEL_PATH = '/opt/data/private/hyy/hf_models/Mixtral-8x7B-Instruct-v0.1'


def save_ordered_dict(ordered_dict, save_path):
    """
    将 OrderedDict 对象保存为 pickle 文件
    
    参数：
    ordered_dict : OrderedDict - 要保存的有序字典对象
    save_path : str - 文件保存路径（建议以 .pkl 结尾）
    
    异常：
    TypeError - 当输入不是 OrderedDict 时
    IOError - 当文件写入失败时
    """
    # 参数类型校验
    if not isinstance(ordered_dict, OrderedDict):
        raise TypeError("输入对象必须是 OrderedDict 类型")
    
    try:
        # 自动创建目录（如果不存在）
        dir_path = os.path.dirname(save_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        # 写入pickle文件
        with open(save_path, 'wb') as f:
            pickle.dump(ordered_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            
    except (PermissionError, IOError) as e:
        raise IOError(f"文件写入失败: {str(e)}") from e
    except Exception as e:
        raise RuntimeError(f"未知错误: {str(e)}") from e

def dispatch_mixtral(model: MixtralForCausalLM):
    num_gpus = torch.cuda.device_count()
    ffn_count = 0
    for module in model.modules():
        if isinstance(module, MixtralSparseMoeBlock):
            target_device = torch.device(f'cuda:{ffn_count % num_gpus}')
            module.to(target_device)

            if not hasattr(module, '_original_forward'):
                module._original_forward = module.forward

            def forward_with_device(self, hidden_states, device):
                hidden_states = hidden_states.to(device)
                final_hidden_states, router_logits = self._original_forward(hidden_states)
                return final_hidden_states.to('cpu'), router_logits.to('cpu')

            new_forward = functools.partial(forward_with_device, device=target_device)
            module.forward = MethodType(new_forward, module)
            ffn_count += 1

    return model

def restore_model(model):
    for module in model.modules():
        if hasattr(module, '_original_forward'):
            module.forward = module._original_forward
            del module._original_forward
    return model

def get_save_model(model: MixtralForCausalLM):
    for layer in model.model.layers:
        assert isinstance(layer, MixtralDecoderLayer)
        experts = layer.block_sparse_moe.experts

        expert_map = dict()
        for expert_id, expert in enumerate(experts):
            assert isinstance(expert, MixtralBlockSparseTop2MLP)
            if expert_id not in expert_map:
                expert_map[expert_id] = set()

            # find shared experts
            for expert_id_shared in list(expert_map.keys()):
                if expert is experts[expert_id_shared]:
                    expert_map[expert_id_shared].add(expert_id)

def get_save_model_one(model: MixtralForCausalLM):
    config = deepcopy(model.config)
    assert isinstance(config, MixtralConfig)
    config.num_local_experts = 1
    config.num_experts_per_tok = 1
    one_expert_model = MixtralForCausalLM(config)
    one_expert_model = one_expert_model.to(torch.bfloat16)

    state_dict = model.state_dict()
    for name, param in model.named_parameters():
        name_splits = name.split('.')
        if not (len(name_splits) == 6 and name_splits[4] == 'gate'):
            target_param = state_dict[name].cpu()
            assert param.dtype == target_param.dtype, f'Different dtype for parameter {name}: {param.dtype} != {target_param.dtype}'
            param.data = target_param.data

        # if len(name_splits) == 8 and name_splits[3] == 'block_sparse_moe':
        #     target_param = state_dict[name].cpu()
        #     assert param.dtype == target_param.dtype, f'Different dtype for parameter {name}: {param.dtype} != {target_param.dtype}'
        #     param.data = target_param.data
        # elif len(name_splits) == 6 and name_splits[4] == 'gate':
        #     param.data = param.data
        # else:
        #     target_param = state_dict[name].cpu()
        #     assert param.dtype == target_param.dtype, f'Different dtype for parameter {name}: {param.dtype} != {target_param.dtype}'
        #     param.data = target_param.data
    return one_expert_model

def evaluate_mcsmoe(
        task: str,
        num_average_groups: int,
        num_fewshot: Optional[int] = 5,
        eval_batch_size: Optional[int] = 32,
        output_path: Optional[str] = None,
        save_path: Optional[str] = None,
):
    eval_ppl = (task == "minipile")
    # tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
    tokenizer = AutoTokenizer.from_pretrained(MIXTRAL_MODEL_PATH)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = MixtralForCausalLM.from_pretrained(
        MIXTRAL_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        # device_map="auto"
    )
    # max_memory = {0: '24GB', 1: '24GB', 2: '24GB', 3: '24GB', 'cpu': '80GB'}
    # device_map = infer_auto_device_map(
    #     model,
    #     max_memory=max_memory,
    #     no_split_module_classes=['MixtralDecoderLayer'],
    # )
    # # device_map['model.embed_tokens'] = 'cpu'
    # device_map['model.layers.31'] = 0
    # device_map['model.layers.0'] = 'cpu'
    # print(f'{device_map=}')
    # model = dispatch_model(model, device_map)
    model = dispatch_mixtral(model)

    dataloader_for_merging = get_minipile_dataloder(
        tokenizer=tokenizer,
        block_size=512,
        batch_size=1,
        subset_ratio=0.1,
    )

    # MC-SMoE!
    print(f"[MC-SMoE] Merging into average {num_average_groups} groups...")

    grouper = ExpertsGrouperForMixtral(config=model.config, similarity_base="router-logits")
    grouper.compute_all_similarities(model, dataloader_for_merging)
    grouper.compute_all_usages(model, dataloader_for_merging)
    dom_experts = grouper.group_experts_globally_from_dominant_experts(
        num_average_groups=num_average_groups, merging_layers=list(range(0, model.config.num_hidden_layers))
    )

    model = merge_by_groups_with_usage_weighted(
        model, grouper=grouper, merging_layers=list(range(0, model.config.num_hidden_layers))
    )

    print(f"[MC-SMoE] ========= Grouping results ========= ")
    for name, state in grouper.group_state_dict().items():
        print(f"Group {name}: {state.tolist()} (DOMs are {dom_experts[name]})")

    print("[MC-SMoE] Number of parameters after merging:", model.num_parameters())

    model = restore_model(model)
    model = model.to('cpu')

    prompt = 'The meaning of life is'
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids
    outputs = model.generate(input_ids, max_new_tokens=128, do_sample=False)[0]
    text = tokenizer.decode(outputs)
    print(f"[MC-SMoE] Sample output of merged model:\n{text}")

    if save_path:
        print(f"[MC-SMoE] Saving merged model to {save_path}")
        # model.save_pretrained(save_path, safe_serialization=False)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

    # if eval_ppl:
    #     evaluate_minipile_perplexity(
    #         model, tokenizer=tokenizer, batch_size=eval_batch_size, log=True
    #     )
    # else:
    #     evaluate_fewshot(
    #         model, tokenizer=tokenizer, task=task, num_fewshot=num_fewshot, output_path=output_path, log=True
    #     )


if __name__ == "__main__":
    Fire(evaluate_mcsmoe)
