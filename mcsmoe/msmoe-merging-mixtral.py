# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2024/2/18
import os
from typing import Optional

import torch
from fire import Fire
from transformers import MixtralForCausalLM, AutoTokenizer

import sys
sys.path.extend(['.', '..'])
from mcsmoe.evaluation import get_minipile_dataloder, evaluate_minipile_perplexity, evaluate_fewshot
from mcsmoe.merging.grouping_mixtral import ExpertsGrouperForMixtral, merge_by_groups_with_usage_weighted


MIXTRAL_MODEL_PATH = '/opt/data/private/hyy/hf_models/Mixtral-8x7B-Instruct-v0.1'

def evaluate_mcsmoe(
        task: str,
        num_average_groups: int,
        num_fewshot: Optional[int] = 5,
        eval_batch_size: Optional[int] = 32,
        output_path: Optional[str] = None,
        save_path: Optional[str] = None,
        shared_experts: Optional[bool] = True,
        cache_dir: Optional[str] = None,
):
    print(f'>>>>>>>>>>>>> {shared_experts=}')
    eval_ppl = (task == "minipile")
    tokenizer = AutoTokenizer.from_pretrained(MIXTRAL_MODEL_PATH)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = MixtralForCausalLM.from_pretrained(
        MIXTRAL_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        # device_map="auto"
    )

    dataloader_for_merging = get_minipile_dataloder(
        tokenizer=tokenizer,
        block_size=512,
        batch_size=1,
        subset_ratio=0.1,
    )

    # MC-SMoE!
    print(f"[MC-SMoE] Merging into average {num_average_groups} groups...")

    grouper = ExpertsGrouperForMixtral(config=model.config, similarity_base="router-logits")
    if cache_dir is not None:
        similarity_loaded = grouper.load_similarity_state_dict(cache_dir)
        if similarity_loaded:
            print(f"[MC-SMoE] Similarity state dict loaded, skipping computation")
        else:
            print(f"[MC-SMoE] Similarity state dict NOT loaded, start computing...")
            grouper.compute_all_similarities(model, dataloader_for_merging)
            grouper.save_similarity_state_dict(cache_dir)
    
        usage_loaded = grouper.load_usage_frequency_state_dict(cache_dir)
        if usage_loaded:
            print(f"[MC-SMoE] Usage state dict loaded, skipping computation")
        else:
            print(f"[MC-SMoE] Usage state dict NOT loaded, start computing...")
            grouper.compute_all_usages(model, dataloader_for_merging)
            grouper.save_usage_frequency_state_dict(cache_dir)
    else:
        grouper.compute_all_similarities(model, dataloader_for_merging)
        grouper.compute_all_usages(model, dataloader_for_merging)
    dom_experts = grouper.group_experts_globally_from_dominant_experts(
        num_average_groups=num_average_groups, merging_layers=list(range(0, model.config.num_hidden_layers))
    )

    model = merge_by_groups_with_usage_weighted(
        model, grouper=grouper, merging_layers=list(range(0, model.config.num_hidden_layers)), shared_experts=shared_experts
    )

    print(f"[MC-SMoE] ========= Grouping results ========= ")
    for name, state in grouper.group_state_dict().items():
        print(f"Group {name}: {state.tolist()} (DOMs are {dom_experts[name]})")

    print("[MC-SMoE] Number of parameters after merging:", model.num_parameters())

    if save_path:
        print(f"[MC-SMoE] Saving merged model to {save_path}")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

    if eval_ppl:
        evaluate_minipile_perplexity(
            model, tokenizer=tokenizer, batch_size=eval_batch_size, log=True
        )
    else:
        evaluate_fewshot(
            model, tokenizer=tokenizer, task=task, num_fewshot=num_fewshot, output_path=output_path, log=True
        )


if __name__ == "__main__":
    Fire(evaluate_mcsmoe)
