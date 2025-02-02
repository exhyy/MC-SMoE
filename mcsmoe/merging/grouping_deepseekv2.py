import os
from copy import deepcopy
from typing import Dict, List, Optional

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
# from transformers import MixtralForCausalLM, MixtralConfig
# from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

from mcsmoe.utils.constants import FP32_EPS

SIMILARITY_MAPPING_FUNCTION = {
    "cosine": lambda x, y: (F.cosine_similarity(x, y, dim=-1, eps=FP32_EPS) + 1).item() / 2,
    "mse": lambda x, y: 1 / (1 + 0.1 * torch.log(F.mse_loss(x, y, reduction="sum"))).item(),
}

LEGAL_SIMILARITY_BASES = ["weight", "feature", "feature.abs", "weight-feature", "gradient", "weight-gradient",
                          "router-logits", "router-weight", "router-weight-feature", "mse", "random",
                          "feature-correlation.lsa", "feature-correlation.max"]


class ExpertsGrouperForDeepseekV2(object):
    def __init__(
            self,
            config,
            similarity_fn: str = "cosine",
            similarity_base: str = "router-logits",
    ):
        if similarity_fn not in SIMILARITY_MAPPING_FUNCTION:
            raise ValueError(
                f"[MC-SMoE]similarity_fn should be one of {SIMILARITY_MAPPING_FUNCTION.keys()}, got {similarity_fn} instead."
            )
        if similarity_base not in LEGAL_SIMILARITY_BASES:
            raise ValueError(
                f"[MC-SMoE] similarity_base should be one of {LEGAL_SIMILARITY_BASES}, got {similarity_base} instead.")

        self.num_experts = config.n_routed_experts
        self.d_model = config.hidden_size
        sparse_layer_indices = []
        for layer_idx in range(config.num_hidden_layers):
            if layer_idx >= config.first_k_dense_replace and layer_idx % config.moe_layer_freq == 0:
                sparse_layer_indices.append(layer_idx)
        self.sparse_layer_indices = sparse_layer_indices
        self.similarity_fn = SIMILARITY_MAPPING_FUNCTION[similarity_fn]
        self.similarity_base = similarity_base
        self._group_state_dict = None
        self._similarity_state_dict = None
        self._usage_frequency_state_dict = None
        self.reset_all()

    def reset_all(self):
        if self.similarity_base == "mse":
            self.similarity_fn = SIMILARITY_MAPPING_FUNCTION["mse"]
            print("[MC-SMoE]Set similarity_fn to mse for mse similarity_base.")
        self._group_state_dict = dict()
        self._similarity_state_dict = dict()
        self._usage_frequency_state_dict = dict()
        # Similarity range: [0, 2]
        for layer_idx in self.sparse_layer_indices:
            ffn_name = f"model.layers.{layer_idx}.mlp"
            self._group_state_dict[ffn_name] = torch.arange(self.num_experts, device="cpu")
            self._similarity_state_dict[ffn_name] = torch.zeros(
                (self.num_experts, self.num_experts), device="cpu") + torch.eye(self.num_experts, device="cpu")
            self._usage_frequency_state_dict[ffn_name] = torch.zeros(self.num_experts, device="cpu")

    def similarity_state_dict(self) -> Dict[str, torch.Tensor]:
        return deepcopy(self._similarity_state_dict)

    def group_state_dict(self) -> Dict[str, torch.LongTensor]:
        return deepcopy(self._group_state_dict)

    def usage_frequency_state_dict(self) -> Dict[str, torch.Tensor]:
        return deepcopy(self._usage_frequency_state_dict)

    def save_similarity(self, mlp_name: str, i: int, j: int, similarity: float):
        self._similarity_state_dict[mlp_name][i, j] = similarity
        self._similarity_state_dict[mlp_name][j, i] = similarity

    def get_similarity(self, mlp_name: str, i: int, j: int) -> float:
        return self._similarity_state_dict[mlp_name][i, j].item()

    def get_similarity_matrix(self, mlp_name: str) -> torch.Tensor:
        return deepcopy(self._similarity_state_dict[mlp_name])

    def save_group_state_dict(self, save_dir: str):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self._group_state_dict, os.path.join(save_dir, "group_state_dict.pt"))

    def load_group_state_dict(self, load_dir: str):
        self._group_state_dict = torch.load(os.path.join(load_dir, "group_state_dict.pt"))

    def _assign_num_groups_per_layer(
            self,
            num_average_groups: int,
            merging_layers: List[int],
    ) -> Dict[str, int]:
        num_grouping_layers = len(merging_layers)
        total_num_groups = num_average_groups * num_grouping_layers + self.num_experts * (
                len(self.sparse_layer_indices) - num_grouping_layers
        )
        all_usage_frequency = []
        usage_frequency_dict = deepcopy(self._usage_frequency_state_dict)
        for i, layer_idx in enumerate(self.sparse_layer_indices):
            ffn_name = f"model.layers.{layer_idx}.mlp"

            # 1. Experts in the excluded layers are always not merged.
            if layer_idx not in merging_layers:
                usage_frequency_dict[ffn_name] = torch.ones_like(usage_frequency_dict[ffn_name])

            # 2. Each layer must have at least one group, set the most used expert in a layer to frequency 1.
            max_usage_index = torch.argmax(usage_frequency_dict[ffn_name])
            usage_frequency_dict[ffn_name][max_usage_index] = 1.0

            # 3. Collect all usage frequency.
            all_usage_frequency.append(usage_frequency_dict[ffn_name])

        all_usage_frequency = torch.cat(all_usage_frequency, dim=0)
        sorted_usage_frequency, sorted_indices = torch.sort(all_usage_frequency, descending=True)
        num_groups_per_layer = dict()

        # Note: When threshold is 0.0, the actual number of groups is smaller than total_num_groups.
        if num_average_groups == self.num_experts:
            total_num_groups = total_num_groups - 1
        frequency_threshold = sorted_usage_frequency[total_num_groups]
        print(f"[MC-SMoE] Frequency threshold: {frequency_threshold}")

        if frequency_threshold == 1.0:
            raise ValueError("[MC-SMoE] The number of groups is too large, please reduce the number of groups.")

        for i, layer_idx in enumerate(self.sparse_layer_indices):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            num_groups_per_layer[ffn_name] = torch.sum(
                (usage_frequency_dict[ffn_name] > frequency_threshold).long()
            ).item()

        return num_groups_per_layer

    def group_experts_globally_from_dominant_experts(
            self,
            num_average_groups: int,
            merging_layers: List[int],
    ) -> Dict[str, List[int]]:
        """
        Globally group experts into clusters by routing-guided clustering, each layer will have different number of
         clusters. The total number of clusters is determined by num_average_groups.

        Parameters
        ----------
        num_average_groups: int
            The average number of clusters for all layers.
        merging_layers: List[int]
            The layers of decoder that are excluded from merging.

        Returns
        -------
        core_experts: Dict[str, List[int]]
            The core experts of each cluster
        """

        # 1. Assign num_groups respectively for each layer according to num_average_groups
        num_groups_per_layer = self._assign_num_groups_per_layer(
            num_average_groups, merging_layers
        )
        print(f"[MC-SMoE] Number of groups per layer: {num_groups_per_layer}")

        # 2. Group experts into clusters for each layer
        dom_experts = dict()
        for layer_idx in tqdm(
                self.sparse_layer_indices,
                desc=f"[MC-SMoE] Globally routing-guided grouping experts into average {num_average_groups} clusters"
        ):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            num_groups = num_groups_per_layer[ffn_name]
            indices_sorted_by_usage = torch.argsort(self._usage_frequency_state_dict[ffn_name], descending=True)

            # 1 Assign top-K most-used experts with label 0 to K-1 respectively
            core_expert_indices = indices_sorted_by_usage[:num_groups]
            dom_experts[ffn_name] = core_expert_indices.tolist()
            for i in range(num_groups):
                self._group_state_dict[ffn_name][indices_sorted_by_usage[i]] = i

            # 2 Assign left unassigned experts to the cluster with the most similar core
            similarity_matrix = self.get_similarity_matrix(ffn_name)
            for i in range(num_groups, self.num_experts):
                # Find the most similar core
                most_similar_core = core_expert_indices[
                    torch.argmax(similarity_matrix[indices_sorted_by_usage[i], core_expert_indices])
                ]
                most_similar_group_label = self._group_state_dict[ffn_name][most_similar_core]
                self._group_state_dict[ffn_name][indices_sorted_by_usage[i]] = most_similar_group_label

        return dom_experts

    def compute_all_usages(
            self,
            model,
            dataloader: DataLoader,
    ):
        model.eval()
        config = model.config
        for batch in tqdm(dataloader, desc=f"[MC-SMoE] Evaluating routing distribution"):
            batch = {k: v.cuda() for k, v in batch.items()}
            if "labels" in batch:
                # We don't need to compute loss here, so remove the labels
                batch.pop("labels")
            with torch.no_grad():
                outputs = model(**batch, output_router_logits=True)
            all_router_logits = outputs.router_logits
            assert isinstance(all_router_logits, (tuple, list))
            all_router_logits = tuple(filter(lambda x: x is not None, all_router_logits))
            all_router_logits = torch.stack(all_router_logits)  # of shape (len(self.sparse_layer_indices), num_tokens, num_experts)
            assert all_router_logits.shape[0] == len(self.sparse_layer_indices)
            selected_experts = torch.topk(all_router_logits, 2, dim=-1)[1].reshape(
                len(self.sparse_layer_indices), -1
            )  # of shape (len(self.sparse_layer_indices), num_tokens * 2)

            logit_indices = list(range(len(self.sparse_layer_indices)))
            for layer_idx, logit_idx in zip(self.sparse_layer_indices, logit_indices):
                ffn_name = f"model.layers.{layer_idx}.mlp"
                unique, counts = torch.unique(selected_experts[logit_idx], return_counts=True)
                self._usage_frequency_state_dict[ffn_name][unique.cpu()] += counts.cpu()
        self._usage_frequency_state_dict = {
            k: v / torch.sum(v) for k, v in self._usage_frequency_state_dict.items()
        }

    def compute_all_similarities(
            self,
            model,
            dataloader: DataLoader = None
    ):
        if self.similarity_base not in ["weight", "router-weight"] and dataloader is None:
            raise ValueError(
                "[MC-SMoE] `dataloader` should be provided when similarity_base is not 'weight' or 'router-weight'")

        model = model.eval()
        if self.similarity_base == "weight":
            self._compute_all_similarities_by_weight(model.state_dict())
        elif self.similarity_base == 'router-weight':
            self._compute_all_similarities_by_router_weight(model.state_dict())
        elif self.similarity_base == 'router-logits':
            self._compute_all_similarities_by_router_logits(model, dataloader)
        else:
            raise NotImplementedError

    def _compute_all_similarities_by_weight(self, state_dict: Dict[str, torch.Tensor]):
        for layer_idx in tqdm(self.sparse_layer_indices, desc="[MC-SMoE]  Computing similarities by weight..."):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            for i in range(self.num_experts):
                for j in range(i + 1, self.num_experts):
                    i_flat = torch.cat(
                        [state_dict[f"{ffn_name}.experts.{i}.gate_proj.weight"].flatten(),
                         state_dict[f"{ffn_name}.experts.{i}.down_proj.weight"].flatten(),
                         state_dict[f"{ffn_name}.experts.{i}.up_proj.weight"].flatten()],
                        dim=0
                    )
                    j_flat = torch.cat(
                        [state_dict[f"{ffn_name}.experts.{j}.gate_proj.weight"].flatten(),
                         state_dict[f"{ffn_name}.experts.{j}.down_proj.weight"].flatten(),
                         state_dict[f"{ffn_name}.experts.{j}.up_proj.weight"].flatten()],
                        dim=0
                    )
                    similarity = self.similarity_fn(i_flat, j_flat)
                    self.save_similarity(ffn_name, i, j, similarity)

    def _compute_all_similarities_by_router_weight(
            self, state_dict: Dict[str, torch.Tensor]
    ):
        for layer_idx in tqdm(self.sparse_layer_indices, desc="[MC-SMoE] Computing similarities by router rows..."):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            for i in range(self.num_experts):
                for j in range(i + 1, self.num_experts):
                    i_flat = state_dict[f"{ffn_name}.gate.weight"][i]
                    j_flat = state_dict[f"{ffn_name}.gate.weight"][j]
                    similarity = self.similarity_fn(i_flat, j_flat)
                    self.save_similarity(ffn_name, i, j, similarity)

    def _compute_all_similarities_by_router_logits(
            self, model, dataloader: DataLoader
    ):
        model.eval()
        all_router_logits = []
        for batch in tqdm(dataloader, desc=f"[MC-SMoE] Running inference to get routing logits"):
            batch = {k: v.cuda() for k, v in batch.items()}
            if "labels" in batch:
                # We don't need to compute loss here, so remove the labels
                batch.pop("labels")
            with torch.no_grad():
                outputs = model(**batch, output_router_logits=True)
            batch_router_logits = outputs.router_logits
            assert isinstance(batch_router_logits, (tuple, list))
            batch_router_logits = tuple(filter(lambda x: x is not None, batch_router_logits))
            batch_router_logits = torch.stack(batch_router_logits)  # (num_hidden_layers, num_tokens, num_experts)
            assert batch_router_logits.shape[0] == len(self.sparse_layer_indices)
            all_router_logits.append(batch_router_logits)

        all_router_logits = torch.cat(all_router_logits, dim=1)  # (num_hidden_layers, *, num_experts)
        logit_indices = list(range(len(self.sparse_layer_indices)))
        for layer_idx, logit_idx in tqdm(zip(self.sparse_layer_indices, logit_indices), total=len(self.sparse_layer_indices), desc="[MC-SMoE] Computing similarities by router logits..."):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            layer_router_logits = all_router_logits[logit_idx].reshape(-1, self.num_experts)
            with torch.no_grad():
                for i in range(self.num_experts):
                    for j in range(i + 1, self.num_experts):
                        i_flat = layer_router_logits[:, i].flatten()
                        j_flat = layer_router_logits[:, j].flatten()
                        similarity = self.similarity_fn(i_flat, j_flat)
                        self.save_similarity(ffn_name, i, j, similarity)


@torch.no_grad()
def _merge_mlp_experts_by_usage_frequency_weighting(
        ffn,
        group_labels: torch.LongTensor,
        usage_frequencies: torch.Tensor,
        shared: bool = True,
):
    for label in group_labels.unique():
        expert_indices = torch.where(group_labels == label)[0]
        gate_proj_weight_list = torch.stack(
            [ffn.experts[expert_idx].gate_proj.weight * usage_frequencies[expert_idx]
             for expert_idx in expert_indices], dim=0
        )
        down_proj_weight_list = torch.stack(
            [ffn.experts[expert_idx].down_proj.weight * usage_frequencies[expert_idx]
             for expert_idx in expert_indices], dim=0
        )
        up_proj_weight_list = torch.stack(
            [ffn.experts[expert_idx].up_proj.weight * usage_frequencies[expert_idx]
             for expert_idx in expert_indices], dim=0
        )
        gate_proj_weight = torch.sum(gate_proj_weight_list, dim=0) / (torch.sum(usage_frequencies[expert_indices], dim=0) + FP32_EPS)
        down_proj_weight = torch.sum(down_proj_weight_list, dim=0) / (torch.sum(usage_frequencies[expert_indices], dim=0) + FP32_EPS)
        up_proj_weight = torch.sum(up_proj_weight_list, dim=0) / (torch.sum(usage_frequencies[expert_indices], dim=0) + FP32_EPS)

        ffn.experts[expert_indices[0]].gate_proj.weight.copy_(gate_proj_weight)
        ffn.experts[expert_indices[0]].down_proj.weight.copy_(down_proj_weight)
        ffn.experts[expert_indices[0]].up_proj.weight.copy_(up_proj_weight)

        for expert_idx in expert_indices[1:]:
            if shared:
                # Binding merged experts to the first of them
                ffn.experts[expert_idx] = ffn.experts[expert_indices[0]]
            else:
                ffn.experts[expert_idx].gate_proj.weight.copy_(gate_proj_weight)
                ffn.experts[expert_idx].down_proj.weight.copy_(down_proj_weight)
                ffn.experts[expert_idx].up_proj.weight.copy_(up_proj_weight)

    return ffn


@torch.no_grad()
def merge_by_groups_with_usage_weighted(
        model,
        grouper: ExpertsGrouperForDeepseekV2,
        merging_layers: Optional[List[int]] = None,
):
    """
    Parameters
    ----------
    model: MixtralForCausalLM
        The model to merge experts.
    grouper: ExpertsGrouperForSwitch
        The grouper to group experts, supposed to have been called `grouper.compute_all_usages()` and
            one of `grouper.group_experts()` (i.e. have grouped labels).
    merging_layers: Optional[List[int]]
        The layers where we merge experts, if None, merge all layers.
    """
    usage_frequency_dict = grouper.usage_frequency_state_dict()
    group_labels_dict = grouper.group_state_dict()

    for layer_idx in tqdm(
            grouper.sparse_layer_indices,
            desc=f"[MC-SMoE] Merging experts with usage-frequency-weighted averaging..."
    ):
        if merging_layers is not None and layer_idx not in merging_layers:
            continue
        ffn_name = f"model.layers.{layer_idx}.mlp"
        group_labels = group_labels_dict[ffn_name]
        usage_frequencies = usage_frequency_dict[ffn_name]
        model.model.layers[layer_idx].mlp = _merge_mlp_experts_by_usage_frequency_weighting(
            ffn=model.model.layers[layer_idx].mlp,
            group_labels=group_labels,
            usage_frequencies=usage_frequencies,
            shared=False,
        )
    return model
