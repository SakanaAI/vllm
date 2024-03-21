from typing import Optional, List

import torch
from torch import nn

from vllm.config import LoRAConfig
from vllm.model_executor.layers.linear import LinearMethodBase
from transformers import PretrainedConfig

from .llama import (
    KVCache,
    InputMetadata,
    LlamaModel,
    ParallelLMHead,
    DEFAULT_VOCAB_PADDING_SIZE,
    Sampler,
    SamplingMetadata,
    SamplerOutput,
    hf_model_weights_iterator,
    default_weight_loader,
)

class EvoMistralModel(LlamaModel):
    def __init__(
        self,
        config: PretrainedConfig,
        linear_method: Optional[LinearMethodBase] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__(config, linear_method, lora_config)
        self.input_scales = nn.Parameter(
            data=torch.zeros(self.config.num_hops).float(), requires_grad=False
        )
        self.input_layers = nn.Parameter(
            data=torch.zeros(self.config.num_hops).int(), requires_grad=False
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for idx, layer_ix in enumerate(self.input_layers):
            layer = self.layers[layer_ix]
            scale = self.input_scales[idx].to(hidden_states.device)
            hidden_states, residual = layer(
                positions,
                hidden_states * scale,
                kv_caches[idx],
                input_metadata,
                residual,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states
    

# copied from llama.LlamaForCausalLM but replaced LlamaModel to EvoMistralModel
class EvoMistralForCausalLM(nn.Module):
    supports_lora = True

    def __init__(
        self,
        config: PretrainedConfig,
        linear_method: Optional[LinearMethodBase] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.linear_method = linear_method
        self.model = EvoMistralModel(config, linear_method, lora_config=lora_config)
        unpadded_vocab_size = config.vocab_size
        if lora_config:
            unpadded_vocab_size += lora_config.lora_extra_vocab_size
        self.lm_head = ParallelLMHead(
            unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE
            # We need bigger padding if using lora for kernel
            # compatibility
            if not lora_config else lora_config.lora_vocab_padding_size,
        )
        self.sampler = Sampler(unpadded_vocab_size, config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   input_metadata)
        return hidden_states

    def sample(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(self.lm_head.weight, hidden_states,
                                   sampling_metadata)
        return next_tokens

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision):
            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
