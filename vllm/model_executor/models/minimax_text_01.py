# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only MiniMaxText01 model."""

from collections.abc import Iterable
from itertools import islice
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    pass

import regex as re
import torch
import torch.distributed
from torch import nn
from transformers import MiniMaxConfig

from vllm.attention import Attention, AttentionMetadata
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.distributed.parallel_state import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.linear_attn import MiniMaxText01LinearAttention
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import maybe_prefix
from vllm.sequence import IntermediateTensors

from .interfaces import HasInnerState, IsHybrid
from .utils import PPMissingLayer, is_pp_missing_parameter, make_layers


import pprint
import numpy as np

def print_tensor_stats(tensor, name):
    """Prints statistics of a PyTorch tensor, mimicking the FD format."""
    # Only print on rank 0 to avoid log spam
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return
        
    if tensor is None:
        print(f"DEBUG_vLLM: {name} is None")
        return
    with torch.no_grad():
        stats = {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        # ========== [修改这里] ==========
        num_elements = tensor.numel()
        if num_elements > 0:
            tensor_cpu_float = tensor.detach().cpu().to(torch.float32)
            stats["max"] = f"{torch.max(tensor_cpu_float).item():.6f}"
            stats["min"] = f"{torch.min(tensor_cpu_float).item():.6f}"
            stats["mean"] = f"{torch.mean(tensor_cpu_float).item():.6f}"

            # 只有当元素数量大于1时才计算std
            if num_elements > 1:
                stats["std"] = f"{torch.std(tensor_cpu_float).item():.6f}"
            else:
                stats["std"] = "0.000000" # 单个元素的std为0
            # ========== [结束修改] ==========
            flat_data = tensor_cpu_float.flatten().numpy()[:5]
            stats["first_5_values"] = flat_data
        
        # Use pprint to format the dictionary nicely
        print(f"\n--- [vLLM DEBUG] {name} ---\n{pprint.pformat(stats, indent=2)}\n----------------------------\n")

def replace_weight_name(
    name: str, key: str = None, to: str = None, count: int = None, prefix: str = None
) -> str:
    name = name.replace(key, to) if count is None else name.replace(key, to, count)
    return name


def weight_loader_with_alias(alias: str):
    def wrapper(func: callable):
        def inner_func(
            param: torch.Tensor,
            loaded_weight: torch.Tensor,
            *args,
            prefix: str = None,
            **kwargs,
        ):
            value = func(param, loaded_weight, *args, **kwargs)
            return value

        return inner_func

    return wrapper


class MiniMaxText01MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        layer_idx: int = None,
        prefix: str = "mlp",
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class MiniMaxText01MoE(nn.Module):
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: Optional[torch.dtype] = None,
        layer_idx: int = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "moe",
    ) -> None:
        super().__init__()

        self.layer_idx = layer_idx
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_total_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size // self.tp_size
        self.quant_config = quant_config

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype

        self.gate = ReplicatedLinear(
            self.hidden_size,
            self.num_total_experts,
            bias=False,
            params_dtype=torch.float32,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )
        self.gate.weight.weight_loader = MiniMaxText01MoE.gate_weight_loader

        self.experts = FusedMoE(
            num_experts=self.num_total_experts,
            top_k=self.top_k,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size * self.tp_size,
            params_dtype=self.params_dtype,
            reduce_results=True,
            renormalize=True,
            quant_config=self.quant_config,
            tp_size=self.tp_size,
            prefix=f"{prefix}.experts",
        )
        return

    @staticmethod
    def gate_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        assert param.size() == loaded_weight.size()
        param.data.copy_(loaded_weight.to(torch.float32))
        return

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        router_logits_fp32, _ = self.gate(hidden_states.to(torch.float32))
        final_hidden_states = self.experts(
            hidden_states, router_logits_fp32.to(hidden_states.dtype)
        )
        final_hidden = final_hidden_states.view(num_tokens, hidden_size)
        return final_hidden


class MiniMaxText01Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        num_kv_heads: int,
        rotary_dim: int,
        max_position: int = 4096 * 32,
        rope_theta: float = 10000,
        sliding_window: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
        layer_idx: int = None,
        cache_config: Optional[CacheConfig] = None,
        prefix: str = "mha",
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.sliding_window = sliding_window
        self.prefix = prefix

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=rotary_dim,
            max_position=max_position,
            base=int(rope_theta),
            is_neox_style=True,
            dtype=torch.float32,
        )
        return

    # def forward(
    #     self,
    #     hidden_states: torch.Tensor,
    #     output: torch.Tensor,
    #     positions: torch.Tensor,
    #     **kwargs,
    # ) -> None:
    #     # ========== [开始新的 GQA 权重 Dump 代码] ==========
    #     # 只 dump 一次
    #     if not hasattr(self, '_gqa_weight_dumped'):
            
    #         # 只在 rank 0 上执行保存操作
    #         if get_tensor_model_parallel_rank() == 0:
    #             print(f"\n--- [vLLM DEBUG] Dumping GQA L{self.layer_idx} QKV weight for rank 0... ---\n")

    #             # 获取最终加载到GPU上的权重参数
    #             weight_shard = self.qkv_proj.weight 

    #             # 保存到文件
    #             torch.save(weight_shard.cpu().float(), f"vllm_gqa_l{self.layer_idx}_qkv_weight_shard_rank0.pt")
    #             print(f"\n--- [vLLM DEBUG] Saved rank 0 weight shard to vllm_gqa_l{self.layer_idx}_qkv_weight_shard_rank0.pt ---\n")

    #         # 在所有 rank 上都设置 flag
    #         type(self)._gqa_weight_dumped = True
    #     # ========== [结束 GQA 权重 Dump 代码] ==========
        
    #     qkv, _ = self.qkv_proj(hidden_states)
    #     q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    #     print_tensor_stats(q, f"VLLM_L7_GQA_Q_BeforeRoPE")
    #     print_tensor_stats(k, f"VLLM_L7_GQA_K_BeforeRoPE")
    #     q, k = self.rotary_emb(positions, q, k)
    #     print_tensor_stats(q, f"VLLM_L7_GQA_Q_AfterRoPE")
    #     print_tensor_stats(k, f"VLLM_L7_GQA_K_AfterRoPE")
    #     attn_output = self.attn(q, k, v)
    #     print_tensor_stats(attn_output, f"VLLM_L7_GQA_Attention_Output")
    #     output[:], _ = self.o_proj(attn_output)
    
    # def forward(
    #     self,
    #     hidden_states: torch.Tensor,
    #     output: torch.Tensor,
    #     positions: torch.Tensor,
    #     **kwargs,
    # ) -> None:
    #     is_target_layer = (self.layer_idx == 7)

    #     qkv, _ = self.qkv_proj(hidden_states)
        
    #     if is_target_layer:
    #         print_tensor_stats(qkv, f"VLLM_L{self.layer_idx}_After_QKV_Proj_Combined")

    #     q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        
    #     if is_target_layer:
    #         print_tensor_stats(q, f"VLLM_L{self.layer_idx}_Q_BeforeRoPE")
    #         print_tensor_stats(k, f"VLLM_L{self.layer_idx}_K_BeforeRoPE")
    #         print_tensor_stats(v, f"VLLM_L{self.layer_idx}_V_Tensor")

    #     q, k = self.rotary_emb(positions, q, k)
        
    #     if is_target_layer:
    #         print_tensor_stats(q, f"VLLM_L{self.layer_idx}_Q_AfterRoPE")
    #         print_tensor_stats(k, f"VLLM_L{self.layer_idx}_K_AfterRoPE")
        
    #     # Note: The raw attention output is not easily accessible here as it's inside the 'self.attn' call
    #     # self.attn is a vllm custom op which is a black box from this perspective.
    #     # The final output after o_proj is what we can compare.
    #     attn_output = self.attn(q, k, v)

    #     if is_target_layer:
    #         print_tensor_stats(attn_output, f"VLLM_L{self.layer_idx}_Attention_Kernel_Output")
        
    #     output[:], _ = self.o_proj(attn_output)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
        positions: torch.Tensor,
        **kwargs,
    ) -> None:
        
        qkv, _ = self.qkv_proj(hidden_states)
        
        print_tensor_stats(qkv, f"VLLM_L{self.layer_idx}_After_QKV_Proj_Combined")

        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        
        print_tensor_stats(q, f"VLLM_L{self.layer_idx}_Q_BeforeRoPE")
        print_tensor_stats(k, f"VLLM_L{self.layer_idx}_K_BeforeRoPE")
        print_tensor_stats(v, f"VLLM_L{self.layer_idx}_V_Tensor")

        q, k = self.rotary_emb(positions, q, k)
        
        print_tensor_stats(q, f"VLLM_L{self.layer_idx}_Q_AfterRoPE")
        print_tensor_stats(k, f"VLLM_L{self.layer_idx}_K_AfterRoPE")
        
        attn_output = self.attn(q, k, v)

        print_tensor_stats(attn_output, f"VLLM_L{self.layer_idx}_Attention_Kernel_Output")
        
        # o_proj's output is the final output of the attention block
        output[:], _ = self.o_proj(attn_output)




class MiniMaxText01DecoderLayer(nn.Module):
    def __init__(
        self,
        config: MiniMaxConfig,
        model_config: Optional[ModelConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        expert_num: int = 1,
        layer_id: int = None,
        linear_layer_id: Optional[int] = None,
        prefix: str = "decoder",
    ) -> None:
        self._ilayer = layer_id
        self._irank = get_tensor_model_parallel_rank()
        self.prefix = prefix
        super().__init__()

        self.hidden_size = config.hidden_size
        self.expert_num = expert_num

        rope_theta = getattr(config, "rope_theta", 10000)

        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = config.hidden_size // config.num_attention_heads
        if hasattr(config, "max_model_len") and isinstance(config.max_model_len, int):
            max_position_embeddings = min(
                config.max_position_embeddings, config.max_model_len
            )
        if config.attention_type == 0:
            use_headxdim = True
            hidden_inner = (
                head_dim * config.num_attention_heads
                if use_headxdim
                else config.hidden_size
            )
            self.self_attn = MiniMaxText01LinearAttention(
                hidden_size=self.hidden_size,
                hidden_inner_size=hidden_inner,
                num_heads=config.num_attention_heads,
                head_dim=head_dim,
                max_position=max_position_embeddings,
                block_size=config.block if hasattr(config, "block") else 256,
                num_hidden_layer=config.num_hidden_layers,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                layer_idx=self._ilayer,
                linear_layer_idx=linear_layer_id,
                prefix=prefix,
            )
        elif config.attention_type == 1:
            self.self_attn = MiniMaxText01Attention(
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                head_dim=head_dim,
                rotary_dim=config.rotary_dim
                if hasattr(config, "rotary_dim")
                else head_dim,
                num_kv_heads=config.num_key_value_heads,
                max_position=max_position_embeddings,
                rope_theta=rope_theta,
                sliding_window=config.sliding_window,
                quant_config=quant_config,
                layer_idx=self._ilayer,
                cache_config=cache_config,
                prefix=prefix,
            )
        else:
            raise ValueError(
                f"Unsupported attention type: {self.config.attention_type}"
            )

        if expert_num == 1:
            self.mlp = MiniMaxText01MLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                quant_config=quant_config,
                layer_idx=self._ilayer,
                prefix=prefix,
            )
        else:
            self.block_sparse_moe = MiniMaxText01MoE(
                num_experts=expert_num,
                top_k=config.num_experts_per_tok,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                layer_idx=self._ilayer,
                quant_config=quant_config,
                prefix=prefix,
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        if config.attention_type == 0:
            self.layernorm_attention_alpha = getattr(
                config,
                "layernorm_linear_attention_alpha",
                getattr(config, "linear_attn_alpha_factor", 1),
            )
            self.layernorm_attention_beta = getattr(
                config,
                "layernorm_linear_attention_beta",
                getattr(config, "linear_attn_beta_factor", 1),
            )
        else:
            self.layernorm_attention_alpha = getattr(
                config,
                "layernorm_full_attention_alpha",
                getattr(config, "full_attn_alpha_factor", 1),
            )
            self.layernorm_attention_beta = getattr(
                config,
                "layernorm_full_attention_beta",
                getattr(config, "full_attn_beta_factor", 1),
            )
        self.layernorm_mlp_alpha = getattr(
            config, "layernorm_mlp_alpha", getattr(config, "mlp_alpha_factor", 1)
        )
        self.layernorm_mlp_beta = getattr(
            config, "layernorm_mlp_beta", getattr(config, "mlp_beta_factor", 1)
        )
        self.postnorm = getattr(config, "postnorm", False)
        self.shared_moe = False

        shared_intermediate = getattr(config, "shared_intermediate_size", 0)
        if isinstance(shared_intermediate, list):
            shared_intermediate = (
                shared_intermediate[layer_id]
                if layer_id < len(shared_intermediate)
                else 0
            )
        if shared_intermediate > 0:
            self.shared_moe = True
            self.shared_mlp = MiniMaxText01MLP(
                hidden_size=self.hidden_size,
                intermediate_size=shared_intermediate,
                quant_config=quant_config,
                layer_idx=self._ilayer,
                prefix=prefix,
            )
            self.coefficient = ReplicatedLinear(
                self.hidden_size,
                1,
                bias=False,
                quant_config=quant_config,
                params_dtype=torch.float32,
            )
            self.coefficient.weight.weight_loader = self.shared_moe_coefficient_loader
            self.shared_moe_mode = getattr(config, "shared_moe_mode", "softmax")
        return

    # def forward(
    #     self,
    #     hidden_states: torch.Tensor,
    #     positions: torch.Tensor,
    #     attn_metadata: AttentionMetadata,
    #     residual: Optional[torch.Tensor],
    #     is_warmup: bool = False,
    #     **kwargs,
    # ) -> tuple[torch.Tensor, torch.Tensor]:
    #     # --- Start of FD Aligned Debug Prints ---
    #     layer_id = self._ilayer
    #     print(f"\n{'='*20} [vLLM DEBUG] Entering DecoderLayer {layer_id} {'='*20}")
    #     print_tensor_stats(hidden_states, f"L{layer_id}:0a_Input_HiddenStates")
    #     print_tensor_stats(residual, f"L{layer_id}:0b_Input_Residual")
    #     # --- End of Debug Prints ---

    #     layernorm_input = hidden_states
    #     layernorm_output = self.input_layernorm(layernorm_input)
        
    #     # --- Debug Print ---
    #     print_tensor_stats(layernorm_output, f"L{layer_id}:1_After_InputLayernorm")
    #     # --- End of Debug Prints ---

    #     residual_attn = layernorm_output if self.postnorm else layernorm_input
        
    #     # self.self_attn is a custom op that calls MiniMaxText01LinearAttention._forward
    #     # So, the prints for LinearAttention will be inside that class.
    #     self_attention_output = torch.empty_like(layernorm_output)
    #     self.self_attn(
    #         hidden_states=layernorm_output,
    #         output=self_attention_output,
    #         positions=positions,
    #     )
        
    #     # --- Debug Print ---
    #     print_tensor_stats(self_attention_output, f"L{layer_id}:2_After_Attention")
    #     # --- End of Debug Prints ---
        
    #     hidden_states_after_attn = (residual_attn * self.layernorm_attention_alpha) + (self_attention_output * self.layernorm_attention_beta)
        
    #     # --- Debug Print ---
    #     print_tensor_stats(hidden_states_after_attn, f"L{layer_id}:3_After_Attn_Residual(alpha={self.layernorm_attention_alpha}, beta={self.layernorm_attention_beta})")
    #     # --- End of Debug Prints ---

    #     layernorm_input = hidden_states_after_attn
    #     layernorm_output_mlp = self.post_attention_layernorm(layernorm_input)
        
    #     # --- Debug Print ---
    #     print_tensor_stats(layernorm_output_mlp, f"L{layer_id}:4_After_PostAttnLayernorm")
    #     # --- End of Debug Prints ---
        
    #     residual_mlp = layernorm_output_mlp if self.postnorm else layernorm_input

    #     if self.expert_num == 1:
    #         mlp_output = self.mlp(layernorm_output_mlp)
    #     else:
    #         moe_layernorm_output = layernorm_output_mlp.clone()
    #         mlp_output = self.block_sparse_moe(moe_layernorm_output)
    #         if self.shared_moe:
    #             # This part is complex, let's add prints inside if needed later
    #             before_moe_dtype = layernorm_output_mlp.dtype
    #             moe_hidden_fp32 = mlp_output.to(torch.float32)
    #             output_mlp_shared = self.shared_mlp(layernorm_output_mlp).to(torch.float32)

    #             coef, _ = self.coefficient(layernorm_output_mlp.to(torch.float32))

    #             if self.shared_moe_mode == "softmax":
    #                 coef = torch.nn.functional.softmax(coef, dim=-1)
    #                 mlp_output = moe_hidden_fp32 * (1 - coef) + output_mlp_shared * coef
    #             elif self.shared_moe_mode == "sigmoid":
    #                 coef = torch.nn.functional.sigmoid(coef)
    #                 mlp_output = moe_hidden_fp32 * (1 - coef) + output_mlp_shared * coef

    #             mlp_output = mlp_output.to(before_moe_dtype)
        
    #     # --- Debug Print ---
    #     print_tensor_stats(mlp_output, f"L{layer_id}:5a_After_MoE_MLP")
    #     if self.shared_moe:
    #          print_tensor_stats(mlp_output, f"L{layer_id}:5b_After_Shared_MLP_Merge")
    #     # --- End of Debug Prints ---

    #     final_output = (residual_mlp * self.layernorm_mlp_alpha) + (mlp_output * self.layernorm_mlp_beta)
        
    #     # --- Debug Print ---
    #     print_tensor_stats(final_output, f"L{layer_id}:6_FinalOutput(alpha={self.layernorm_mlp_alpha}, beta={self.layernorm_mlp_beta})")
    #     print(f"{'='*20} [vLLM DEBUG] Exiting DecoderLayer {layer_id} {'='*20}\n")
    #     # --- End of Debug Prints ---

    #     return final_output, None
    
    
    # def forward(
    #     self,
    #     hidden_states: torch.Tensor,
    #     positions: torch.Tensor,
    #     attn_metadata: AttentionMetadata,
    #     residual: Optional[torch.Tensor],
    #     is_warmup: bool = False,
    #     **kwargs,
    # ) -> tuple[torch.Tensor, torch.Tensor]:
    #     layer_id = self._ilayer
    #     # We only focus on GQA Layer 7
    #     is_target_layer = (layer_id == 7)

    #     if is_target_layer:
    #         print(f"\n{'='*20} [vLLM GQA DEBUG] Entering DecoderLayer {layer_id} {'='*20}")
    #         print_tensor_stats(hidden_states, f"VLLM_L{layer_id}:0a_Input_HiddenStates")

    #     layernorm_input = hidden_states
    #     layernorm_output = self.input_layernorm(layernorm_input)
        
    #     if is_target_layer:
    #         print_tensor_stats(layernorm_output, f"VLLM_L{layer_id}:1_After_InputLayernorm")

    #     residual_attn = layernorm_output if self.postnorm else layernorm_input
        
    #     self_attention_output = torch.empty_like(layernorm_output)
    #     # self.self_attn (MiniMaxText01Attention.forward) will handle its internal prints
    #     self.self_attn(
    #         hidden_states=layernorm_output,
    #         output=self_attention_output,
    #         positions=positions,
    #     )
        
    #     if is_target_layer:
    #         print_tensor_stats(self_attention_output, f"VLLM_L{layer_id}:2_After_Attention")
        
    #     hidden_states_after_attn = (residual_attn * self.layernorm_attention_alpha) + (self_attention_output * self.layernorm_attention_beta)
        
    #     if is_target_layer:
    #         print_tensor_stats(hidden_states_after_attn, f"VLLM_L{layer_id}:3_After_Attn_Residual")

    #     layernorm_input = hidden_states_after_attn
    #     layernorm_output_mlp = self.post_attention_layernorm(layernorm_input)
        
    #     # --- Debug Print ---
    #     print_tensor_stats(layernorm_output_mlp, f"L{layer_id}:4_After_PostAttnLayernorm")
    #     # --- End of Debug Prints ---
        
    #     residual_mlp = layernorm_output_mlp if self.postnorm else layernorm_input

    #     if self.expert_num == 1:
    #         mlp_output = self.mlp(layernorm_output_mlp)
    #     else:
    #         moe_layernorm_output = layernorm_output_mlp.clone()
    #         mlp_output = self.block_sparse_moe(moe_layernorm_output)
    #         if self.shared_moe:
    #             # This part is complex, let's add prints inside if needed later
    #             before_moe_dtype = layernorm_output_mlp.dtype
    #             moe_hidden_fp32 = mlp_output.to(torch.float32)
    #             output_mlp_shared = self.shared_mlp(layernorm_output_mlp).to(torch.float32)

    #             coef, _ = self.coefficient(layernorm_output_mlp.to(torch.float32))

    #             if self.shared_moe_mode == "softmax":
    #                 coef = torch.nn.functional.softmax(coef, dim=-1)
    #                 mlp_output = moe_hidden_fp32 * (1 - coef) + output_mlp_shared * coef
    #             elif self.shared_moe_mode == "sigmoid":
    #                 coef = torch.nn.functional.sigmoid(coef)
    #                 mlp_output = moe_hidden_fp32 * (1 - coef) + output_mlp_shared * coef

    #             mlp_output = mlp_output.to(before_moe_dtype)
        
    #     # --- Debug Print ---
    #     print_tensor_stats(mlp_output, f"L{layer_id}:5a_After_MoE_MLP")
    #     if self.shared_moe:
    #          print_tensor_stats(mlp_output, f"L{layer_id}:5b_After_Shared_MLP_Merge")
    #     # --- End of Debug Prints ---

    #     final_output = (residual_mlp * self.layernorm_mlp_alpha) + (mlp_output * self.layernorm_mlp_beta)
        
    #     # --- Debug Print ---
    #     print_tensor_stats(final_output, f"L{layer_id}:6_FinalOutput(alpha={self.layernorm_mlp_alpha}, beta={self.layernorm_mlp_beta})")
    #     print(f"{'='*20} [vLLM DEBUG] Exiting DecoderLayer {layer_id} {'='*20}\n")
    #     # --- End of Debug Prints ---

    #     return final_output, None
    
    # in work/vllm/vllm/model_executor/models/minimax_text_01.py

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        is_warmup: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layer_id = self._ilayer
        
        attn_type_str = "GQA" if self.self_attn.__class__.__name__ == 'MiniMaxText01Attention' else "LinearAttn"
        
        print(f"\n{'='*20} [vLLM DEBUG] Entering DecoderLayer {layer_id} ({attn_type_str}) {'='*20}")
        print_tensor_stats(hidden_states, f"VLLM_L{layer_id}:0a_Input_HiddenStates")

        layernorm_input = hidden_states
        layernorm_output = self.input_layernorm(layernorm_input)
        
        print_tensor_stats(layernorm_output, f"VLLM_L{layer_id}:1_After_InputLayernorm")

        residual_attn = layernorm_output if self.postnorm else layernorm_input
        
        self_attention_output = torch.empty_like(layernorm_output)
        # self.self_attn (Attention/LinearAttention) will handle its internal prints
        self.self_attn(
            hidden_states=layernorm_output,
            output=self_attention_output,
            positions=positions,
        )
        
        print_tensor_stats(self_attention_output, f"VLLM_L{layer_id}:2_After_Attention")
        
        hidden_states_after_attn = (residual_attn * self.layernorm_attention_alpha) + (self_attention_output * self.layernorm_attention_beta)
        
        print_tensor_stats(hidden_states_after_attn, f"VLLM_L{layer_id}:3_After_Attn_Residual")

        # --- MLP Block ---
        layernorm_input = hidden_states_after_attn
        layernorm_output_mlp = self.post_attention_layernorm(layernorm_input)
        
        print_tensor_stats(layernorm_output_mlp, f"VLLM_L{layer_id}:4_After_PostAttnLayernorm")

        residual_mlp = layernorm_output_mlp if self.postnorm else layernorm_input

        if self.expert_num == 1:
            mlp_output = self.mlp(layernorm_output_mlp)
        else: # MoE logic
            moe_layernorm_output = layernorm_output_mlp.clone()
            mlp_output = self.block_sparse_moe(moe_layernorm_output)
            if self.shared_moe:
                before_moe_dtype = layernorm_output_mlp.dtype
                moe_hidden_fp32 = mlp_output.to(torch.float32)
                output_mlp_shared = self.shared_mlp(layernorm_output_mlp).to(torch.float32)
                coef, _ = self.coefficient(layernorm_output_mlp.to(torch.float32))
                if self.shared_moe_mode == "sigmoid":
                    coef = torch.nn.functional.sigmoid(coef)
                else: # softmax
                    coef = torch.nn.functional.softmax(coef, dim=-1)
                mlp_output = moe_hidden_fp32 * (1 - coef) + output_mlp_shared * coef
                mlp_output = mlp_output.to(before_moe_dtype)
        
        print_tensor_stats(mlp_output, f"VLLM_L{layer_id}:5a_After_MoE_MLP")
        if self.shared_moe:
             print_tensor_stats(mlp_output, f"VLLM_L{layer_id}:5b_After_Shared_MLP_Merge")

        # --- Final Residual Connection ---
        final_output = (residual_mlp * self.layernorm_mlp_alpha) + (mlp_output * self.layernorm_mlp_beta)
        
        print_tensor_stats(final_output, f"VLLM_L{layer_id}:6_FinalOutput")
        print(f"{'='*20} [vLLM DEBUG] Exiting DecoderLayer {layer_id} ({attn_type_str}) {'='*20}\n")

        return final_output, None
    
    
    @staticmethod
    def shared_moe_coefficient_loader(
        param: torch.Tensor, loaded_weight: torch.Tensor
    ) -> None:
        assert param.size() == loaded_weight.size()

        param.data.copy_(loaded_weight.to(torch.float32))
        return


@support_torch_compile
class MiniMaxText01Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: MiniMaxConfig = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        quant_config = vllm_config.quant_config
        cache_config = vllm_config.cache_config
        scheduler_config = vllm_config.scheduler_config

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.decoder_attention_types = getattr(
            config, "attn_type_list", False
        ) or getattr(config, "decoder_attention_types", False)
        # The HF format uses "layer_types" instead of "attn_type_list"
        # where "linear_attention" is 0 and "full_attention" is 1
        if not self.decoder_attention_types and hasattr(config, "layer_types"):
            self.decoder_attention_types = []
            for layer_type in config.layer_types:
                if layer_type == "linear_attention":
                    self.decoder_attention_types.append(0)
                elif layer_type == "full_attention":
                    self.decoder_attention_types.append(1)
                else:
                    raise ValueError(f"Unsupported layer type: {layer_type}")
        # Default to full attention
        if not self.decoder_attention_types:
            self.decoder_attention_types = [1] * config.num_hidden_layers
        self.num_layers = config.num_hidden_layers

        self._layer_barrier = False
        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
                org_num_embeddings=self.vocab_size,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        def layer_fn(prefix):
            layer_idx = int(prefix.split(".")[-1])
            layer_config = config
            layer_config.attention_type = self.decoder_attention_types[layer_idx]
            layer_config.layer_idx = layer_idx

            decoder_kwargs = {
                "quant_config": quant_config,
                "layer_id": layer_idx,
                "model_config": model_config,
                "cache_config": cache_config,
            }

            if layer_config.attention_type == 0:
                decoder_kwargs["linear_layer_id"] = sum(
                    1 for i in range(layer_idx) if self.decoder_attention_types[i] == 0
                )
            else:
                decoder_kwargs["linear_layer_id"] = None

            if hasattr(config, "num_local_experts") and isinstance(
                config.num_local_experts, list
            ):
                decoder_kwargs["expert_num"] = config.num_local_experts[layer_idx]
            elif hasattr(config, "num_local_experts") and isinstance(
                config.num_local_experts, int
            ):
                decoder_kwargs["expert_num"] = config.num_local_experts
            else:
                decoder_kwargs["expert_num"] = 1

            return MiniMaxText01DecoderLayer(
                layer_config, **decoder_kwargs, prefix=prefix
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, layer_fn, prefix=f"{prefix}.layers"
        )

        linear_layer_nums = sum(
            1
            for i in range(config.num_hidden_layers)
            if self.decoder_attention_types[i] == 0
        )
        max_slots_number = scheduler_config.max_num_seqs
        self.cache_shape = (
            linear_layer_nums,
            max_slots_number,
            config.num_attention_heads // get_tensor_model_parallel_world_size(),
            config.head_dim,
            config.head_dim,
        )
        _dummy = torch.zeros(1)
        self._dtype = _dummy.dtype
        del _dummy

        norm_kwargs = {}
        if hasattr(config, "rms_norm_eps"):
            norm_kwargs["eps"] = config.rms_norm_eps
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, **norm_kwargs)
        else:
            self.norm = PPMissingLayer()
        self.embed_scale = 1.0
        return

    def _clear_prefill_cache(
        self, attn_metadata, minimax_cache_tensors: torch.Tensor, **kwargs
    ):
        seq_to_slot_maps = {}
        seq_id_map = sum(list(kwargs["request_ids_to_seq_ids"].values()), [])
        for _, seq_to_slot_map in self.minimax_cache.cache_indices_mapping.items():
            seq_to_slot_maps.update(seq_to_slot_map)

        slots_to_clear = []
        for _prefill_id in range(getattr(attn_metadata, "num_prefills", 0)):
            if _prefill_id >= len(seq_id_map):
                break
            seq_id = seq_id_map[_prefill_id]
            if (
                attn_metadata.context_lens_tensor[_prefill_id] == 0
                and seq_id in seq_to_slot_maps
            ):
                slots_to_clear.append(seq_to_slot_maps[seq_id])

        if slots_to_clear:
            slots_tensor = torch.tensor(
                slots_to_clear, device=minimax_cache_tensors.device, dtype=torch.long
            )
            minimax_cache_tensors[:, slots_tensor, ...] = 0

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        # forward_context = get_forward_context()
        # ==================== 区分 Profile 和 正式推理 ====================
        forward_context = get_forward_context()
        is_profile_run = forward_context.attn_metadata is None
        run_mode = "[PROFILE]" if is_profile_run else "[INFERENCE]"
        
        if get_tensor_model_parallel_rank() == 0:
            print(f"\n{'#'*20} vLLM RUN MODE: {run_mode} {'#'*20}\n")
        # =================================================================
        attn_metadata = forward_context.attn_metadata

        if get_pp_group().is_first_rank:
            if inputs_embeds is None:
                # --- Debug Print ---
                print_tensor_stats(input_ids, "TOP:0_InputIDs")
                # --- End of Debug Prints ---
                hidden_states = self.embed_scale * self.embed_tokens(input_ids)
            else:
                hidden_states = inputs_embeds
            
            # --- Debug Print ---
            print_tensor_stats(hidden_states, "TOP:1_AfterEmbedding")
            # --- End of Debug Prints ---
            
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(
                hidden_states=hidden_states,
                positions=positions,
                attn_metadata=attn_metadata,
                residual=residual,
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        
        # The final norm happens after the loop.
        # FD does it before the final output, vLLM does it here.
        if residual is not None:
            # This path is unlikely for this model architecture
            hidden_states_normed, _ = self.norm(hidden_states, residual)
        else:
            hidden_states_normed = self.norm(hidden_states)
        
        # --- Debug Print ---
        # Note: FD calls this "TOP:3_FinalOutput", which corresponds to the output
        # of the final norm before the lm_head.
        print_tensor_stats(hidden_states_normed, "TOP:3_FinalOutput")
        # --- End of Debug Prints ---

        return hidden_states_normed


class MiniMaxText01ForCausalLM(nn.Module, HasInnerState, IsHybrid):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        lora_config = vllm_config.lora_config
        self.config = config
        self.lora_config = lora_config

        if not hasattr(config, "sliding_window"):
            config.sliding_window = None

        self.CONCAT_FFN = True

        self.unpadded_vocab_size = self.config.vocab_size
        if hasattr(vllm_config.model_config, "max_model_len"):
            self.config.max_model_len = vllm_config.model_config.max_model_len
        self.model = MiniMaxText01Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                self.unpadded_vocab_size,
                self.config.hidden_size,
                org_num_embeddings=self.config.vocab_size,
                padding_size=DEFAULT_VOCAB_PADDING_SIZE,
                prefix=maybe_prefix(prefix, "lm_head"),
            )

            self.logits_processor = LogitsProcessor(
                self.unpadded_vocab_size, self.config.vocab_size
            )

        else:
            self.lm_head = PPMissingLayer()
        self.lm_head.float()
        flash_layer_count = sum(
            1 for attn_type in self.model.decoder_attention_types if attn_type == 1
        )
        self.kv_cache = [torch.tensor([]) for _ in range(flash_layer_count)]
        return

    def copy_inputs_before_cuda_graphs(self, input_buffers, **kwargs):
        return self.model.minimax_cache.copy_inputs_before_cuda_graphs(
            input_buffers, **kwargs
        )

    def get_seqlen_agnostic_capture_inputs(self, batch_size: int):
        return self.model.minimax_cache.get_seqlen_agnostic_capture_inputs(batch_size)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
        )

        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states.float())

        return logits

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors:
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    (batch_size, self.config.hidden_size), dtype=dtype, device=device
                ),
                "residual": torch.zeros(
                    (batch_size, self.config.hidden_size), dtype=dtype, device=device
                ),
            }
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        def which_layer(name: str) -> int:
            if "layers" in name:
                after_layer = name.split("layers")[-1]
                return int(after_layer.split(".")[1])
            return None

        def is_linear_attn_layer(layer_idx: int) -> bool:
            if layer_idx is None or layer_idx >= len(
                self.model.decoder_attention_types
            ):
                return False
            return self.model.decoder_attention_types[layer_idx] == 0

        def is_moe_weight(name: str) -> bool:
            return "block_sparse_moe" in name and not name.endswith(".bias")

        def get_expert_id(param_name):
            pattern = r"model\.layers\.\d+\.block_sparse_moe\.experts\.(\d+)\."
            match = re.search(pattern, param_name)
            if match:
                return match.group(1)
            return None

        def load_sparse_moe_weight(
            name: str, loaded_weight: torch.Tensor, self
        ) -> None:
            if isinstance(self.config.num_local_experts, list):
                expert_params_mapping = [
                    (
                        "w13_weight" if weight_name in ["w1", "w3"] else "w2_weight",
                        f"experts.{expert_id}.{weight_name}.weight",
                        expert_id,
                    )
                    for expert_id in range(max(self.config.num_local_experts))
                    for weight_name in ["w1", "w2", "w3"]
                ]
            else:
                expert_params_mapping = [
                    (
                        "w13_scale" if weight_name in ["w1", "w3"] else "w2_scale",
                        f"{expert_id}.{weight_name}.weight_scale",
                        expert_id,
                        weight_name,
                    )
                    for expert_id in range(self.config.num_local_experts)
                    for weight_name in ["w1", "w2", "w3"]
                ] + [
                    (
                        "w13_weight" if weight_name in ["w1", "w3"] else "w2_weight",
                        f"{expert_id}.{weight_name}.weight",
                        expert_id,
                        weight_name,
                    )
                    for expert_id in range(self.config.num_local_experts)
                    for weight_name in ["w1", "w2", "w3"]
                ]
            for param_name, weight_name, expert_id, shard_id in expert_params_mapping:
                name_expert_id = get_expert_id(name)
                if name_expert_id is not None and int(name_expert_id) != int(expert_id):
                    continue
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if is_pp_missing_parameter(name, self):
                    return
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader = weight_loader_with_alias(name)(weight_loader)
                weight_loader(
                    param,
                    loaded_weight,
                    weight_name,
                    expert_id=expert_id,
                    shard_id=shard_id,
                )
                loaded_params.add(name)
                break
            else:
                if is_pp_missing_parameter(name, self):
                    return
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader = weight_loader_with_alias(name)(weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)
            return

        def is_shared_mlp_weight(name: str) -> bool:
            return "shared_mlp" in name and not name.endswith(".bias")

        def load_shared_mlp_weight(
            name: str, loaded_weight: torch.Tensor, self
        ) -> None:
            if not self.CONCAT_FFN:
                if "gate_proj" in name:
                    name = name.replace("gate_proj", "w1", 1)
                elif "up_proj" in name:
                    name = name.replace("up_proj", "w3", 1)
                elif "down_proj" in name:
                    name = name.replace("down_proj", "w2", 1)
            else:
                if "gate_proj" in name:
                    name = name.replace("gate_proj", "gate_up_proj", 1)
                    loaded_shard_id = 0
                elif "up_proj" in name:
                    name = name.replace("up_proj", "gate_up_proj", 1)
                    loaded_shard_id = 1
            if is_pp_missing_parameter(name, self):
                return
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader = weight_loader_with_alias(name)(weight_loader)
            if not self.CONCAT_FFN:
                weight_loader(param, loaded_weight)
            else:
                if "gate_up_proj" in name:
                    weight_loader(param, loaded_weight, loaded_shard_id)
                elif "down_proj" in name:
                    weight_loader(param, loaded_weight)
                else:
                    raise AssertionError("MLP weight not in [gate_up_proj, down_proj]")
            loaded_params.add(name)
            return

        def is_mha_weight(name: str) -> bool:
            return "self_attn" in name and not name.endswith(".bias")

        def load_linear_attn_weight(
            name: str, loaded_weight: torch.Tensor, self
        ) -> None:
            if is_pp_missing_parameter(name, self):
                return
            param = params_dict[name]

            weight_loader = getattr(
                param, "weight_loader", MiniMaxText01LinearAttention.weight_direct_load
            )
            weight_loader = weight_loader_with_alias(name)(weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
            return

        def load_flash_attn_weight(
            name: str, loaded_weight: torch.Tensor, self
        ) -> None:
            flash_mha_params_mapping = [
                ("qkv_proj", "q_proj", "q"),
                ("qkv_proj", "k_proj", "k"),
                ("qkv_proj", "v_proj", "v"),
                ("gate_up_proj", "gate_proj", 0),
                ("gate_up_proj", "up_proj", 1),
            ]
            for param_name, weight_name, shard_id in flash_mha_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if is_pp_missing_parameter(name, self):
                    return
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader = weight_loader_with_alias(name)(weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                if is_pp_missing_parameter(name, self):
                    return
                param = params_dict[name]

                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader = weight_loader_with_alias(name)(weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)
            return

        def is_layer_norm_weight(name: str) -> bool:
            return "norm" in name and not name.endswith(".bias") and name in params_dict

        def load_layer_norm_weight(
            name: str, loaded_weight: torch.Tensor, self
        ) -> None:
            if is_pp_missing_parameter(name, self):
                return
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader = weight_loader_with_alias(name)(weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
            return

        def load_basic_weight(name: str, loaded_weight: torch.Tensor, self) -> None:
            if is_pp_missing_parameter(name, self):
                return
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader = weight_loader_with_alias(name)(weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
            return

        for name, loaded_weight in weights:
            if "layers.0.self_attn.qkv_proj.weight" in name:
                tp_rank = get_tensor_model_parallel_rank()
                if tp_rank == 0:
                    print("\n" + "="*20 + " VLLM RAW WEIGHT DEBUG " + "="*20)
                    
                    # 1. 打印原始加载的全量权重
                    full_weight_flat = loaded_weight.detach().cpu().to(torch.float32).flatten().numpy()
                    print("VLLM RAW FULL weight, first 20:", full_weight_flat[:20])
                    
                    # 2. 找到 rank=0 对应的参数，并打印最终加载到参数里的值
                    # 注意: `self.model` 指向 MiniMaxText01ForCausalLM 实例
                    # param_rank0 = self.model.model.layers[0].self_attn.qkv_proj.weight
                    param_rank0 = self.model.layers[0].self_attn.qkv_proj.weight
                    final_shard_flat = param_rank0.detach().cpu().to(torch.float32).flatten().numpy()
                    print("VLLM FINAL SHARD (rank 0), first 20:", final_shard_flat[:20])
                    
                    print("="*20 + " END VLLM DEBUG " + "="*20 + "\n")
            
            
            if "layers.0.self_attn.qkv_proj.weight" in name:
                print("!!! VLLM RAW LOADED QKV WEIGHT !!!")
                print("Shape:", loaded_weight.shape)
                
                # --- 核心修复 ---
                # 先将 bfloat16 转换为 float32，然后再进行后续操作
                loaded_weight_float32 = loaded_weight.to(torch.float32)
                # --- 结束修复 ---

                # 现在对 float32 版本的 tensor 进行操作
                print("First 5 values:", loaded_weight_float32.flatten().cpu().numpy()[:5])
                print("Min:", loaded_weight_float32.min().item())
                print("Mean:", loaded_weight_float32.mean().item())
                print("Std:", loaded_weight_float32.std().item()) # 最好也加上 std
            weight_at_layer = which_layer(name)
            if weight_at_layer and weight_at_layer >= len(
                self.model.decoder_attention_types
            ):
                continue

            if is_layer_norm_weight(name):
                load_layer_norm_weight(name, loaded_weight, self)
                continue
            if is_mha_weight(name):
                if is_linear_attn_layer(weight_at_layer):
                    load_linear_attn_weight(name, loaded_weight, self)
                else:
                    load_flash_attn_weight(name, loaded_weight, self)
                continue
            if is_moe_weight(name):
                load_sparse_moe_weight(name, loaded_weight, self)
                continue
            if is_shared_mlp_weight(name):
                load_shared_mlp_weight(name, loaded_weight, self)
                continue

            if "rotary_emb.inv_freq" in name:
                continue

            load_basic_weight(name, loaded_weight, self)
        return loaded_params

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.linear_attention_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[tuple[int, ...], ...]:
        """Calculate shape for MiniMaxText01LinearAttention cache.

        Args:
            vllm_config: vLLM config

        Returns:
            Tuple containing:
            - state_shape: Shape of the cache
        """
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_config

        return MambaStateShapeCalculator.linear_attention_state_shape(
            num_heads=hf_config.num_attention_heads,
            tp_size=parallel_config.tensor_parallel_size,
            head_dim=hf_config.head_dim,
        )
