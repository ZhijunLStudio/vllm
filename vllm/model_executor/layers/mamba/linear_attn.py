# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionBackend

from typing import TYPE_CHECKING

import torch
import torch.distributed
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, ModelConfig, get_current_vllm_config
from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.lightning_attn import (
    lightning_attention,
    linear_decode_forward_triton,
)
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.utils import direct_register_custom_op
from vllm.v1.attention.backends.linear_attn import LinearAttentionMetadata

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionBackend

import torch
import torch.distributed

import pprint
import numpy as np

def print_tensor_stats(tensor, name):
    """Prints statistics of a PyTorch tensor, mimicking the FD format."""
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return
    # ... (完整的 print_tensor_stats 函数实现，和你之前用的一样)
    if tensor is None:
        print(f"DEBUG_vLLM_INTERNAL: {name} is None")
        return
    with torch.no_grad():
        stats = {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        if tensor.numel() > 0:
            tensor_cpu_float = tensor.detach().cpu().to(torch.float32)
            
            has_nan = torch.any(torch.isnan(tensor_cpu_float)).item()
            has_inf = torch.any(torch.isinf(tensor_cpu_float)).item()
            stats["has_nan"] = has_nan
            stats["has_inf"] = has_inf

            if not has_nan and not has_inf:
                stats["max"] = f"{torch.max(tensor_cpu_float).item():.6f}"
                stats["min"] = f"{torch.min(tensor_cpu_float).item():.6f}"
                stats["mean"] = f"{torch.mean(tensor_cpu_float).item():.6f}"
                stats["std"] = f"{torch.std(tensor_cpu_float).item():.6f}"
            else:
                stats["max"] = "NaN/Inf Present"
                stats["min"] = "NaN/Inf Present"
                stats["mean"] = "NaN/Inf Present"
                stats["std"] = "NaN/Inf Present"

            flat_data = tensor_cpu_float.flatten().numpy()[:5]
            stats["first_5_values"] = flat_data
        
        print(f"\n--- [vLLM INTERNAL DEBUG] {name} ---\n{pprint.pformat(stats, indent=2)}\n--------------------------\n")
# --- 结束新增 ---


class MiniMaxText01RMSNormTP(CustomOp):
    name = "MiniMaxText01RMSNormTP"

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.tp_world = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.weight = nn.Parameter(torch.ones(int(hidden_size / self.tp_world)))

        self.weight.weight_loader = self.weight_loader
        self.variance_epsilon = eps
        self.prefix = "UnknownRMSNormTP"
        return

    @staticmethod
    def weight_loader(
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
    ) -> None:
        tp_world = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()

        shard_size = loaded_weight.shape[0] // tp_world
        shard = slice(tp_rank * shard_size, (tp_rank + 1) * shard_size)
        param.data.copy_(loaded_weight[shard])
        return

    # def _forward(
    #     self,
    #     x: torch.Tensor,
    # ) -> torch.Tensor:
    #     orig_dtype = x.dtype
    #     x = x.to(torch.float32)
    #     variance = x.pow(2).mean(dim=-1, keepdim=True, dtype=torch.float32)
    #     if self.tp_world > 1:
    #         variance = tensor_model_parallel_all_reduce(variance) / self.tp_world
    #     x = x * torch.rsqrt(variance + self.variance_epsilon)
    #     x = x.to(orig_dtype) * self.weight
    #     return x

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # --- [修改] 在这里加入详细的对称日志 ---
        print(f"--- [vLLM DEBUG] Entering RMSNormTP for '{self.prefix}' ---")
        print_tensor_stats(x, f"RMSNorm_Input_{self.prefix}")

        orig_dtype = x.dtype
        x = x.to(torch.float32)

        variance = x.pow(2).mean(dim=-1, keepdim=True, dtype=torch.float32)
        print_tensor_stats(variance, f"RMSNorm_Variance_Before_AllReduce_{self.prefix}")

        if self.tp_world > 1:
            variance = tensor_model_parallel_all_reduce(variance) / self.tp_world
            print_tensor_stats(variance, f"RMSNorm_Variance_After_AllReduce_{self.prefix}")

        # 与FD侧对齐，打印最终的 variance
        print_tensor_stats(variance, f"RMSNorm_Variance_{self.prefix}")

        inv_std = torch.rsqrt(variance + self.variance_epsilon)
        print_tensor_stats(inv_std, f"RMSNorm_InvStd_{self.prefix}")
        
        x = x * inv_std
        x = x.to(orig_dtype) * self.weight

        print(f"--- [vLLM DEBUG] Exiting RMSNormTP for '{self.prefix}' ---")
        return x

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        assert residual is None, "RMSNorm does not support residual connection."
        return self._forward(x)


class MiniMaxText01LinearKernel:
    @staticmethod
    def jit_linear_forward_prefix(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_caches: torch.Tensor,
        slope_rate: torch.Tensor,
        block_size: int,
        layer_idx: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        slope_rate = slope_rate.to(torch.float32)
        should_pad_dim = q.dim() == 3
        if should_pad_dim:
            q = q.unsqueeze(0)
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)
        b, h, n, d = q.shape
        e = d
        kv_history = kv_caches.reshape(1, h, d, e).contiguous()
        output, kv_history = lightning_attention(
            q, k, v, slope_rate, block_size=block_size, kv_history=kv_history
        )

        print_tensor_stats(output, "VLLM_Lightning_Attention_Output")
        kv_caches.copy_(kv_history[:, :, -1, :, :].reshape(h, d, e))
        assert output.shape[0] == 1, "batch size must be 1"
        return rearrange(output.squeeze(0), "h n d -> n (h d)")


class MiniMaxText01LinearAttention(nn.Module, MambaBase):
    @property
    def mamba_type(self) -> str:
        return "linear_attention"

    def get_attn_backend(self) -> type["AttentionBackend"]:
        from vllm.v1.attention.backends.linear_attn import LinearAttentionBackend

        return LinearAttentionBackend

    def get_state_dtype(self) -> tuple[torch.dtype]:
        assert self.model_config is not None
        assert self.cache_config is not None
        return MambaStateDtypeCalculator.linear_attention_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
        )

    def get_state_shape(self) -> tuple[tuple[int, int, int], ...]:
        return MambaStateShapeCalculator.linear_attention_state_shape(
            num_heads=self.num_heads, tp_size=self.tp_size, head_dim=self.head_dim
        )

    def __init__(
        self,
        hidden_size: int,
        hidden_inner_size: int,
        num_heads: int,
        head_dim: int,
        max_position: int,
        block_size: int,
        num_hidden_layer: int,
        model_config: Optional[ModelConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        layer_idx: int = 0,
        linear_layer_idx: int = 0,
        prefix: str = "linear_attn",
    ) -> None:
        super().__init__()

        self.layer_idx = layer_idx
        self.BLOCK = block_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.total_num_heads = num_heads
        self.hidden_inner_size = hidden_inner_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        assert self.total_num_heads % self.tp_size == 0
        self.tp_heads = self.total_num_heads // self.tp_size
        self.qkv_size = self.num_heads * self.head_dim
        self.tp_hidden = self.head_dim * self.tp_heads
        self.model_config = model_config
        self.cache_config = cache_config
        self.prefix = prefix

        self.qkv_proj = ColumnParallelLinear(
            hidden_size,
            self.hidden_inner_size * 3,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.output_gate = ColumnParallelLinear(
            hidden_size,
            self.hidden_inner_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.output_gate",
        )
        self.out_proj = RowParallelLinear(
            self.hidden_inner_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )
        self.norm = MiniMaxText01RMSNormTP(
            self.hidden_inner_size,
            eps=1e-5,
        )

        slope_rate = MiniMaxText01LinearAttention._build_slope_tensor(self.num_heads)
        if num_hidden_layer <= 1:
            self.slope_rate = slope_rate * (1 + 1e-5)
        else:
            self.slope_rate = slope_rate * (
                1 - layer_idx / (num_hidden_layer - 1) + 1e-5
            )
        self.tp_slope = self.slope_rate[
            self.tp_rank * self.tp_heads : (self.tp_rank + 1) * self.tp_heads
        ].contiguous()

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    @staticmethod
    def weight_direct_load(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        assert param.size() == loaded_weight.size()
        param.data.copy_(loaded_weight)
        return

    @staticmethod
    def _build_slope_tensor(n_attention_heads: int):
        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = 2 ** (-(2 ** -(math.log2(n) - 3)))
                ratio = start
                return [start * ratio**i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)
            else:
                closest_power_of_2 = 2 ** math.floor(math.log2(n))
                return (
                    get_slopes_power_of_2(closest_power_of_2)
                    + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
                )

        slopes = torch.tensor(
            get_slopes(n_attention_heads), dtype=torch.float32
        ).reshape(n_attention_heads, 1, 1)
        return slopes

    def _prefill_and_mix_infer(
        self, q, k, v, kv_cache, state_indices_tensor, attn_metadata
    ):
        hidden = []
        for _prefill_idx in range(getattr(attn_metadata, "num_prefills", 0)):
            if _prefill_idx >= len(attn_metadata.query_start_loc):
                break
            if _prefill_idx >= len(state_indices_tensor):
                break
            offset = attn_metadata.num_decode_tokens
            _start = attn_metadata.query_start_loc[offset + _prefill_idx]
            _end = attn_metadata.query_start_loc[offset + _prefill_idx + 1]
            slot_id = state_indices_tensor[offset + _prefill_idx]
            qs = q[_start:_end].transpose(0, 1).contiguous()
            ks = k[_start:_end].transpose(0, 1).contiguous()
            vs = v[_start:_end].transpose(0, 1).contiguous()
            slice_layer_cache = kv_cache[slot_id, ...]

            out_slice = MiniMaxText01LinearKernel.jit_linear_forward_prefix(
                qs,
                ks,
                vs,
                slice_layer_cache,
                self.tp_slope,
                self.BLOCK,
                layer_idx=self.layer_idx,
            )
            hidden.append(out_slice.contiguous())
        if attn_metadata.num_decode_tokens > 0:
            hidden_decode = self._decode_infer(
                q, k, v, kv_cache, state_indices_tensor, attn_metadata
            )
            hidden.insert(0, hidden_decode)

        if not hidden:
            return torch.empty((0, q.size(-1)), device=q.device, dtype=q.dtype)

        hidden = torch.concat(hidden, dim=0).contiguous()
        return hidden

    def _decode_infer(self, q, k, v, kv_cache, state_indices_tensor, attn_metadata):
        q = q[: attn_metadata.num_decode_tokens].unsqueeze(2).contiguous()
        k = k[: attn_metadata.num_decode_tokens].unsqueeze(2).contiguous()
        v = v[: attn_metadata.num_decode_tokens].unsqueeze(2).contiguous()
        slot_id = state_indices_tensor[: attn_metadata.num_decodes]
        hidden = linear_decode_forward_triton(
            q, k, v, kv_cache, self.tp_slope, slot_id, 32
        )
        return hidden

    def forward(
        self, hidden_states: torch.Tensor, output: torch.Tensor, positions: torch.Tensor
    ) -> None:
        torch.ops.vllm.linear_attention(
            hidden_states,
            output,
            positions,
            self.prefix,
        )

    def _forward(
        self, hidden_states: torch.Tensor, output: torch.Tensor, positions: torch.Tensor
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata: AttentionMetadata = forward_context.attn_metadata
        if self.layer_idx == 0: # 只打印第一层
            print_tensor_stats(self.qkv_proj.weight, "VLLM_L0_QKV_PROJ_WEIGHT")
        if attn_metadata is not None:
            assert isinstance(attn_metadata, dict)
            attn_metadata = attn_metadata[self.prefix]
            assert isinstance(attn_metadata, LinearAttentionMetadata)
            num_actual_tokens = (
                attn_metadata.num_prefill_tokens + attn_metadata.num_decode_tokens
            )
        else:
            num_actual_tokens = hidden_states.shape[0]
            
        # ========== [新增调试代码] ==========
        if self.layer_idx == 0: # <--- 修正为 layer_idx
            # hidden_states[:num_actual_tokens] 是即将进入 matmul 的输入
            print_tensor_stats(hidden_states[:num_actual_tokens], f"VLLM_MATMUL_INPUT_L{self.layer_idx}")
            # PyTorch的Linear层权重是 [out, in]，而 matmul(input, weight.T)
            # 这里 ColumnParallelLinear 内部处理了，我们直接打印 weight 本身
            print_tensor_stats(self.qkv_proj.weight, f"VLLM_MATMUL_WEIGHT_L{self.layer_idx}")
        # ========== [结束新增] ==========
        
        # ========== [修改这里的调试代码] ==========
        # 只在 rank 0 并且是第一次调用时执行保存操作
        # 我们用一个简单的 flag 来防止重复保存
        if self.layer_idx == 0 and not hasattr(self, '_weight_dumped'):
            
            # 打印信息，确认正在保存
            if get_tensor_model_parallel_rank() == 0:
                print(f"\n--- [vLLM DEBUG] Dumping L{self.layer_idx} QKV weight for rank 0... ---\n")
            
            # 获取最终加载到GPU上的权重参数
            weight_shard = self.qkv_proj.weight 

            # 保存到文件
            # 注意：vLLM的权重布局是 [output_features, input_features]
            if get_tensor_model_parallel_rank() == 0:
                torch.save(weight_shard.cpu().float(), "vllm_qkv_weight_shard_rank0.pt")
                print(f"\n--- [vLLM DEBUG] Saved rank 0 weight shard to vllm_qkv_weight_shard_rank0.pt ---\n")

            # 设置 flag，防止重复保存
            self._weight_dumped = True

        # ========== [结束修改] ==========

        qkv, _ = self.qkv_proj(hidden_states[:num_actual_tokens])
        qkv32 = qkv.to(torch.float32)
        qkvact = torch.nn.functional.silu(qkv32)
        qkvact = qkvact.view((qkv.shape[0], self.tp_heads, -1))
        q, k, v = torch.split(qkvact, [self.head_dim] * 3, dim=-1)
        if attn_metadata is not None:
            kv_cache = self.kv_cache[forward_context.virtual_engine][0]
            state_indices_tensor = attn_metadata.state_indices_tensor

            num_prefills = getattr(attn_metadata, "num_prefills", 0)
            if num_prefills > 0:
                num_decode_tokens = getattr(attn_metadata, "num_decode_tokens", 0)
                for prefill_idx in range(num_prefills):
                    q_start = attn_metadata.query_start_loc[
                        num_decode_tokens + prefill_idx
                    ]
                    q_end = attn_metadata.query_start_loc[
                        num_decode_tokens + prefill_idx + 1
                    ]
                    query_len = q_end - q_start
                    context_len = (
                        attn_metadata.seq_lens[num_decode_tokens + prefill_idx]
                        - query_len
                    )
                    if context_len == 0:
                        block_to_clear = state_indices_tensor[
                            num_decode_tokens + prefill_idx
                        ]
                        kv_cache[block_to_clear, ...] = 0

        decode_only = getattr(attn_metadata, "num_prefills", 0) == 0
        if attn_metadata is None:
            hidden = torch.empty(
                (q.shape[0], q.shape[1] * q.shape[2]), device=q.device, dtype=q.dtype
            )
        else:
            if not decode_only:
                hidden = self._prefill_and_mix_infer(
                    q, k, v, kv_cache, state_indices_tensor, attn_metadata
                )
            else:
                hidden = self._decode_infer(
                    q, k, v, kv_cache, state_indices_tensor, attn_metadata
                )
        hidden = self.norm._forward(hidden)
        gate, _ = self.output_gate(hidden_states[:num_actual_tokens])
        hidden = F.sigmoid(gate) * hidden
        hidden = hidden.to(hidden_states.dtype)

        output[:num_actual_tokens], _ = self.out_proj(hidden)


def linear_attention(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    positions: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self._forward(hidden_states=hidden_states, output=output, positions=positions)


def linear_attention_fake(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    positions: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="linear_attention",
    op_func=linear_attention,
    mutates_args=["output"],
    fake_impl=linear_attention_fake,
)
