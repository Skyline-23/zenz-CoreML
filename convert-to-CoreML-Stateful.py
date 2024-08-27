import torch
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel, GPT2Attention, GPT2_ATTENTION_CLASSES
from transformers import AutoTokenizer
import coremltools as ct
from typing import Optional, Tuple
import numpy as np
from transformers.cache_utils import Cache
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class SliceUpdateKeyValueCache(Cache):
    def __init__(self, shape: Tuple[int, ...], device="cpu", dtype=torch.float32) -> None:
        """KV cache of shape (#layers, batch_size, #kv_heads, context_size, head_dim)."""
        super().__init__()
        self.past_seen_tokens: int = 0
        self.k_cache: torch.Tensor = torch.zeros(shape, dtype=dtype, device=device)
        self.v_cache: torch.Tensor = torch.zeros(shape, dtype=dtype, device=device)

    def update(self, k_state: torch.Tensor, v_state: torch.Tensor, layer_idx: int, slice_indices: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update key/value cache tensors for slice [slice_indices[0], slice_indices[1])."""
        if len(slice_indices) != 2:
            raise ValueError(f"Expect tuple of integers [start, end), got {slice_indices=}.")
        begin, end = slice_indices
        self.k_cache[layer_idx, :, : k_state.shape[1], begin:end, :] = k_state
        self.v_cache[layer_idx, :, : v_state.shape[1], begin:end, :] = v_state
        return self.k_cache[layer_idx, :, :, :end, :], self.v_cache[layer_idx, :, :, :end, :]

    def get_seq_length(self, _: int = 0) -> int:
        """Get the sequence length of the cache."""
        return self.past_seen_tokens

    def to_past_key_values(self):
        """Convert the internal cache to a format expected by GPT2."""
        return [(self.k_cache[layer], self.v_cache[layer]) for layer in range(self.k_cache.size(0))]

class SliceUpdateGPT2Attention(GPT2Attention):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config=config, layer_idx=layer_idx)

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor, 
                layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                attention_mask: Optional[torch.FloatTensor] = None, 
                head_mask: Optional[torch.FloatTensor] = None,
                use_cache: bool = False,
                output_attentions: bool = False) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Compute query, key, and value tensors
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        # Handle past key/value tensors using tensor-based condition
        if layer_past is not None:
            past_key, past_value = layer_past
            if past_key.size(-2) > 0:
                key = torch.cat([past_key, key], dim=-2)
                value = torch.cat([past_value, value], dim=-2)

        # Optimize attention mask handling
        if attention_mask is not None:
            attention_mask = attention_mask[:, :, :, -key.size(-2):]

        # Calculate attention output
        attn_output, _ = self._attn(query, key, value, attention_mask, head_mask)
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)

        # Return the updated cache if use_cache is True
        present = (key, value) if use_cache else None
        return attn_output, present
# Load the model and tokenizer
model_name = "Miwa-Keita/zenz-v1-checkpoints"
GPT2_ATTENTION_CLASSES["sdpa"] = SliceUpdateGPT2Attention
model = GPT2LMHeadModel.from_pretrained(model_name).eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare input data
text = "Example sentence"
inputs = tokenizer(text, return_tensors="pt")

# Model tracing
class StatefulZenz(torch.nn.Module):
    def __init__(self, model, max_context_size: int = 256, batch_size: int = 1):
        super(StatefulZenz, self).__init__()
        self.model = model
        config = self.model.config
        self.kv_cache_shape: Tuple[int, ...] = (
            config.num_hidden_layers,
            batch_size,
            config.n_head,
            max_context_size,
            config.hidden_size // config.num_attention_heads,
        )
        self.kv_cache = SliceUpdateKeyValueCache(shape=self.kv_cache_shape)
        self.register_buffer("keyCache", self.kv_cache.k_cache)
        self.register_buffer("valueCache", self.kv_cache.v_cache)

    @torch.no_grad()
    def forward(self, input_ids, attention_mask):
        self.kv_cache.past_seen_tokens = attention_mask.shape[-1] - input_ids.shape[-1]
        past_key_values = self.kv_cache.to_past_key_values()

        # Reintroduce the attention mask extension logic
        attention_mask = self._extend_attention_mask(attention_mask, past_key_values)

        outputs = self.model(input_ids, attention_mask=attention_mask, past_key_values=past_key_values, use_cache=True)
        return outputs.logits

    def _extend_attention_mask(self, attention_mask, past_key_values):
        """Adjust the attention mask to match the size of the key/value cache."""
        if past_key_values is not None:
            past_length = past_key_values[0][0].size(-2)
            new_length = past_length + attention_mask.size(-1)
            extended_attention_mask = torch.ones(
                (attention_mask.size(0), 1, 1, new_length), dtype=attention_mask.dtype, device=attention_mask.device
            )
            extended_attention_mask[:, :, :, -attention_mask.size(-1):] = attention_mask
        else:
            extended_attention_mask = attention_mask
        return extended_attention_mask

# Create the traced model
stateful_zenz = StatefulZenz(model).eval()
traced_model = torch.jit.trace(stateful_zenz, (inputs['input_ids'], inputs['attention_mask']))

# Convert the model to CoreML
mlmodel = ct.convert(
    traced_model,
    inputs=[
        ct.TensorType(name="input_ids", shape=(1, ct.RangeDim(1, 256))),  # 上限を256に設定
        ct.TensorType(name="attention_mask", shape=(1, ct.RangeDim(1, 256)))  # 上限を256に設定
    ],
    outputs=[
        ct.TensorType(dtype=np.float32, name="output")
    ],
    states=[
        ct.StateType(
            wrapped_type=ct.TensorType(
                shape=stateful_zenz.kv_cache_shape
            ),
            name="keyCache",
        ),
        ct.StateType(
            wrapped_type=ct.TensorType(
                shape=stateful_zenz.kv_cache_shape
            ),
            name="valueCache",
        ),
    ],
    minimum_deployment_target=ct.target.iOS18
)

# Save the converted model
mlmodel.save("zenz_v1_cached.mlpackage")
print("Model successfully converted and saved as: zenz_v1_cached.mlpackage")