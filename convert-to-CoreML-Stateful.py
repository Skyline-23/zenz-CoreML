import os
from typing import List, Tuple

import coremltools as ct
import numpy as np
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel
from transformers.cache_utils import Cache

from transformers.models.gpt2 import modeling_gpt2 as gpt2_mod


def _patched_create_causal_mask(*args, **kwargs):
    """
    TorchScript-friendly causal mask for GPT-2.

    KR: TorchScript 변환 시 깨지는 최신 masking_utils / vmap / dynamo 경로를 우회하기 위해
        가장 단순한 causal(하삼각) 마스크를 직접 생성한다. 과거와 현재 토큰만 attend 가능하고,
        미래 토큰 위치에는 -inf를 넣어서 softmax에서 제외한다.

    JP: TorchScript 変換で壊れやすい最新の masking_utils / vmap / dynamo の経路を避けるため、
        単純な causal（三角）マスクを自前で生成する。過去と現在のトークンのみ参照し、
        未来のトークン位置には -inf を入れて softmax の対象外にする。

    EN: This replaces the newer masking_utils / vmap / dynamo-based masking logic with a
        simple lower-triangular causal mask. Only past and current tokens are allowed
        to attend; future positions are filled with -inf so that softmax ignores them.
    """
    import torch  # ensure torch is available in traced context

    # 1) batch / sequence 길이 추출
    batch_size = None
    q_len = None
    past_kv_len = int(kwargs.get("past_key_values_length", 0))

    input_ids = kwargs.get("input_ids", None)
    inputs_embeds = kwargs.get("inputs_embeds", None)
    input_shape = kwargs.get("input_shape", None)

    if input_ids is not None and isinstance(input_ids, torch.Tensor):
        batch_size, q_len = int(input_ids.shape[0]), int(input_ids.shape[1])
        device = input_ids.device
    elif inputs_embeds is not None and isinstance(inputs_embeds, torch.Tensor):
        batch_size, q_len = int(inputs_embeds.shape[0]), int(inputs_embeds.shape[1])
        device = inputs_embeds.device
    elif input_shape is not None:
        # input_shape is typically (batch, seq)
        batch_size, q_len = int(input_shape[0]), int(input_shape[1])
        device = kwargs.get("device", torch.device("cpu"))
    elif len(args) >= 2:
        # args[0] might be config, args[1] is often input_ids or input_shape
        candidate = args[1]
        if isinstance(candidate, torch.Tensor):
            batch_size, q_len = int(candidate.shape[0]), int(candidate.shape[1])
            device = candidate.device
        elif isinstance(candidate, (tuple, list)) and len(candidate) >= 2:
            batch_size, q_len = int(candidate[0]), int(candidate[1])
            device = kwargs.get("device", torch.device("cpu"))
        else:
            # 최악의 경우, 안전한 기본값
            batch_size, q_len = 1, int(candidate)
            device = kwargs.get("device", torch.device("cpu"))
    else:
        # 그래도 못 찾으면 아주 보수적인 기본값
        batch_size, q_len = 1, 1
        device = kwargs.get("device", torch.device("cpu"))

    dtype = kwargs.get("dtype", torch.float32)

    k_len = q_len + past_kv_len

    # 2) 단순 causal mask 생성: (batch, 1, q_len, k_len)
    #  - 허용 위치: 현재 및 과거 토큰 (값 0.0)
    #  - 미래 토큰: -inf (softmax 전에 더해지는 additive mask)
    mask = torch.zeros((q_len, k_len), dtype=dtype, device=device)
    future = torch.triu(torch.ones((q_len, k_len), dtype=torch.bool, device=device), diagonal=1)
    mask = mask.masked_fill(future, float("-inf"))

    # (1, 1, q_len, k_len) -> (batch, 1, q_len, k_len)
    mask = mask.unsqueeze(0).unsqueeze(0)
    if batch_size is not None:
        mask = mask.expand(batch_size, 1, q_len, k_len)

    return mask


# KR: GPT-2가 내부에서 사용하는 create_causal_mask를 우리가 정의한 TorchScript 친화적인 버전으로 교체한다.
# JP: GPT-2 内部の create_causal_mask を、TorchScript 向けに安全な実装に差し替える。
# EN: Override GPT-2's internal create_causal_mask with our TorchScript-friendly implementation.
gpt2_mod.create_causal_mask = _patched_create_causal_mask

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_NAME = "Miwa-Keita/zenz-v1-checkpoints"
MAX_CONTEXT_SIZE = 256
BATCH_SIZE = 1


class SliceUpdateKeyValueCache:
    """
    KV cache of shape:
      (num_layers, batch_size, num_heads, context_size, head_dim)

    KR: Core ML state(keyCache / valueCache)와 동일한 모양의 KV 캐시를 PyTorch 쪽에서 관리하기 위한
        헬퍼 클래스이다. Hugging Face GPT-2가 내보내는 (batch, head, seq_len, dim) 형식의 K/V 텐서를
        레이어별로 모아서 위의 5차원 텐서에 저장한다.

    JP: Core ML の state（keyCache / valueCache）と同じ形状の KV キャッシュを、PyTorch 側で管理する
        ためのヘルパークラス。Hugging Face GPT-2 が返す (batch, head, seq_len, dim) 形式の K/V テンソルを
        レイヤごとに集約し、上記 5 次元テンソルに格納する。

    EN: Helper class that keeps a KV cache with the same shape as the Core ML state
        tensors (keyCache / valueCache). It aggregates per-layer K/V tensors from
        Hugging Face GPT-2 (shape: (batch, head, seq_len, dim)) into a single 5D tensor.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        # 현재 단계에서는 transformers.Cache 기능을 직접 사용하지 않고,
        # Core ML state에 맞는 KV 텐서를 관리하는 용도로만 사용한다.
        self.past_seen_tokens: int = 0
        self.k_cache: torch.Tensor = torch.zeros(shape, dtype=dtype, device=device)
        self.v_cache: torch.Tensor = torch.zeros(shape, dtype=dtype, device=device)

    def update(
        self,
        k_state: torch.Tensor,
        v_state: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update key/value cache tensors for a given layer using full-sequence KV.

        KR: Hugging Face GPT-2가 반환한 단일 레이어의 K/V 텐서
            (batch, num_heads, seq_len, head_dim)를 받아서, 내부 캐시 텐서
            (num_layers, batch, num_heads, context_size, head_dim)에 써 넣는다.

        JP: Hugging Face GPT-2 が返す単一レイヤの K/V テンソル
            (batch, num_heads, seq_len, head_dim) を受け取り、内部のキャッシュ
            (num_layers, batch, num_heads, context_size, head_dim) に書き込む。

        EN: Takes a single layer's K/V tensors from Hugging Face GPT-2
            (batch, num_heads, seq_len, head_dim) and writes them into the
            internal cache tensor (num_layers, batch, num_heads, context_size, head_dim).
        """
        bsz, num_heads, seq_len, head_dim = k_state.shape
        max_len = self.k_cache.shape[3]

        # KR/JP/EN: 시퀀스 길이가 context_size를 넘을 경우 잘라서 저장한다.
        #           If seq_len exceeds context_size, truncate to max_len.
        write_len = min(seq_len, max_len)

        # k_cache[layer, batch, head, pos, dim]
        # k_state[batch, head, seq_len, dim]
        self.k_cache[layer_idx, :bsz, :num_heads, :write_len, :] = k_state[:, :, :write_len, :]
        self.v_cache[layer_idx, :bsz, :num_heads, :write_len, :] = v_state[:, :, :write_len, :]

        # 반환값은 현재 레이어의 유효 KV 부분
        k_cache = self.k_cache[layer_idx, :, :, :write_len, :]
        v_cache = self.v_cache[layer_idx, :, :, :write_len, :]

        # past_seen_tokens는 전체 시퀀스 길이로 갱신 (모든 레이어가 동일한 길이를 공유한다고 가정)
        self.past_seen_tokens = write_len

        return k_cache, v_cache

    def get_seq_length(self, _: int | None = 0) -> int:
        return self.past_seen_tokens


class StatefulGPT2ForCausalLM(torch.nn.Module):
    """
    Mistral의 StatefulMistralForCausalLM 구조를 GPT-2에 맞게 단순화한 버전.

    KR:
    - Core ML state (keyCache / valueCache)를 Hugging Face GPT-2의 past_key_values와
      양방향으로 연결해서 토큰 단위 KV 캐시를 유지한다.
    - iOS 측에서는 keyCache / valueCache를 state로 넘기고, PyTorch 측에서는
      past_key_values로 다시 조립하여 GPT-2에 전달한다.

    JP:
    - Mistral の StatefulMistralForCausalLM の構造を GPT-2 向けに簡略化したクラス。
    - Core ML 側の state（keyCache / valueCache）と、Hugging Face GPT-2 の past_key_values
      を相互に変換しながら、トークン単位で KV キャッシュを維持する。

    EN:
    - Simplified GPT-2 counterpart of Mistral's StatefulMistralForCausalLM.
    - Bridges Core ML states (keyCache / valueCache) with Hugging Face GPT-2
      past_key_values, maintaining a token-wise KV cache for incremental decoding.
    """

    def __init__(self, model_name: str, max_context_size: int = MAX_CONTEXT_SIZE, batch_size: int = BATCH_SIZE):
        super().__init__()

        self.model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()

        # KR: 최신 transformers는 기본 attention 구현이 "sdpa"라서, TorchScript + Core ML 변환 시
        #     masking_utils + vmap 경로를 타며 `unordered_map::at` 에러를 일으킨다.
        #     이를 피하기 위해 가능한 경우 "eager" 구현으로 강제로 설정한다.
        # JP: 最近の transformers ではデフォルトの attention 実装が "sdpa" になっており、
        #     TorchScript + Core ML 変換時に masking_utils + vmap の経路でクラッシュする。
        #     そのため、可能なら "eager" 実装に強制的に切り替える。
        # EN: Newer transformers use "sdpa" as the default attention implementation, which tends
        #     to crash during TorchScript + Core ML conversion due to masking_utils + vmap.
        #     To avoid this, we force the attention implementation to "eager" when possible.
        cfg = self.model.config
        if hasattr(cfg, "attn_implementation"):
            cfg.attn_implementation = "eager"
        elif hasattr(cfg, "_attn_implementation"):
            cfg._attn_implementation = "eager"  # 일부 버전에서 내부 필드 이름

        config = self.model.config

        self.num_layers = config.n_layer
        self.num_heads = config.n_head
        self.head_dim = config.n_embd // config.n_head

        self.kv_cache_shape: Tuple[int, ...] = (
            self.num_layers,
            batch_size,
            self.num_heads,
            max_context_size,
            self.head_dim,
        )

        # KV cache object (Python level)
        self.kv_cache = SliceUpdateKeyValueCache(shape=self.kv_cache_shape)

        # Core ML state용 buffer (이름 중요)
        self.register_buffer("keyCache", self.kv_cache.k_cache)
        self.register_buffer("valueCache", self.kv_cache.v_cache)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        GPT-2의 past_key_values를 Core ML state(keyCache / valueCache)와 연결한 stateful forward.

        - self.kv_cache.past_seen_tokens: 현재까지 누적된 토큰 수
        - keyCache / valueCache: (num_layers, batch, num_heads, context_size, head_dim)
        - Hugging Face GPT-2 past_key_values:
          튜플 길이 num_layers, 각 요소는 (k, v) with shape (batch, num_heads, seq_len, head_dim)
        """

        # KR: Core ML에서 넘어온 state(keyCache / valueCache)를 past_key_values로 재구성한 뒤,
        #     GPT-2를 한 번 호출하고, 그 결과의 past_key_values를 다시 state에 반영한다.
        # JP: Core ML から渡された state（keyCache / valueCache）を past_key_values に組み立て直し、
        #     GPT-2 を 1 ステップ実行してから、新しい past_key_values を state に書き戻す。
        # EN: Rebuild past_key_values from the Core ML states, run a single GPT-2 step,
        #     then write back the updated past_key_values into the Core ML states.

        batch_size, current_len = input_ids.shape

        # KR: 과거 토큰이 존재하는 경우, 레이어별 K/V를 모아서 HF 포맷의 past_key_values 튜플을 만든다.
        # JP: すでに過去のトークンがある場合、レイヤごとの K/V を集めて HF 形式の past_key_values を組み立てる。
        # EN: If there are previous tokens, collect per-layer K/V tensors to build
        #     the Hugging Face-style past_key_values tuple.
        past_len = self.kv_cache.get_seq_length()
        past_key_values = None
        if past_len > 0:
            pkv = []
            for layer_idx in range(self.num_layers):
                # (batch, head, past_len, dim)
                k_past = self.kv_cache.k_cache[layer_idx, :batch_size, :, :past_len, :]
                v_past = self.kv_cache.v_cache[layer_idx, :batch_size, :, :past_len, :]
                pkv.append((k_past, v_past))
            past_key_values = tuple(pkv)

        # KR/JP/EN: patched causal mask + eager attention 설정을 사용하여 한 스텝 forward.
        outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )

        logits = outputs.logits
        present_key_values = outputs.past_key_values

        # KR: GPT-2가 반환한 현재까지의 전체 K/V(past + current)를 다시 state 텐서에 반영한다.
        # JP: GPT-2 が返した（過去 + 現在）の K/V 全体を state テンソルに書き戻す。
        # EN: Write back the full K/V tensors (past + current) returned by GPT-2 into the state.
        if present_key_values is not None:
            # 모든 레이어의 시퀀스 길이가 동일하다고 가정
            total_seq_len = present_key_values[0][0].shape[2]

            for layer_idx, (k_layer, v_layer) in enumerate(present_key_values):
                self.kv_cache.update(
                    k_state=k_layer,
                    v_state=v_layer,
                    layer_idx=layer_idx,
                )

        # 4) Core ML에서 state를 "사용"하고 있다고 인식시키기 위한 더미 참조
        key_sample = self.keyCache.reshape(-1)[0].to(logits.dtype)
        value_sample = self.valueCache.reshape(-1)[0].to(logits.dtype)
        dummy = (key_sample + value_sample) * 0.0

        return logits + dummy


def convert_model(model_name: str, output_path: str) -> None:
    # 토크나이저는 그대로
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Torch 쪽 stateful 래퍼
    torch_model = StatefulGPT2ForCausalLM(
        model_name=model_name,
        max_context_size=MAX_CONTEXT_SIZE,
        batch_size=BATCH_SIZE,
    ).eval()

    # trace용 예제 입력 (길이는 아무거나, 상한은 MAX_CONTEXT_SIZE)
    example_input_ids = torch.zeros((BATCH_SIZE, 4), dtype=torch.long)
    example_attention_mask = torch.ones((BATCH_SIZE, 4), dtype=torch.long)

    # TorchScript trace (masking_utils 고차 연산 대신, 우리가 패치한 causal mask 사용)
    traced_model = torch.jit.trace(
        torch_model,
        (example_input_ids, example_attention_mask),
        check_trace=False,
    )

    kv_cache_shape = torch_model.kv_cache_shape

    # Core ML 입력/출력/상태 스펙 (Mistral export.py 스타일)
    query_length = ct.RangeDim(lower_bound=1, upper_bound=MAX_CONTEXT_SIZE, default=1)

    inputs: List[ct.TensorType] = [
        ct.TensorType(
            shape=(BATCH_SIZE, query_length),
            dtype=np.int32,
            name="input_ids",
        ),
        ct.TensorType(
            shape=(BATCH_SIZE, query_length),
            dtype=np.int32,
            name="attention_mask",
        ),
    ]

    outputs: List[ct.TensorType] = [
        ct.TensorType(dtype=np.float16, name="logits"),
    ]

    # KR: Core ML 모델이 유지할 state 정의. keyCache / valueCache는 PyTorch 쪽의
    #     SliceUpdateKeyValueCache.k_cache / v_cache와 모양이 정확히 일치해야 한다.
    # JP: Core ML モデルが保持する state の定義。keyCache / valueCache の形状は、PyTorch 側の
    #     SliceUpdateKeyValueCache.k_cache / v_cache と完全に一致している必要がある。
    # EN: Definition of the Core ML model states. keyCache / valueCache must have exactly the
    #     same shape as SliceUpdateKeyValueCache.k_cache / v_cache on the PyTorch side.
    states: List[ct.StateType] = [
        ct.StateType(
            wrapped_type=ct.TensorType(shape=kv_cache_shape, dtype=np.float16),
            name="keyCache",
        ),
        ct.StateType(
            wrapped_type=ct.TensorType(shape=kv_cache_shape, dtype=np.float16),
            name="valueCache",
        ),
    ]

    mlmodel_fp16: ct.MLModel = ct.convert(
        traced_model,
        inputs=inputs,
        outputs=outputs,
        states=states,
        source="pytorch",
        minimum_deployment_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.ALL,
        skip_model_load=True,
    )

    # Inspect spec to find actual input/output feature names, including stateful ones.
    spec = mlmodel_fp16.get_spec()
    input_names = {f.name for f in spec.description.input}
    output_names = {f.name for f in spec.description.output}

    def _resolve_state_name(base: str) -> Tuple[str | None, str | None]:
        """
        KR: state 이름이 'keyCache', 'keyCache_in', 'keyCache_out' 등으로 붙었을 수 있으니
            실제 input/output에 존재하는 이름을 찾아서 반환한다.
        JP: state 名が 'keyCache', 'keyCache_in', 'keyCache_out' などになっている可能性があるため、
            実際に存在する input/output 名を探索して返す。
        EN: State features might be named 'keyCache', 'keyCache_in', or 'keyCache_out', etc.
            This helper returns the first matching input and output names, if any.
        """
        candidates = (base, base + "_in", base + "_out")
        in_name = None
        out_name = None
        for name in candidates:
            if in_name is None and name in input_names:
                in_name = name
            if out_name is None and name in output_names:
                out_name = name
        return in_name, out_name

    key_in_name, key_out_name = _resolve_state_name("keyCache")
    val_in_name, val_out_name = _resolve_state_name("valueCache")

    # Set feature descriptions for stateful model
    mlmodel_fp16.input_description["input_ids"] = (
        "Input token IDs for the stateful zenz-v1 language model.\n"
        "Shape: [batch, query_length] with Int32 values."
    )
    mlmodel_fp16.input_description["attention_mask"] = (
        "Attention mask for the input tokens. 1 for valid tokens, 0 for padding.\n"
        "Shape: [batch, query_length] with Int32 values."
    )
    mlmodel_fp16.output_description["logits"] = (
        "Unnormalized next-token logits for each vocabulary token, taking into account the current KV cache state.\n"
        "Shape: [batch, query_length, vocab_size]."
    )

    # State tensors description (KV cache)
    if key_in_name is not None:
        mlmodel_fp16.input_description[key_in_name] = (
            "State tensor storing past key values for all transformer layers.\n"
            "Shape: [num_layers, batch, num_heads, max_context_size, head_dim] in fp16."
        )
    if key_out_name is not None:
        mlmodel_fp16.output_description[key_out_name] = (
            "State tensor storing past key values for all transformer layers.\n"
            "Shape: [num_layers, batch, num_heads, max_context_size, head_dim] in fp16."
        )

    if val_in_name is not None:
        mlmodel_fp16.input_description[val_in_name] = (
            "State tensor storing past value values for all transformer layers.\n"
            "Shape: [num_layers, batch, num_heads, max_context_size, head_dim] in fp16."
        )
    if val_out_name is not None:
        mlmodel_fp16.output_description[val_out_name] = (
            "State tensor storing past value values for all transformer layers.\n"
            "Shape: [num_layers, batch, num_heads, max_context_size, head_dim] in fp16."
        )

    # Author information: original + Core ML conversion
    mlmodel_fp16.author = (
        "Original model: Miwa-Keita\n"
        "Stateful Core ML conversion: Skyline-23 (Buseong Kim)"
    )

    # License information
    mlmodel_fp16.license = (
        "CC-BY-SA 4.0"
    )

    # Short description for Xcode
    mlmodel_fp16.short_description = (
        "Stateful Core ML variant of the zenz-v1 GPT-2–style language model.\n"
        "Maintains key/value attention cache (keyCache/valueCache) to enable efficient incremental text generation."
    )

    # Preview type and version
    mlmodel_fp16.user_defined_metadata["com.apple.coreml.model.preview.type"] = "textGenerator"
    mlmodel_fp16.version = "1.0.0-stateful"

    mlmodel_fp16.save(output_path)

if __name__ == "__main__":
    convert_model(MODEL_NAME, "zenz_v1_stateful.mlpackage")