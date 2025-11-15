import os
from typing import List, Tuple

import coremltools as ct
from coremltools.models import MLModel
from coremltools.optimize.coreml import OpPalettizerConfig, OptimizationConfig, palettize_weights
import numpy as np
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel
from transformers.cache_utils import Cache

from transformers.models.gpt2 import modeling_gpt2 as gpt2_mod


def _patched_create_causal_mask(*args, **kwargs):
    """
    TorchScript 변환 시 깨지는 최신 masking_utils / vmap / dynamo 경로를 우회하기 위해 가장 단순한 causal(하삼각) 마스크를 직접 생성한다. 과거와 현재 토큰만 attend 가능하고, 미래 토큰 위치에는 -inf를 넣어서 softmax에서 제외한다。
    TorchScript 変換で壊れやすい最新の masking_utils / vmap / dynamo の経路を避けるため、単純な causal（三角）マスクを自前で生成する。過去と現在のトークンのみ参照し、未来のトークン位置には -inf を入れて softmax の対象外にする。
    This replaces the newer masking_utils / vmap / dynamo-based masking logic with a simple lower-triangular causal mask. Only past and current tokens are allowed to attend; future positions are filled with -inf so that softmax ignores them.
    """
    import torch  # ensure torch is available in traced context

    # 1) batch / sequence 길이 추출。
    # バッチ / シーケンス長の抽出。
    # Extract batch / sequence length.
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
            # 최악의 경우, 안전한 기본값。
            # 最悪の場合、安全なデフォルト値。
            # In the worst case, use a safe default.
            batch_size, q_len = 1, int(candidate)
            device = kwargs.get("device", torch.device("cpu"))
    else:
        # 그래도 못 찾으면 아주 보수적인 기본값。
        # それでも見つからなければ、非常に保守的なデフォルト値。
        # If still not found, use a very conservative default.
        batch_size, q_len = 1, 1
        device = kwargs.get("device", torch.device("cpu"))

    dtype = kwargs.get("dtype", torch.float32)

    k_len = q_len + past_kv_len

    # 2) 단순 causal mask 생성: (batch, 1, q_len, k_len)。
    #  - 허용 위치: 현재 및 과거 토큰 (값 0.0)。
    #  - 미래 토큰: -inf (softmax 전에 더해지는 additive mask)。
    # 2) 単純な causal マスク生成: (batch, 1, q_len, k_len)。
    #  - 許可位置: 現在および過去トークン (値 0.0)。
    #  - 未来トークン: -inf (softmax 前に加算されるアディティブマスク)。
    # 2) Create a simple causal mask: (batch, 1, q_len, k_len).
    #  - Allowed positions: current and past tokens (value 0.0).
    #  - Future tokens: -inf (additive mask added before softmax).
    mask = torch.zeros((q_len, k_len), dtype=dtype, device=device)
    future = torch.triu(torch.ones((q_len, k_len), dtype=torch.bool, device=device), diagonal=1)
    mask = mask.masked_fill(future, float("-inf"))

    # (1, 1, q_len, k_len) -> (batch, 1, q_len, k_len).
    mask = mask.unsqueeze(0).unsqueeze(0)
    if batch_size is not None:
        mask = mask.expand(batch_size, 1, q_len, k_len)

    return mask


 # GPT-2가 내부에서 사용하는 create_causal_mask를 우리가 정의한 TorchScript 친화적인 버전으로 교체한다。
 # GPT-2 内部の create_causal_mask を、TorchScript 向けに安全な実装に差し替える。
 # Override GPT-2's internal create_causal_mask with our TorchScript-friendly implementation.
gpt2_mod.create_causal_mask = _patched_create_causal_mask

os.environ["TOKENIZERS_PARALLELISM"] = "false"

 # 변환에 사용할 Hugging Face 모델 이름과, Core ML에서 사용할 최대 컨텍스트 길이 / 배치 크기 설정。
 # 変換に使用する Hugging Face モデル名と、Core ML で使う最大コンテキスト長 / バッチサイズの設定。
 # Name of the Hugging Face model to convert, and max context length / batch size for the Core ML export.
MODEL_NAME = "Miwa-Keita/zenz-v3.1-small"
MAX_CONTEXT_SIZE = 128
BATCH_SIZE = 1


class StatefulGPT2ForCausalLM(torch.nn.Module):
    """
    Mistral의 StatefulMistralForCausalLM 구조를 GPT-2에 맞게 단순화한 버전。
    Mistral の StatefulMistralForCausalLM の構造を GPT-2 向けに簡略化したバージョン。
    Simplified GPT-2 counterpart of Mistral's StatefulMistralForCausalLM.

    - Core ML state (keyCache / valueCache / pastLen)를 Hugging Face GPT-2의 past_key_values와 양방향으로 연결해서 토큰 단위 KV 캐시를 유지한다。
    - Core ML 側の state（keyCache / valueCache / pastLen）と Hugging Face GPT-2 の past_key_values を相互に変換しながら、トークン単位で KV キャッシュを維持する。
    - Bridges Core ML states (keyCache / valueCache / pastLen) with Hugging Face GPT-2 past_key_values, maintaining a token-wise KV cache for incremental decoding.
    - 단, 더 이상 register_buffer로 내부에 숨기지 않고, forward(input_ids, attention_mask, keyCache, valueCache, pastLen) → (logits, newKeyCache, newValueCache, newPastLen) 형태의 “순수 state 머신”으로 설계했다。 Core ML이 이 입력/출력 시그니처를 그대로 stateful 모델로 매핑할 수 있게 하기 위함이다。
    - ただし、state はもはや register_buffer には保持せず、forward(input_ids, attention_mask, keyCache, valueCache, pastLen) → (logits, newKeyCache, newValueCache, newPastLen) という純粋なステートマシンとして設計した。 Core ML がこの入出力シグネチャをそのまま stateful モデルにマッピングできるようにするためである。
    - Unlike the previous version, the state is no longer hidden as register_buffers. Instead, the module is a pure state machine: forward(input_ids, attention_mask, keyCache, valueCache, pastLen) → (logits, newKeyCache, newValueCache, newPastLen), which Core ML can more reliably convert into a stateful mlprogram.
    """

    def __init__(self, model_name: str, max_context_size: int = MAX_CONTEXT_SIZE, batch_size: int = BATCH_SIZE):
        super().__init__()

        self.model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()

        # 최신 transformers는 기본 attention 구현이 "sdpa"라서, TorchScript + Core ML 변환 시 masking_utils + vmap 경로를 타며 `unordered_map::at` 에러를 일으킨다。 이를 피하기 위해 가능한 경우 "eager" 구현으로 강제로 설정한다。
        # 最近の transformers ではデフォルトの attention 実装が "sdpa" になっており、TorchScript + Core ML 変換時に masking_utils + vmap の経路でクラッシュする。そのため、可能なら "eager" 実装に強制的に切り替える。
        # Newer transformers use "sdpa" as the default attention implementation, which tends to crash during TorchScript + Core ML conversion due to masking_utils + vmap. To avoid this, we force the attention implementation to "eager" when possible.
        cfg = self.model.config
        if hasattr(cfg, "attn_implementation"):
            cfg.attn_implementation = "eager"
        elif hasattr(cfg, "_attn_implementation"):
            cfg._attn_implementation = "eager"  # 일부 버전에서 내부 필드 이름

        config = self.model.config

        self.num_layers = config.n_layer
        self.num_heads = config.n_head
        self.head_dim = config.n_embd // config.n_head

        self.max_context_size = max_context_size
        self.batch_size = batch_size

        # Core ML state로 유지할 KV 캐시의 전체 shape 정의 (레이어 수, 배치 크기, 헤드 수, 시퀀스 길이, 헤드 차원)。
        # Core ML の state として保持する KV キャッシュの shape を定義。（レイヤー数, バッチサイズ, ヘッド数, シーケンス長, ヘッド次元）
        # Shape of the KV cache tensor that will be stored as Core ML state: (num_layers, batch_size, num_heads, seq_len, head_dim).
        self.kv_cache_shape: Tuple[int, ...] = (
            self.num_layers,
            batch_size,
            self.num_heads,
            max_context_size,
            self.head_dim,
        )

        # Core ML에서 state로 유지될 버퍼들 (KV 캐시 + 현재까지 본 토큰 길이)。
        # Core ML の state として保持されるバッファ（KV キャッシュ + これまでのトークン長）。
        # Buffers that Core ML will treat as state (KV cache + length of tokens seen so far).
        self.register_buffer("keyCache", torch.zeros(self.kv_cache_shape, dtype=torch.float16))
        self.register_buffer("valueCache", torch.zeros(self.kv_cache_shape, dtype=torch.float16))
        self.register_buffer("pastLen", torch.zeros((1,), dtype=torch.float16))

    @torch.no_grad()
    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        register_buffer 기반 내부 state를 사용하는 Stateful GPT-2 forward입니다。 Core ML state는 명시적 입력이 아닌 버퍼를 통해 노출됩니다。
        register_buffer ベースの内部 state を使った Stateful GPT-2 forward。Core ML の state は明示的な入力ではなくバッファ経由で公開されます。
        Stateful GPT-2 forward using register_buffer-based internal state. Core ML states are exposed via buffers, not as explicit inputs.
        """
        batch_size, cur_len = input_ids.shape
        past_len = int(self.pastLen[0].item())
        max_len = self.max_context_size

        # Core ML state로부터 Hugging Face가 기대하는 past_key_values 형태로 복원。
        # Core ML の state から、Hugging Face GPT-2 が使う past_key_values 形式に復元する。
        # Rebuild Hugging Face-style past_key_values from the Core ML state buffers.
        past_key_values = None
        if past_len > 0:
            pkv = []
            for layer_idx in range(self.num_layers):
                k_past = self.keyCache[layer_idx, :batch_size, :, :past_len, :].to(self.model.dtype)
                v_past = self.valueCache[layer_idx, :batch_size, :, :past_len, :].to(self.model.dtype)
                pkv.append((k_past, v_past))
            past_key_values = tuple(pkv)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )

        logits = outputs.logits
        # 전체 시퀀스 로그릿에서 마지막 시점만 남김 → [B, T, V] → [B, 1, V]。
        # 全タイムステップの logits から最後のステップのみを残す → [B, T, V] → [B, 1, V]。
        # Keep only the last time step logits → [B, T, V] → [B, 1, V].
        logits = logits[:, -1:, :]

        present_key_values = outputs.past_key_values

        # 새로 계산된 KV를 잘라서(max_context_size 기준) 내부 state 버퍼에 다시 저장한다。
        # 新しく計算された KV を max_context_size に合わせてスライスし、内部 state バッファに書き戻す。
        # Slice the newly computed KV to fit max_context_size and write it back into the state buffers.
        if present_key_values is not None and len(present_key_values) > 0:
            total_seq_len = present_key_values[0][0].shape[2]
            new_total_len = min(total_seq_len, max_len)
            start = total_seq_len - new_total_len

            for layer_idx, (k_full, v_full) in enumerate(present_key_values):
                k_slice = k_full[:, :, start:, :].to(torch.float16)
                v_slice = v_full[:, :, start:, :].to(torch.float16)
                self.keyCache[layer_idx, :batch_size, :, :new_total_len, :] = k_slice
                self.valueCache[layer_idx, :batch_size, :, :new_total_len, :] = v_slice

            self.pastLen[0] = float(new_total_len)

        return logits


def debug_compare_stateful_vs_stateless():
    """
    Stateful 래퍼(StatefulGPT2ForCausalLM)가 Hugging Face의 기본 GPT2LMHeadModel과 동일한 토큰 시퀀스를 생성하는지 검증하기 위한 디버그 함수입니다。
    Stateful ラッパー(StatefulGPT2ForCausalLM) が Hugging Face の標準 GPT2LMHeadModel と同じトークン列を生成するかどうか検証するためのデバッグ用関数です。
    Debug helper to verify that the StatefulGPT2ForCausalLM wrapper produces the same token sequence as the original GPT2LMHeadModel from Hugging Face.
    """
    device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(device).eval()

    # ✅ Match attention implementation with the stateful wrapper: force "eager" if available
    cfg2 = base_model.config
    if hasattr(cfg2, "attn_implementation"):
        cfg2.attn_implementation = "eager"
    elif hasattr(cfg2, "_attn_implementation"):
        cfg2._attn_implementation = "eager"

    stateful = StatefulGPT2ForCausalLM(
        model_name=MODEL_NAME,
        max_context_size=MAX_CONTEXT_SIZE,
        batch_size=1,
    ).to(device).eval()

    def greedy_stateless(prompt_ids: torch.Tensor, max_new_tokens: int = 16) -> torch.Tensor:
        # prompt_ids: [1, T]
        ids = prompt_ids.clone()
        # Use an all-ones attention mask, since we do not pad in this debug routine.
        attn = torch.ones_like(ids, device=device)

        for _ in range(max_new_tokens):
            with torch.no_grad():
                out = base_model(input_ids=ids, attention_mask=attn)
            logits = out.logits[:, -1, :]  # [1, vocab]
            next_id = int(torch.argmax(logits, dim=-1))
            ids = torch.cat(
                [ids, torch.tensor([[next_id]], dtype=ids.dtype, device=ids.device)],
                dim=1,
            )
            # Rebuild attention mask for the extended sequence (still all ones).
            attn = torch.ones_like(ids, device=device)

        return ids

    def greedy_stateful(prompt_ids: torch.Tensor, max_new_tokens: int = 16) -> torch.Tensor:
        # state 초기화。
        # state の初期化。
        # Initialize state.
        stateful.keyCache.zero_()
        stateful.valueCache.zero_()
        stateful.pastLen.zero_()

        attn = torch.ones_like(prompt_ids, device=device)
        ids = prompt_ids.clone()

        with torch.no_grad():
            # (1) 프롬프트 전체를 한 번 넣어서 KV 캐시 초기화。
            # (1) プロンプト全体を一度入れて KV キャッシュを初期化。
            # (1) Feed the whole prompt once to initialize the KV cache.
            logits = stateful(input_ids=ids, attention_mask=attn)  # [1, 1, vocab]
            next_id = int(torch.argmax(logits[:, -1, :], dim=-1))
            ids = torch.cat(
                [ids, torch.tensor([[next_id]], dtype=ids.dtype, device=ids.device)],
                dim=1,
            )

            # (2) 이후에는 마지막 토큰만 [1,1]로 넣으면서 한 토큰씩 생성。
            # (2) 以降は最後のトークンのみ [1,1] で入れて一トークンずつ生成。
            # (2) Afterwards, feed only the last token [1,1] to generate one token at a time.
            for _ in range(max_new_tokens - 1):
                last = ids[:, -1:]
                attn_step = torch.ones_like(last, device=device)
                logits = stateful(input_ids=last, attention_mask=attn_step)  # [1, 1, vocab]
                next_id = int(torch.argmax(logits[:, -1, :], dim=-1))
                ids = torch.cat(
                    [ids, torch.tensor([[next_id]], dtype=ids.dtype, device=ids.device)],
                    dim=1,
                )
        return ids

    test_texts = [
        "ニホンゴ",
        "カンコクゴヲベンキョウスル",
        "オハヨウゴザイマス",
    ]

    for text in test_texts:
        print("=" * 80)
        print("TEXT:", text)
        enc = tokenizer.encode(text, return_tensors="pt").to(device)

        ids_stateless = greedy_stateless(enc)
        ids_stateful = greedy_stateful(enc)

        dec_stateless = tokenizer.decode(ids_stateless[0], skip_special_tokens=True)
        dec_stateful = tokenizer.decode(ids_stateful[0], skip_special_tokens=True)

        print("[Stateless] decoded:", dec_stateless)
        print("[Stateful ] decoded:", dec_stateful)
        print("Same token ids?:", torch.equal(ids_stateless, ids_stateful))
        print("Stateless ids:", ids_stateless.tolist())
        print("Stateful  ids:", ids_stateful.tolist())
        print("=" * 80)

def convert_model(model_name: str, output_path: str) -> None:
    # 토크나이저는 그대로。
    # トークナイザーはそのまま。
    # Tokenizer remains unchanged.
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Torch 쪽 stateful 래퍼。
    # Torch 側の stateful ラッパー。
    # Torch side stateful wrapper.
    torch_model = StatefulGPT2ForCausalLM(
        model_name=model_name,
        max_context_size=MAX_CONTEXT_SIZE,
        batch_size=BATCH_SIZE,
    ).eval()

    kv_cache_shape = torch_model.kv_cache_shape
    vocab_size = torch_model.model.config.vocab_size

    # trace용 예제 입력 (길이는 아무거나, 상한은 MAX_CONTEXT_SIZE)。
    # trace 用のサンプル入力（長さは任意、上限は MAX_CONTEXT_SIZE）。
    # Example input for tracing (any length, upper bound is MAX_CONTEXT_SIZE).
    example_input_ids = torch.zeros((BATCH_SIZE, 4), dtype=torch.long)
    example_attention_mask = torch.ones((BATCH_SIZE, 4), dtype=torch.long)

    # Only trace with true inputs (no explicit state tensors)。
    # 実際の入力のみで trace（明示的な state テンソルなし）。
    # Only trace with true inputs (no explicit state tensors).
    traced_model = torch.jit.trace(
        torch_model,
        (example_input_ids, example_attention_mask),
        check_trace=False,
    )

    # Core ML 입력/출력/상태 스펙 (Mistral export.py 스타일)。
    # Core ML 入出力/状態仕様（Mistral export.py スタイル）。
    # Core ML input/output/state spec (Mistral export.py style).
    query_length = ct.RangeDim(lower_bound=1, upper_bound=MAX_CONTEXT_SIZE, default=1)

    input_ids_type = ct.TensorType(
        shape=(BATCH_SIZE, query_length),
        dtype=np.int32,
        name="input_ids",
    )
    attention_mask_type = ct.TensorType(
        shape=(BATCH_SIZE, query_length),
        dtype=np.int32,
        name="attention_mask",
    )

    inputs: List[ct.TensorType] = [
        input_ids_type,
        attention_mask_type,
    ]

    outputs: List[ct.TensorType] = [
        ct.TensorType(
            dtype=np.float16,
            name="logits",
        ),
    ]

    # state 텐서를 위한 익명 TensorType 정의, ct.StateType에서 사용。
    # state テンソル用の匿名 TensorType を定義し、ct.StateType で利用。
    # Define anonymous TensorTypes for state tensors, use in ct.StateType
    keycache_state_type = ct.TensorType(
        shape=kv_cache_shape,
        dtype=np.float16,
    )
    valuecache_state_type = ct.TensorType(
        shape=kv_cache_shape,
        dtype=np.float16,
    )
    pastlen_state_type = ct.TensorType(
        shape=(1,),
        dtype=np.float16,
    )

    # PyTorch 쪽 register_buffer 이름과 동일한 Core ML state를 정의한다。
    # PyTorch 側の register_buffer 名と対応する Core ML の state を定義する。
    # Define Core ML states that correspond to the PyTorch register_buffer names.
    states: List[ct.StateType] = [
        ct.StateType(
            wrapped_type=keycache_state_type,
            name="keyCache",
        ),
        ct.StateType(
            wrapped_type=valuecache_state_type,
            name="valueCache",
        ),
        ct.StateType(
            wrapped_type=pastlen_state_type,
            name="pastLen",
        ),
    ]

    mlmodel_fp16: ct.MLModel = ct.convert(
        traced_model,
        inputs=inputs,
        outputs=outputs,
        states=states,
        source="pytorch",
        minimum_deployment_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
        skip_model_load=True,
    )

    # Set feature descriptions for stateful model
    mlmodel_fp16.input_description["input_ids"] = (
        "Input token IDs for the stateful zenz-v1 language model."
        "Shape: [batch, query_length] with Int32 values."
    )
    mlmodel_fp16.input_description["attention_mask"] = (
        "Attention mask for the input tokens. 1 for valid tokens, 0 for padding."
        "Shape: [batch, query_length] with Int32 values."
    )
    mlmodel_fp16.output_description["logits"] = (
        "Unnormalized next-token logits for each vocabulary token, taking into account the current KV cache state."
        "Shape: [batch, query_length, vocab_size]."
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
    mlmodel_fp16.version = "3.1.0-stateful"

    # Save the base FP16 stateful model
    mlmodel_fp16.save(output_path)

    # iOS 18용 8bit 팔레타이제이션 압축
    # iOS 18 向けの 8bit パレット化圧縮
    # 8-bit palettization compression for iOS 18
    op_config = OpPalettizerConfig(nbits=8)
    opt_config = OptimizationConfig(global_config=op_config)
    compressed = palettize_weights(
        mlmodel_fp16,
        opt_config,
    )
    compressed_path = output_path.replace(".mlpackage", "-8bit.mlpackage")
    compressed.save(compressed_path)

if __name__ == "__main__":
    # Step 1: 디버그 모드 - stateful vs stateless 비교。
    # Step 1: デバッグモード - stateful vs stateless の比較。
    # Step 1: Debug mode - compare stateful vs stateless.
    # debug_compare_stateful_vs_stateless()

    # 이후 Core ML 변환을 다시 돌리고 싶으면 아래 주석을 풀어서 사용하세요。
    # Core ML 変換を再実行したい場合は、以下のコメントを外して使ってください。
    # If you want to rerun Core ML conversion, uncomment and use the line below.
    convert_model(MODEL_NAME, "zenz_v3.1_stateful.mlpackage")