import os
from typing import List

import coremltools as ct
import numpy as np
import torch
from transformers import GPT2LMHeadModel, AutoTokenizer
from transformers.models.gpt2 import modeling_gpt2 as gpt2_mod

# Disable tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_NAME = "Miwa-Keita/zenz-v1-checkpoints"
MAX_CONTEXT_SIZE = 128
BATCH_SIZE = 1


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

    batch_size = None
    q_len = None

    input_ids = kwargs.get("input_ids", None)
    inputs_embeds = kwargs.get("inputs_embeds", None)
    input_shape = kwargs.get("input_shape", None)
    past_kv_len = int(kwargs.get("past_key_values_length", 0))

    if input_ids is not None and isinstance(input_ids, torch.Tensor):
        batch_size, q_len = int(input_ids.shape[0]), int(input_ids.shape[1])
        device = input_ids.device
    elif inputs_embeds is not None and isinstance(inputs_embeds, torch.Tensor):
        batch_size, q_len = int(inputs_embeds.shape[0]), int(inputs_embeds.shape[1])
        device = inputs_embeds.device
    elif input_shape is not None:
        batch_size, q_len = int(input_shape[0]), int(input_shape[1])
        device = kwargs.get("device", torch.device("cpu"))
    elif len(args) >= 2:
        candidate = args[1]
        if isinstance(candidate, torch.Tensor):
            batch_size, q_len = int(candidate.shape[0]), int(candidate.shape[1])
            device = candidate.device
        elif isinstance(candidate, (tuple, list)) and len(candidate) >= 2:
            batch_size, q_len = int(candidate[0]), int(candidate[1])
            device = kwargs.get("device", torch.device("cpu"))
        else:
            batch_size, q_len = 1, int(candidate)
            device = kwargs.get("device", torch.device("cpu"))
    else:
        batch_size, q_len = 1, 1
        device = kwargs.get("device", torch.device("cpu"))

    dtype = kwargs.get("dtype", torch.float32)
    k_len = q_len + past_kv_len

    # (q_len, k_len) lower-triangular mask: 0 for allowed, -inf for future
    mask = torch.zeros((q_len, k_len), dtype=dtype, device=device)
    future = torch.triu(
        torch.ones((q_len, k_len), dtype=torch.bool, device=device),
        diagonal=1,
    )
    mask = mask.masked_fill(future, float("-inf"))

    # (1, 1, q_len, k_len) -> (batch, 1, q_len, k_len)
    mask = mask.unsqueeze(0).unsqueeze(0)
    if batch_size is not None:
        mask = mask.expand(batch_size, 1, q_len, k_len)

    return mask


# GPT-2 내부 causal mask를 TorchScript 친화 버전으로 교체
gpt2_mod.create_causal_mask = _patched_create_causal_mask


class StatelessGPT2ForCausalLM(torch.nn.Module):
    """
    Stateless Core ML용 GPT-2 래퍼.

    - attention_mask / past_key_values 없이 input_ids → logits만 사용
    - TorchScript + Core ML 변환을 위해 attention 구현을 "eager"로 강제
    """

    def __init__(self, model_name: str):
        super().__init__()
        self.model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained(model_name)
        cfg = self.model.config
        # sdpa + vmap 경로를 피하기 위해 eager attention으로 강제
        if hasattr(cfg, "attn_implementation"):
            cfg.attn_implementation = "eager"
        elif hasattr(cfg, "_attn_implementation"):
            cfg._attn_implementation = "eager"
        self.model.eval()

    @torch.no_grad()
    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, use_cache=False)
        return outputs.logits


def convert_model(output_path: str) -> None:
    # 토크나이저 로드 + JSON 저장 (Swift에서 쓸 수 있도록)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained("./tokenizer/")

    # Torch 쪽 stateless 래퍼
    torch_model = StatelessGPT2ForCausalLM(MODEL_NAME).eval()

    # trace 예제 입력
    example_input_ids = torch.zeros((BATCH_SIZE, 4), dtype=torch.long)

    traced_model = torch.jit.trace(
        torch_model,
        (example_input_ids,),
        check_trace=False,
    )

    # Core ML 입력/출력 스펙 (RangeDim으로 가변 길이 시퀀스)
    query_length = ct.RangeDim(lower_bound=1, upper_bound=MAX_CONTEXT_SIZE, default=1)

    inputs: List[ct.TensorType] = [
        ct.TensorType(
            name="input_ids",
            shape=(BATCH_SIZE, query_length),
            dtype=np.int32,
        ),
    ]

    outputs: List[ct.TensorType] = [
        ct.TensorType(
            name="logits",
            dtype=np.float16,
        ),
    ]

    mlmodel_fp16: ct.MLModel = ct.convert(
        traced_model,
        inputs=inputs,
        outputs=outputs,
        source="pytorch",
        minimum_deployment_target=ct.target.iOS16,
        compute_units=ct.ComputeUnit.ALL,
        skip_model_load=True,
    )

    # Xcode에 보이는 설명들
    mlmodel_fp16.input_description["input_ids"] = (
        "Input token IDs for the zenz-v1 language model."
        "Shape: [batch, sequence_length] with Int32 values."
    )
    mlmodel_fp16.output_description["logits"] = (
        "Unnormalized next-token logits for each vocabulary token."
        "Shape: [batch, sequence_length, vocab_size]."
    )

    # 메타데이터
    mlmodel_fp16.author = (
        "Original model: Miwa-Keita\n"
        "Core ML conversion: Skyline-23 (Buseong Kim)"
    )
    mlmodel_fp16.license = "CC-BY-SA 4.0"
    mlmodel_fp16.short_description = (
        "zenz-v1 GPT-2–style causal language model converted to Core ML.\n"
        "Given input token IDs, it produces next-token logits for text generation on Apple devices."
    )
    mlmodel_fp16.user_defined_metadata["com.apple.coreml.model.preview.type"] = "textGenerator"
    mlmodel_fp16.version = "1.0.0"

    mlmodel_fp16.save(output_path)


if __name__ == "__main__":
    convert_model("zenz_v1.mlpackage")