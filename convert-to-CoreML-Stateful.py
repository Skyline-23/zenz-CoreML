import os
from typing import Tuple

import coremltools as ct
import numpy as np
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_NAME = "Miwa-Keita/zenz-v1-checkpoints"
MAX_CONTEXT_SIZE = 256
BATCH_SIZE = 1


class StatefulGPT2(torch.nn.Module):

    def __init__(self, model: GPT2LMHeadModel, max_context_size: int = MAX_CONTEXT_SIZE, batch_size: int = BATCH_SIZE):
        super().__init__()
        self.model = model
        config = self.model.config

        self.num_layers = config.num_hidden_layers
        self.num_heads = config.n_head
        self.head_dim = config.hidden_size // config.n_head

        # KV-cache용 state 텐서 모양
        # [num_layers, batch, num_heads, max_context, head_dim]
        self.kv_cache_shape: Tuple[int, ...] = (
            self.num_layers,
            batch_size,
            self.num_heads,
            max_context_size,
            self.head_dim,
        )

        # Core ML state가 될 버퍼들 (이름이 중요함!)
        self.register_buffer(
            "keyCache",
            torch.zeros(self.kv_cache_shape, dtype=torch.float16),
            persistent=True,
        )
        self.register_buffer(
            "valueCache",
            torch.zeros(self.kv_cache_shape, dtype=torch.float16),
            persistent=True,
        )

    @torch.no_grad()
    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor):
        """현재 버전은 KV-cache state를 실제로 사용하지 않고,
        순수하게 logits만 반환하는 형태야.

        중요한 포인트는:
        - keyCache / valueCache 가 buffer 로 등록되어 있고
        - ct.convert 에서 StateType 으로 노출된다는 점.

        여기서는 tracing 호환성을 위해 attention_mask는 입력으로만 받고,
        실제 Hugging Face GPT-2 호출에는 전달하지 않는다.
        이후 KV-cache를 붙이는 단계에서 attention_mask를 직접 처리하도록 확장할 수 있다.
        """

        # GPT-2는 causal LM이라 기본적으로 attention_mask 없이도 동작한다.
        # masking_utils의 vmap 기반 마스킹 로직을 피하기 위해 여기서는 사용하지 않는다.
        outputs = self.model(input_ids=input_ids)

        # Core ML 변환 파이프라인에서 state가 "unused" 로 간주되면 에러가 나므로,
        # keyCache / valueCache 를 그래프에 가볍게 참조만 해준다.
        # 한 원소만 읽어서 0을 곱한 뒤 logits에 더하므로, 수치적으로는 완전히 no-op 이다.
        key_sample = self.keyCache.reshape(-1)[0].to(outputs.logits.dtype)
        value_sample = self.valueCache.reshape(-1)[0].to(outputs.logits.dtype)
        dummy = (key_sample + value_sample) * 0.0

        return outputs.logits + dummy


def convert_model(model_name: str, output_path: str):
    # 1) 모델 / 토크나이저 로드
    model = GPT2LMHeadModel.from_pretrained(model_name).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 2) 예제 입력 준비 (trace용)
    text = "Example sentence"
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]  # [1, seq_len]
    attention_mask = inputs["attention_mask"]  # [1, seq_len]

    # 3) stateful 래퍼 생성
    stateful_model = StatefulGPT2(model).eval()

    # 4) torch.jit.trace 로 그래프 캡처
    example_inputs = (input_ids, attention_mask)
    traced_model = torch.jit.trace(stateful_model, example_inputs, check_trace=False)

    # 5) Core ML 입력/출력/상태 타입 정의 (Apple 문서 패턴 그대로)
    query_length = ct.RangeDim(lower_bound=1, upper_bound=MAX_CONTEXT_SIZE, default=int(input_ids.shape[1]))

    inputs_spec = [
        ct.TensorType(
            name="input_ids",
            shape=(BATCH_SIZE, query_length),
            dtype=np.int32,  # 토큰 ID
        ),
        ct.TensorType(
            name="attention_mask",
            shape=(BATCH_SIZE, query_length),
            dtype=np.int32,  # 0/1 mask
        ),
    ]

    outputs_spec = [
        ct.TensorType(
            name="logits",
            dtype=np.float32,
        )
    ]

    # register_buffer 이름과 반드시 같아야 함: "keyCache", "valueCache"
    states_spec = [
        ct.StateType(
            wrapped_type=ct.TensorType(
                shape=stateful_model.kv_cache_shape,
                dtype=np.float16,
            ),
            name="keyCache",
        ),
        ct.StateType(
            wrapped_type=ct.TensorType(
                shape=stateful_model.kv_cache_shape,
                dtype=np.float16,
            ),
            name="valueCache",
        ),
    ]

    # 6) Core ML 변환 (stateful 모델)
    mlmodel = ct.convert(
        traced_model,
        inputs=inputs_spec,
        outputs=outputs_spec,
        states=states_spec,
        minimum_deployment_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
    )

    mlmodel.save(output_path)

    print(f"Stateful GPT-2 Core ML model saved as: {output_path}")
    print(f"  - max context: {MAX_CONTEXT_SIZE}")
    print(f"  - kv_cache shape: {stateful_model.kv_cache_shape}")


if __name__ == "__main__":
    convert_model(MODEL_NAME, "zenz_v1_stateful.mlpackage")