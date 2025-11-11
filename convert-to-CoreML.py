import torch
from transformers import GPT2LMHeadModel, AutoTokenizer
import coremltools as ct
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 모델과 토크나이저 로드 / Load the model and tokenizer
model_name = "Miwa-Keita/zenz-v1-checkpoints"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 입력 데이터 준비 / Prepare input data
text = "Example sentence"
inputs = tokenizer(text, return_tensors="pt")

# 토크나이저를 JSON 파일로 저장 / Save the tokenizer to a JSON file
tokenizer.save_pretrained("./tokenizer/")

# 모델 추적 (Tracing)용 래퍼
# attention_mask를 사용하지 않고, input_ids만 받아서 logits만 반환하도록 단순화
class TracedModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        # GPT2는 기본적으로 causal LM이라 attention_mask 없이도 동작함
        outputs = self.model(input_ids=input_ids)
        return outputs.logits

traced_model_wrapper = TracedModelWrapper(model).eval()

# trace 시에도 input_ids만 사용
example_input_ids = inputs["input_ids"]
traced_model = torch.jit.trace(traced_model_wrapper, (example_input_ids,))

# 모델을 CoreML로 변환 / Convert the model to CoreML
mlmodel = ct.convert(
    traced_model,
    inputs=[
        ct.TensorType(
            name="input_ids",
            shape=(1, ct.RangeDim(1, 256)),  # 시퀀스 길이 상한 256 / Max sequence length 256 / 上限を256に設定
        ),
    ],
    outputs=[
        ct.TensorType(name="logits"),
    ],
    minimum_deployment_target=ct.target.iOS15,
)

# Set feature descriptions (these show up as comments in Xcode)
mlmodel.input_description["input_ids"] = (
    "Input token IDs for the zenz-v1 language model.\n"
    "Shape: [batch, sequence_length] with Int32 values."
)
mlmodel.output_description["logits"] = (
    "Unnormalized next-token logits for each vocabulary token.\n"
    "Shape: [batch, sequence_length, vocab_size]."
)

# Set model author name (original + Core ML conversion)
mlmodel.author = (
    "Original model: Miwa-Keita\n"
    "Stateful Core ML conversion: Skyline-23 (Buseong Kim)"
)

# Set the license of the model
mlmodel.license = (
    "CC-BY-SA 4.0"
)

# Set a short description for the Xcode UI
mlmodel.short_description = (
    "zenz-v1 GPT-2–style causal language model converted to Core ML.\n"
    "Given input token IDs, it produces next-token logits for text generation on Apple devices."
)

# Set the preview type (custom text-oriented tag)
mlmodel.user_defined_metadata["com.apple.coreml.model.preview.type"] = "textGenerator"

# Set a version for the model
mlmodel.version = "1.0.0"

# 변환된 모델 저장 / Save the converted model
mlmodel.save("zenz_v1.mlpackage")