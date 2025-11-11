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
    minimum_deployment_target=ct.target.iOS15,
)

# 변환된 모델 저장 / Save the converted model
mlmodel.save("zenz_v1.mlpackage")

print("모델이 성공적으로 변환되고 저장되었습니다: zenz_v1.mlpackage / Model successfully converted and saved as: zenz_v1.mlpackage")
print("토크나이저가 tokenizer/ 폴더에 저장되었습니다 / Tokenizer successfully saved into tokenizer/ directory")
