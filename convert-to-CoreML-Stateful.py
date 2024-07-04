import torch
from transformers import GPT2LMHeadModel, AutoTokenizer
import coremltools as ct
import numpy as np

# 모델과 토크나이저 로드 / Load the model and tokenizer
model_name = "Miwa-Keita/zenz-v1-checkpoints"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 입력 데이터 준비 / Prepare input data
text = "Example sentence"
inputs = tokenizer(text, return_tensors="pt")

# 토크나이저를 JSON 파일로 저장 / Save the tokenizer to a JSON file
tokenizer.save_pretrained("./tokenizer/")

# 모델 추적 (Tracing) / Model tracing
class StatefulModel(torch.nn.Module):
    def __init__(self, model):
        super(StatefulModel, self).__init__()
        self.model = model

        # 각 레이어마다 키와 값 캐시를 저장할 텐서를 만듭니다.
        # TODO: - Set Tensor shape
        self.register_buffer("keyCache", )
        self.register_buffer("valueCache", )

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask, self.keyCache, self.valueCache)
        return outputs.logits

torch_model = StatefulModel(model).eval()
traced_model = torch.jit.trace(torch_model, [inputs['input_ids'], inputs['attention_mask']])

# 모델을 CoreML로 변환 / Convert the model to CoreML
mlmodel = ct.convert(
    traced_model,
    inputs = [
        ct.TensorType(name="input_ids", shape=(1, ct.RangeDim(1, 256))),  # 上限を256に設定
        ct.TensorType(name="attention_mask", shape=(1, ct.RangeDim(1, 256)))  # 上限を256に設定
    ],
    outputs = [
        ct.TensorType(dtype=np.float32, name="logits")
    ],
    states = [
        ct.StateType(
            wrapped_type=ct.TensorType(
                shape=(1, ct.RangeDim(1, 256))
            ),
            name="keyCache",
        ),
        ct.StateType(
            wrapped_type=ct.TensorType(
                shape=(1, ct.RangeDim(1, 256))
            ),
            name="valueCache",
        ),
    ],
    minimum_deployment_target=ct.target.iOS18
)

# 변환된 모델 저장 / Save the converted model
mlmodel.save("zenz_v1_cached.mlpackage")

print("모델이 성공적으로 변환되고 저장되었습니다: zenz_v1.mlpackage / Model successfully converted and saved as: zenz_v1.mlpackage")
print("토크나이저가 JSON 파일로 저장되었습니다: tokenizer_config.json 및 vocab.json / Tokenizer successfully saved as: tokenizer_config.json and vocab.json")
