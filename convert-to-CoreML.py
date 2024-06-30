import torch
from transformers import GPT2LMHeadModel, AutoTokenizer
import coremltools as ct

# 모델과 토크나이저 로드 / Load the model and tokenizer
model_name = "Miwa-Keita/zenz-v1-checkpoints"
model = GPT2LMHeadModel.from_pretrained(model_name).eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 입력 데이터 준비 / Prepare input data
text = "Example sentence"
inputs = tokenizer(text, return_tensors="pt")

# 토크나이저를 JSON 파일로 저장 / Save the tokenizer to a JSON file
tokenizer.save_pretrained("./")

# 모델 추적 (Tracing) / Model tracing
class TracedModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(TracedModelWrapper, self).__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

traced_model_wrapper = TracedModelWrapper(model)
traced_model = torch.jit.trace(traced_model_wrapper, (inputs['input_ids'], inputs['attention_mask']))

# 모델을 CoreML로 변환 / Convert the model to CoreML
mlmodel = ct.convert(
    traced_model,
    inputs=[
        ct.TensorType(name="input_ids", shape=(1, ct.RangeDim(1, 256),)),  # 上限を256に設定
        ct.TensorType(name="attention_mask", shape=(1, ct.RangeDim(1, 256),))  # 上限を256に設定
    ],
    minimum_deployment_target=ct.target.iOS15
)

# 변환된 모델 저장 / Save the converted model
mlmodel.save("zenz_v1.mlpackage")

print("모델이 성공적으로 변환되고 저장되었습니다: zenz_v1.mlpackage / Model successfully converted and saved as: zenz_v1.mlpackage")
print("토크나이저가 JSON 파일로 저장되었습니다: tokenizer_config.json 및 vocab.json / Tokenizer successfully saved as: tokenizer_config.json and vocab.json")
