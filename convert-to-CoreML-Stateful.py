import torch
from transformers import GPT2LMHeadModel, AutoTokenizer
import coremltools as ct
import numpy as np

# Load the model and tokenizer
model_name = "Miwa-Keita/zenz-v1-checkpoints"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare input data
text = "Example sentence"
inputs = tokenizer(text, return_tensors="pt")

# Save the tokenizer to a JSON file
tokenizer.save_pretrained("./tokenizer/")

# Model tracing
class StatefulModel(torch.nn.Module):
    def __init__(self, model):
        super(StatefulModel, self).__init__()
        self.model = model

        # Register keyCache and valueCache buffers with initial minimal dimensions
        config = model.config
        self.register_buffer('keyCache', torch.zeros(config.n_layer, config.n_head, 0, config.n_embd // config.n_head))
        self.register_buffer('valueCache', torch.zeros(config.n_layer, config.n_head, 0, config.n_embd // config.n_head))

    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.size(0)
        seq_length = input_ids.size(1)
        
        if self.keyCache.size(2) == 0 and self.valueCache.size(2) == 0:
            # Create empty caches with the correct dimensions
            key_cache = torch.zeros(self.model.config.n_layer, batch_size, self.model.config.n_head, 0, self.model.config.n_embd // self.model.config.n_head)
            value_cache = torch.zeros(self.model.config.n_layer, batch_size, self.model.config.n_head, 0, self.model.config.n_embd // self.model.config.n_head)
        else:
            # Concatenate new zeros with past_key_values to match the input sequence length
            new_key_cache = torch.zeros(self.model.config.n_layer, batch_size, self.model.config.n_head, seq_length, self.model.config.n_embd // self.model.config.n_head)
            new_value_cache = torch.zeros(self.model.config.n_layer, batch_size, self.model.config.n_head, seq_length, self.model.config.n_embd // self.model.config.n_head)
            
            key_cache = torch.cat((self.keyCache, new_key_cache), dim=-2)
            value_cache = torch.cat((self.valueCache, new_value_cache), dim=-2)

        past_key_values = (key_cache, value_cache)
        
        outputs = self.model(input_ids, past_key_values=past_key_values, attention_mask=attention_mask)
        return outputs.logits

torch_model = StatefulModel(model).eval()
traced_model = torch.jit.trace(torch_model, [inputs['input_ids'], inputs['attention_mask']])

# Convert the model to CoreML
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
                shape=(model.config.n_layer, model.config.n_head, 0, model.config.n_embd // model.config.n_head),
                dtype=np.float16,
            ),
            name="keyCache",
        ),
        ct.StateType(
            wrapped_type=ct.TensorType(
                shape=(model.config.n_layer, model.config.n_head, 0, model.config.n_embd // model.config.n_head),
                dtype=np.float16,
            ),
            name="valueCache",
        ),
    ],
    minimum_deployment_target=ct.target.iOS18
)

# Save the converted model
mlmodel.save("zenz_v1_cached.mlpackage")

print("Model successfully converted and saved as: zenz_v1_cached.mlpackage")
print("Tokenizer successfully saved as: tokenizer_config.json and vocab.json")