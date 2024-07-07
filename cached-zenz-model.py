import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config

class ZenzModel(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        # Register keyCache and valueCache buffers with initial minimal dimensions
        self.register_buffer('keyCache', torch.zeros(config.n_layer, config.n_head, 0, config.n_embd // config.n_head))
        self.register_buffer('valueCache', torch.zeros(config.n_layer, config.n_head, 0, config.n_embd // config.n_head))

    def forward(self, input_ids, past_key_values=None, attention_mask=None):
        batch_size = input_ids.size(0)
        seq_length = input_ids.size(1)
        
        if past_key_values is None:
            # Create empty caches with the correct dimensions
            key_cache = torch.zeros(self.config.n_layer, batch_size, self.config.n_head, 0, self.config.n_embd // self.config.n_head)
            value_cache = torch.zeros(self.config.n_layer, batch_size, self.config.n_head, 0, self.config.n_embd // self.config.n_head)
            past_key_values = (key_cache, value_cache)
        else:
            # Concatenate new zeros with past_key_values to match the input sequence length
            new_key_cache = torch.zeros(self.config.n_layer, batch_size, self.config.n_head, seq_length, self.config.n_embd // self.config.n_head)
            new_value_cache = torch.zeros(self.config.n_layer, batch_size, self.config.n_head, seq_length, self.config.n_embd // self.config.n_head)
            
            key_cache = torch.cat((past_key_values[0], new_key_cache), dim=-2)
            value_cache = torch.cat((past_key_values[1], new_value_cache), dim=-2)
            past_key_values = (key_cache, value_cache)
        
        output = super().forward(input_ids, past_key_values=past_key_values, attention_mask=attention_mask)
        return output.logits

# Load the model and tokenizer
model_name = "Miwa-Keita/zenz-v1-checkpoints"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
config = GPT2Config.from_pretrained(model_name)
model = ZenzModel.from_pretrained(model_name, config=config)

# Example inference
input_text = "こんにちは"
input_ids = tokenizer.encode(input_text, return_tensors='pt')
output = model(input_ids)
decoded_output = tokenizer.decode(output.argmax(dim=-1)[0], skip_special_tokens=True)
print(decoded_output)