from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
model_name = 'Qwen/Qwen2.5-1.5B'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,  # Use bfloat16 for Flash Attention 2.0
)
model.save_pretrained('./models/Qwen2.5-1.5B')
tokenizer.save_pretrained('./models/Qwen2.5-1.5B')

