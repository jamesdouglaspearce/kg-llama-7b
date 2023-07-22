import torch
import os
from peft import PeftModel
import transformers
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          LlamaTokenizer, StoppingCriteria,
                          StoppingCriteriaList, TextIteratorStreamer)
os.system('cp ../tokens/huggingface.txt /home/ubuntu/.cache/huggingface/token')


model_name = "meta-llama/Llama-2-7b-hf"
checkpoint_path = "../model_artifacts/checkpoint-1000"

print(f"Starting to load the model {model_name} into memory")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    #load_in_4bit=True,
    torch_dtype=torch.bfloat16,
    device_map={"": 0}
)
model = PeftModel.from_pretrained(model, checkpoint_path)
model = model.merge_and_unload()
print(f"Successfully loaded the model {model_name} into memory")

tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
    tokenizer=tokenizer,
)

#tok = LlamaTokenizer.from_pretrained(model_name)
#tok.bos_token_id = 1
#stop_token_ids = [0]

sequences = pipeline(
    "Jason is John's father ->",
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")


