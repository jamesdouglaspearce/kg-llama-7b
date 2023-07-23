import torch
import os
from peft import PeftModel
import transformers
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          LlamaTokenizer, StoppingCriteria,
                          StoppingCriteriaList, TextIteratorStreamer)
os.system('cp ../tokens/huggingface.txt /home/ubuntu/.cache/huggingface/token')


model_name = "meta-llama/Llama-2-7b-hf"
checkpoint_path = "../model_artifacts/checkpoint-4900"

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

test = ["Jason is John's father's brother. His is 17 years old. -> ",
        "You have modified the pretrained model configuration to control generation. This is a deprecated strategy to control generation and will be removed soon, in a future version. -> ",
        "HD 115600 has an apparent magnitude of 8.22. -> ",
        "Canta Comigo is a Brazilian television channel. -> ",
        "The Combat: Woman Pleading for the Vanquished is made of oil paint. -> "]

sequences = pipeline(
    test,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200,
)
for seq in sequences:
    for sub in seq:
        print(f"Result: {sub['generated_text']}")


