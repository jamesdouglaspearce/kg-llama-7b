from transformers import AutoTokenizer, BitsAndBytesConfig, TrainingArguments, AutoModelForCausalLM
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from tqdm import tqdm
from trl import SFTTrainer
import transformers
import torch
from datasets import load_dataset
import pandas as pd
import json
import os
from utils.print import trainable_params

# Save huggingface token
os.system('cp ../tokens/huggingface.txt /home/ubuntu/.cache/huggingface/token')

#model_name = "meta-llama/Llama-2-7b-chat-hf"
model_name = "meta-llama/Llama-2-7b-hf"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=BitsAndBytesConfig(load_in_4bit=False, load_in_8bit=True),
    device_map={"": 0},
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    use_auth_token=True
)

#model.gradient_checkpointing_enable()
#model = prepare_model_for_kbit_training(model)

# Print trainable parameters
print('Trainable params before LoRA')
trainable_params(model)

data_path = '../data'

files = {'train': os.path.join(data_path, 'processed-train.txt'),
         'validation': os.path.join(data_path, 'processed-validation.txt'),
         'test': os.path.join(data_path, 'processed-test.txt')}

data = load_dataset('text', data_files=files)

output_dir = '../model_artifacts'
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=3,
    gradient_accumulation_steps=1,
    learning_rate=1.41e-5,
    logging_steps=10,
    gradient_checkpointing=False,
    save_steps=100,
)

peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=["q_proj","v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
)

#model = get_peft_model(model, peft_config)
#model.config.use_cache = False

print('Trainable params after LoRA')
trainable_params(model)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=data['train'],
    dataset_text_field='text',
    peft_config=peft_config,
    max_seq_length=2048,
    packing=True,
)

trainer.train()