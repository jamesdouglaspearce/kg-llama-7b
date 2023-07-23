from transformers import AutoTokenizer, BitsAndBytesConfig, TrainingArguments, AutoModelForCausalLM, DataCollatorForLanguageModeling
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from tqdm import tqdm
from trl import SFTTrainer
import transformers
import torch
from datasets import load_dataset
import pandas as pd
import json
import os
from utils.utils import print_trainable_parameters, find_all_linear_names

# Save huggingface token
os.system('cp ../tokens/huggingface.txt /home/ubuntu/.cache/huggingface/token')

#model_name = "meta-llama/Llama-2-7b-chat-hf"
model_name = "meta-llama/Llama-2-7b-hf"

bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map={"": 0},
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    use_auth_token=True
)

#model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# Print trainable parameters
print('Trainable params before LoRA')
print_trainable_parameters(model)
print(model.get_memory_footprint())

data_path = '../data'
files = {'train': os.path.join(data_path, 'processed-train.txt'),
         #'validation': os.path.join(data_path, 'processed-validation.txt'),
         #'test': os.path.join(data_path, 'processed-test.txt')}
         }

data = load_dataset('text', data_files=files)

output_dir = '../model_artifacts'
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=2e-4,
    logging_steps=10,
    #gradient_checkpointing=False,
    fp16=True,
    optim="paged_adamw_8bit",
    save_steps=100,
)

modules = find_all_linear_names(model)
print(f'Target modules for QLoRA: {modules}')
peft_config = LoraConfig(
    r=16,  # dimension of the updated matrices
    lora_alpha=64,  # parameter for scaling
    target_modules=modules,
    lora_dropout=0.05,  # dropout probability for layers
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)
model.config.use_cache = False

print('Trainable params after LoRA')
print_trainable_parameters(model)
print(model.get_memory_footprint())

tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
tokenizer.pad_token = tokenizer.eos_token

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=data['train'],
    dataset_text_field='text',
    peft_config=peft_config,
    max_seq_length=1024,
    packing=True,
    #data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

trainer.train()