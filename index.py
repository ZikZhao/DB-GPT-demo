# This script fine-tunes the Qwen3-0.6B model on the Spider dataset using LoRA and 4-bit quantization.

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# import necessary libraries
from socket import gethostname
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, \
                         DataCollatorForLanguageModeling, Trainer, TrainingArguments

# load model
MODEL_ID = "Qwen/Qwen3-0.6B"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                         bnb_4bit_quant_type="nf4",
                                         bnb_4bit_use_double_quant=True,
                                         bnb_4bit_compute_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=quantization_config)

# convert to peft model
peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                         inference_mode=False,
                         r=8,
                         lora_alpha=32,
                         lora_dropout=0.1,
                         target_modules=["q_proj", "v_proj"],)
model = get_peft_model(model, peft_config).to(device)
model.print_trainable_parameters()
tokenizer.pad_token = tokenizer.eos_token

# function to convert query token id sequence to padded label sequence
def label_query(question, query):
    windows = question.unfold(0, len(query), 1)
    position = (windows == torch.tensor(query).unsqueeze(0)).all(dim=1).nonzero().squeeze().item()
    return [-100] * position + query + [-100] * (512 - len(query) - position)

# function to preprocess the dataset
def preprocess(examples):
    chat = [[
        {"role": "system", "content": "You are a database engine.\nAfter review, OUTPUT ONLY the full corrected SQL, with no extra text ot explanation."},
        {"role": "user", "content": question},
        {"role": "assistant", "content": query}
    ] for question, query in zip(examples["question"], examples["query"])]
    input_ids = tokenizer.apply_chat_template(chat,
                                              padding="max_length",
                                              return_tensors="pt",
                                              truncation=True,
                                              max_length=512)
    attention_masks = input_ids.ne(tokenizer.pad_token_id).int()
    tokenized_query = tokenizer(examples["query"])["input_ids"]
    labels = [label_query(question, query) for question, query in zip(input_ids, tokenized_query)]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels,
    }

# load and preprocess the dataset
train_dataset = load_dataset("xlangai/spider", split="train").map(preprocess, batched=True, batch_size=64)
eval_dataset = load_dataset("xlangai/spider", split="validation").map(preprocess, batched=True, batch_size=64)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)

# configure training arguments
training_args = TrainingArguments(
    logging_dir=f"./results/{gethostname()}/logs",
    logging_first_step=True,
    logging_steps=20,
    report_to="tensorboard",
    save_strategy="steps",
    save_steps=80,
    eval_strategy="steps",
    eval_steps=80,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=3e-4,
    num_train_epochs=1,
    bf16=True,
    tf32=True,
    label_names=["labels"],
    optim="adamw_bnb_8bit",
    dataloader_num_workers=4,
    dataloader_prefetch_factor=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    output_dir=f"./results/{gethostname()}/output"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator
)

trainer.train()
