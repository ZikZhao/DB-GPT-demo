import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import peft

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B").to(device)

model = peft.PeftModel.from_pretrained(model, "./output/checkpoint-320/", device_map="auto")
model = model.merge_and_unload().to(device)
tokenizer.pad_token = tokenizer.eos_token

while True:
    user_input = input(">>> ")
    template = [
        {"role": "system", "content": "You are a database engine.\nAfter review, OUTPUT ONLY the full corrected SQL, with no extra text ot explanation."},
        {"role": "user", "content": user_input}
    ]
    input_ids = tokenizer.apply_chat_template(template,
                                              padding="max_length",
                                              return_tensors="pt",
                                              truncation=True,
                                              max_length=512,
                                              add_generation_prompt=True,
                                              return_tensor="pt")
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
            max_new_tokens=1000,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    print(tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True))