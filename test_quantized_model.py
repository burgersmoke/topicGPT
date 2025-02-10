from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "c:/git/Mistral-7B-Instruct-v0.3-quantized.w4a16"

print('Loading model...')

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast = False)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="cuda",
)

print('Model loaded...')

messages = [
    {"role": "user", "content": "Who are you?"},
    {"role": "assistant", "content": "You are a pirate chatbot who always responds in pirate speak!"},
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))