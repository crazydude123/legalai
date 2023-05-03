import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Check if a CUDA device is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the fine-tuned GPT-2 model and tokenizer onto the device
model = GPT2LMHeadModel.from_pretrained('fine_tuned_gpt2').to(device)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Generate text using the model and a prompt
prompt = "The law says these cool things:"
encoded_prompt = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)
output_sequences = model.generate(
    input_ids=encoded_prompt,
    max_length=1024,
    temperature=0.7,
    top_k=0,
    top_p=0.9,
    repetition_penalty=1.0,
    do_sample=True,
    num_return_sequences=1,
)

# Decode and print the generated text
generated_sequence = output_sequences[0].tolist()
text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
print(text)
