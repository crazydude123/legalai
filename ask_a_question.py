import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Check if a CUDA device is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('fine_tuned_gpt2').to(device)

# Set up the prompt and its attention mask
prompt = "What is the meaning of life?"
input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
attention_mask = torch.ones_like(input_ids)

# Generate text based on the prompt
output = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_length=1024,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.0,
    num_return_sequences=1,
)


# Decode and print the generated text
generated_sequence = output[0].tolist()
text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
print(text)
