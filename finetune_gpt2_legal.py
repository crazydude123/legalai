import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load pre-trained GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Load and preprocess the text document
with open('legal1.txt', 'r', encoding="utf-8") as f:
    text = f.read()

# Split the text into chunks of 1024 tokens or less
chunk_size = 1024
chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Tokenize the chunks and convert to PyTorch tensors
input_ids = []
for chunk in chunks:
    encoded_chunk = tokenizer.encode(chunk, add_special_tokens=True)
    input_ids.append(torch.tensor(encoded_chunk).unsqueeze(0))

# Fine-tune the model on your text document
model.train()
model.resize_token_embeddings(len(tokenizer))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(5):
    total_loss = 0
    for chunk in input_ids:
        outputs = model(chunk, labels=chunk)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    print(f'Epoch {epoch+1} loss: {total_loss/len(input_ids)}')

# Save the fine-tuned model
model.save_pretrained('fine_tuned_gpt2')
