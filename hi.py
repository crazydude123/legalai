import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cpu")

# Open the text document and read the contents
with open("legal1.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Encode the text document using the tokenizer
input_ids = tokenizer.encode(text, return_tensors="pt")

# Move the input tensor to the device
input_ids = input_ids.to(device)

# Move the model to the device
model.to(device)

# Fine-tune the model
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

for epoch in range(20):
    # Zero out the gradients
    optimizer.zero_grad()

    # Forward pass through the model
    outputs = model(input_ids)

    # Compute the loss
    loss = outputs.loss

    # Backward pass and optimization step
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1} loss: {loss.item()}")

# Save the fine-tuned model
model.save_pretrained("fine_tuned_gpt2")
tokenizer.save_pretrained("fine_tuned_gpt2")
