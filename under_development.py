import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Define the dataset class
class TextDataset(Dataset):
    def __init__(self, file_path, block_size):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.examples = f.readlines()
        self.block_size = block_size
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx].strip()
        input_ids = self.tokenizer.encode(example, add_special_tokens=True, truncation=True, max_length=self.block_size)
        return torch.tensor(input_ids)

# Define the fine-tuning function
def fine_tune(num_epochs, train_dataloader, model, tokenizer, learning_rate=1e-4, warmup_steps=5000):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Set optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=len(train_dataloader)*num_epochs)

    # Set loss function
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # Train the model
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for i, batch in enumerate(train_dataloader):
            # Prepare batch
            batch = batch.to(device)
            input_ids = batch[:, :-1]
            labels = batch[:, 1:]

            # Forward pass
            outputs = model(input_ids, labels=labels)
            loss = loss_fn(outputs.logits.view(-1, tokenizer.vocab_size), labels.view(-1))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            # Print progress every 10 batches
            if i % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs} | Batch {i+1}/{len(train_dataloader)} | Loss {loss.item()}')

        # Print average loss for the epoch
        avg_loss = total_loss / len(train_dataloader)
        print(f'Epoch {epoch+1}/{num_epochs} | Average Loss {avg_loss}')

    # Save the fine-tuned model
    model.save_pretrained('model')
    tokenizer.save_pretrained('save_model')

def generate_response(query_prompt, tokenizer, model, max_length=50):
    # tokenize the query prompt and convert to tensor
    input_ids = torch.tensor(tokenizer.encode(query_prompt)).unsqueeze(0).to(model.device)
    
    # generate a response using the model
    output = model.generate(
        input_ids=input_ids,
        do_sample=True,
        max_length=max_length,
        top_p=0.92,
        top_k=0
    )
    
    # decode the generated response and return as a string
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response




# Load the data
train_data = TextDataset('legal1.txt', block_size=1024)
train_dataloader = DataLoader(train_data, batch_size=2, shuffle=True)

# Load the tokenizer and GPT-2 model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForSequenceClassification.from_pretrained("gpt2")

# finetune the model
fine_tune(20, train_dataloader, model, tokenizer)

# start the main loop for generating responses
while True:
    query_prompt = input("Enter a query prompt: ")
    response = generate_response(query_prompt, tokenizer, model)
    print(response)
