import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# Load pre-trained GPT-2 model and tokenizer
model_name = 'gpt2'  # or 'gpt2-medium', 'gpt2-large', etc. for different sizes
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Load and preprocess your text data
train_file = 'legal1.txt'

# Define a custom dataset using TextDataset
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=train_file,
    block_size=128  # set the block size as desired
)

# Create a data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # set to True if you want to use masked language modeling
)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./output',  # specify the output directory for saving model checkpoints
    overwrite_output_dir=True,
    num_train_epochs=3,  # specify the number of epochs for training
    per_device_train_batch_size=4,  # specify the batch size per GPU
    save_steps=10_000,  # specify the number of steps for saving model checkpoints
    save_total_limit=2,  # specify the maximum number of model checkpoints to save
)

# Create the Trainer for fine-tuning
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset
)

# Move model to CUDA
trainer.model.to('cuda')

# Fine-tune the model with batches iteratively sent to CUDA
for epoch in range(training_args.num_train_epochs):
    epoch_loss = 0.0
    num_batches = len(train_dataset) // training_args.per_device_train_batch_size
    for i in range(0, len(train_dataset), training_args.per_device_train_batch_size):
        batch = train_dataset[i: i + training_args.per_device_train_batch_size]
        inputs = data_collator.collate_batch([batch])
        inputs = {k: v.to('cuda') for k, v in inputs.items()}  # send inputs to CUDA
        outputs = trainer.model(**inputs, labels=inputs['labels'])
        loss = outputs.loss
        epoch_loss += loss.item()
        loss.backward()
        trainer.optimizer.step()
        trainer.optimizer.zero_grad()
    print(f'Epoch {epoch + 1}, Loss: {epoch_loss / num_batches:.4f}')

# Save the fine-tuned model
trainer.save_model('./output/fine-tuned-model')

# Load the fine-tuned model
fine_tuned_model = GPT2LMHeadModel.from_pretrained('./output/fine-tuned-model')

# Prompt the model to generate text and extract required information
prompt = "Once upon a time, in a land far, far away, there was a"
input_ids = tokenizer.encode(prompt, return_tensors='pt')
input_ids = input_ids.to('cuda')  # send input_ids to CUDA
output = fine_tuned_model.generate(input_ids, max_length=100, num_return_sequences=1)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
# Extract required information from the generated text
required_info = generated_text.split("your required information marker")[1]  # Update with the marker you used
print("Extracted information:", required_info)