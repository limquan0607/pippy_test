import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch.optim as optim

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set model to training mode
model.train()

# Define training parameters
batch_size = 2
sequence_length = 10
vocab_size = model.config.vocab_size  # typically 50257 for GPT-2

# Create a random tensor as input data (random token IDs)
random_input_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, sequence_length))
# For language modeling, we typically use the same tokens as labels
labels = random_input_ids.clone()

# Move model and data to device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
random_input_ids = random_input_ids.to(device)
labels = labels.to(device)

# Define an optimizer
optimizer = optim.Adam(model.parameters(), lr=5e-5)

# A simple training step
optimizer.zero_grad()
outputs = model(random_input_ids, labels=labels)
loss = outputs.loss
print("Loss:", loss.item())

loss.backward()
optimizer.step()

print("Training step completed.")
