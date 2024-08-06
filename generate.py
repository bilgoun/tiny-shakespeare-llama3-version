import torch
import random
from decoder_only_model import Transformer, ModelArgs, Tokenizer, load_and_prepare_data, get_batch, estimate_loss

data, tokenizer, train_data, val_data = load_and_prepare_data()
if data is None:
    exit(1)
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model parameters
args = ModelArgs(
    dim=384,
    n_heads=6,
    vocab_size=tokenizer.n_words,
    max_seq_len=256,
)

# Initialize the model
model = Transformer(args).to(device)

model.load_state_dict(torch.load("shakespeare_transformer.pth", map_location=device))

model.eval()

#We don't have tokenized_datasets anymore, we use val_data directly
random_idx = random.randint(0, len(val_data) - 101)
context = val_data[random_idx:random_idx+100].unsqueeze(0).to(device)

# Generate text
with torch.no_grad():
    generated = model.generate(context, max_new_tokens=100)

# Decode and print the original context and the generated text
original_text = tokenizer.decode(context[0].tolist())
generated_text = tokenizer.decode(generated[0].tolist())

print("Original context from validation set:")
print(original_text)
print("\nGenerated continuation:")
print(generated_text[len(original_text):]) 