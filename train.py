import os
import matplotlib.pyplot as plt
from model import LanguageModel
from utils import load_and_preprocess, split_data, get_batches
import torch
import torch.nn as nn

# ==========================
# CHOOSE EXPERIMENT TYPE HERE
# ==========================
EXPERIMENT = "bestfit"  # change to "underfit", "overfit", or "bestfit"

# ==========================
# SET CONFIGS FOR EACH CASE
# ==========================
if EXPERIMENT == "underfit":
    embed_size, hidden_size, num_layers, epochs = 64, 64, 1, 5
elif EXPERIMENT == "overfit":
    embed_size, hidden_size, num_layers, epochs = 512, 512, 3, 25
else:  # bestfit
    embed_size, hidden_size, num_layers, epochs = 128, 256, 2, 10

# Create experiment folder
os.makedirs(f"experiments/{EXPERIMENT}", exist_ok=True)

# Load dataset
encoded, word2idx, idx2word = load_and_preprocess("dataset/Pride_and_Prejudice-Jane_Austen.txt")
train_data, val_data, test_data = split_data(encoded)

# Initialize model
vocab_size = len(word2idx)
model = LanguageModel(vocab_size, embed_size, hidden_size, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training setup
seq_len, batch_size = 30, 64
train_losses, val_losses = [], []
for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    for x, y in get_batches(train_data, seq_len, batch_size):
        optimizer.zero_grad()
        output, _ = model(x)
        loss = criterion(output.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    avg_train_loss = total_train_loss / len(train_data)
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for x, y in get_batches(val_data, seq_len, batch_size):
            output, _ = model(x)
            val_loss = criterion(output.view(-1, vocab_size), y.view(-1))
            total_val_loss += val_loss.item()
    avg_val_loss = total_val_loss / len(val_data)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
# Save plot and model
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title(f'Training vs Validation Loss - {EXPERIMENT.capitalize()}')
plt.legend()
plt.savefig(f"experiments/{EXPERIMENT}/loss_{EXPERIMENT}.png")
torch.save(model.state_dict(), f"experiments/{EXPERIMENT}/model_{EXPERIMENT}.pth")
print(f"✅ Saved {EXPERIMENT} model and plot successfully!")
