import math
import torch
from model import LanguageModel
from utils import load_and_preprocess, split_data, get_batches

# ===== STEP 1: Load and prepare data =====
print("📘 Loading and preparing data...")
encoded, word2idx, idx2word = load_and_preprocess("dataset/Pride_and_Prejudice-Jane_Austen.txt")
_, val_data, _ = split_data(encoded)
vocab_size = len(word2idx)
print(f"✅ Vocabulary size: {vocab_size}")
print(f"✅ Validation data length: {len(val_data)}")

# ===== STEP 2: Load trained model =====
print("\n🧠 Loading trained model...")
model = LanguageModel(vocab_size)
model.load_state_dict(torch.load("model_checkpoint.pth", map_location=torch.device('cpu')))
model.eval()
print("✅ Model loaded successfully!\n")

# ===== STEP 3: Inspect model architecture =====
print("🔍 Model Architecture:\n")
print(model)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n📊 Total Parameters: {total_params:,}")
print(f"📊 Trainable Parameters: {trainable_params:,}\n")

# ===== STEP 4: Evaluate model performance =====
print("⚙️ Evaluating model on validation set...\n")
criterion = torch.nn.CrossEntropyLoss()
total_loss = 0
count = 0

with torch.no_grad():
    for x, y in get_batches(val_data, seq_len=30, batch_size=64):
        output, _ = model(x)
        loss = criterion(output.view(-1, vocab_size), y.view(-1))
        total_loss += loss.item()
        count += 1

avg_val_loss = total_loss / count
perplexity = math.exp(avg_val_loss)

# ===== STEP 5: Print evaluation results =====
print("✅ Evaluation Complete!\n")
print(f"📉 Validation Loss: {avg_val_loss:.4f}")
print(f"🧩 Validation Perplexity: {perplexity:.2f}")
