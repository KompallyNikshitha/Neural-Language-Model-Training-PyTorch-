import torch
from model import LanguageModel
from utils import load_and_preprocess

# Load vocab and model
encoded, word2idx, idx2word = load_and_preprocess("dataset/Pride_and_Prejudice-Jane_Austen.txt")
vocab_size = len(word2idx)

model = LanguageModel(vocab_size)
model.load_state_dict(torch.load("model_checkpoint.pth", map_location=torch.device('cpu')))
model.eval()

def generate_text(start_word="Gutenberg", length=100):
    model.eval()
    words = [start_word.lower()]
    input_seq = torch.tensor([[word2idx.get(start_word.lower(), 0)]])
    hidden = None
    for _ in range(length):
        with torch.no_grad():
            output, hidden = model(input_seq, hidden)
            next_word_id = torch.argmax(output[:, -1, :], dim=-1).item()
            next_word = idx2word[next_word_id]
            words.append(next_word)
            input_seq = torch.tensor([[next_word_id]])
    return ' '.join(words)

# Generate new text
generated = generate_text("Gutenberg", 100)
print("\n🧠 Generated Text:\n")
print(generated)
