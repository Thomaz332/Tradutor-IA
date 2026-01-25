import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import TranslationDataset, collate_fn
from model import TransformerModel

# ===============================
# CONFIG
# ===============================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32
EPOCHS = 3
LR = 1e-4
PAD_ID = 0

# DATA

dataset = TranslationDataset("data/processed/tokenized_pairs.json")
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

print("Dataset:", len(dataset))

# VOCAB SIZE

def get_vocab_size(dataset, key):
    return max(max(item[key]) for item in dataset.data) + 1

SRC_VOCAB = get_vocab_size(dataset, "ja")
TGT_VOCAB = get_vocab_size(dataset, "pt")

# MODELO

model = TransformerModel(
    src_vocab_size=SRC_VOCAB,
    tgt_vocab_size=TGT_VOCAB,
    pad_id=PAD_ID
).to(DEVICE)

criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# TREINO

model.train()

for epoch in range(EPOCHS):
    total_loss = 0

    for ja, pt in loader:
        ja = ja.to(DEVICE)
        pt = pt.to(DEVICE)

        tgt_input = pt[:, :-1]
        tgt_output = pt[:, 1:]

        logits = model(ja, tgt_input)

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt_output.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} - Loss: {total_loss / len(loader):.4f}")

torch.save(model.state_dict(), "checkpoints/transformer_ja_pt.pth")
print("Modelo salvo!")
