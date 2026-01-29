import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import TranslationDataset, collate_fn
from model import TransformerModel

# CONFIG

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 64         
EPOCHS = 20
LR = 1e-4
PAD_ID = 0
ACCUM_STEPS = 4         

SAVE_EPOCHS = [3, 5, 8, 20]

# DATA

dataset = TranslationDataset("data/processed/tokenized_pairs.json")
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

print("Dataset:", len(dataset))
print("Usando dispositivo:", DEVICE)

# VOCAB SIZE

def get_vocab_size(dataset, key):
    return max(max(item[key]) for item in dataset.data) + 1

SRC_VOCAB = get_vocab_size(dataset, "ja")
TGT_VOCAB = get_vocab_size(dataset, "pt")

print("SRC vocab:", SRC_VOCAB)
print("TGT vocab:", TGT_VOCAB)

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
    optimizer.zero_grad()

    for step, (ja, pt) in enumerate(loader):
        ja = ja.to(DEVICE)
        pt = pt.to(DEVICE)

        tgt_input = pt[:, :-1]
        tgt_output = pt[:, 1:]

        logits = model(ja, tgt_input)

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt_output.reshape(-1)
        )

        loss = loss / ACCUM_STEPS
        loss.backward()

        if (step + 1) % ACCUM_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    epoch_num = epoch + 1

    print(f"Epoch {epoch_num} - Loss: {avg_loss:.4f}")

    if epoch_num in SAVE_EPOCHS:
        path = f"checkpoints/transformer_ja_pt_epoch{epoch_num}.pth"
        torch.save(model.state_dict(), path)
        print(f"Checkpoint salvo em {path}")

    torch.cuda.empty_cache()

print("Treinamento finalizado!")
