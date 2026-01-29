import torch
import sentencepiece as spm
from src.model import TransformerModel

# CONFIG

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SPM_PATH = r"data/spm/spm.model"
CHECKPOINT_PATH = r"checkpoints/transformer_ja_pt_epoch5.pth"

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
MAX_LEN = 50

# TOKENIZER

sp = spm.SentencePieceProcessor()
sp.load(SPM_PATH)

# LOAD CHECKPOINT

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

# Detect vocab sizes AUTOMATICAMENTE
src_vocab_size = checkpoint["src_embedding.weight"].shape[0]
tgt_vocab_size = checkpoint["tgt_embedding.weight"].shape[0]
d_model = checkpoint["src_embedding.weight"].shape[1]

print("SRC vocab:", src_vocab_size)
print("TGT vocab:", tgt_vocab_size)
print("d_model:", d_model)

# MODEL

model = TransformerModel(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    d_model=d_model,
    pad_id=PAD_ID
).to(DEVICE)

model.load_state_dict(checkpoint)
model.eval()

print("Usando dispositivo:", DEVICE)

# TRANSLATE

@torch.no_grad()
def translate(sentence):
    src_ids = sp.encode(sentence, out_type=int)
    src = torch.tensor(src_ids).unsqueeze(0).to(DEVICE)

    tgt_ids = [BOS_ID]

    for _ in range(MAX_LEN):
        tgt = torch.tensor(tgt_ids).unsqueeze(0).to(DEVICE)
        out = model(src, tgt)
        next_id = out[0, -1].argmax().item()

        if next_id == EOS_ID:
            break

        tgt_ids.append(next_id)

    return sp.decode(tgt_ids[1:])

# LOOP

while True:
    text = input("\nDigite uma frase em japonês (ou exit): ")
    if text.lower() == "exit":
        break
    print("Tradução:", translate(text))
