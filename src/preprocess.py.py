import random
import json
import sentencepiece as spm

# CAMINHOS (SIMPLES)

ja_path = r"data/ja-pt.txt/TED2020.ja-pt.ja"
pt_path = r"data/ja-pt.txt/TED2020.ja-pt.pt"

spm_model_path = r"data/spm/spm.model"
output_path = r"data/processed/tokenized_pairs.json"

# CARREGAR TOKENIZER

sp = spm.SentencePieceProcessor()
sp.load(spm_model_path)

# LEITURA E ALINHAMENTO

pairs = []

with open(ja_path, encoding="utf-8") as f_ja, \
     open(pt_path, encoding="utf-8") as f_pt:

    for ja, pt in zip(f_ja, f_pt):
        ja = ja.strip()
        pt = pt.strip()

        if not ja or not pt:
            continue

        pairs.append((ja, pt))

print("Pares válidos:", len(pairs))

# AMOSTRA DE SANIDADE

print("\nExemplos aleatórios:")
for ja, pt in random.sample(pairs, 5):
    print("JA:", ja)
    print("PT:", pt)
    print("-" * 40)

# TOKENIZAÇÃO

tokenized_pairs = []

for ja, pt in pairs:
    ja_ids = sp.encode(ja, out_type=int)
    pt_ids = sp.encode(pt, out_type=int)

    if len(ja_ids) == 0 or len(pt_ids) == 0:
        continue

    tokenized_pairs.append({
        "ja": ja_ids,
        "pt": pt_ids
    })

print("\nPares tokenizados:", len(tokenized_pairs))

# SALVAR DATASET

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(tokenized_pairs, f, ensure_ascii=False)

print("Dataset tokenizado salvo com sucesso em:", output_path)

# TESTE FINAL

exemplo = tokenized_pairs[0]

print("\nExemplo tokenizado:")
print("JA IDs:", exemplo["ja"])
print("PT IDs:", exemplo["pt"])

print("\nDecodificando para conferir:")
print("JA:", sp.decode(exemplo["ja"]))
print("PT:", sp.decode(exemplo["pt"]))