import sentencepiece as spm

# CAMINHOS (SIMPLES)

ja_path = r"data/ja-pt.txt/TED2020.ja-pt.ja"
pt_path = r"data/ja-pt.txt/TED2020.ja-pt.pt"

spm_input = r"data/spm/spm_input.txt"
model_prefix = r"data/spm/spm"

# CRIAR spm_input.txt

with open(spm_input, "w", encoding="utf-8") as out, \
     open(ja_path, encoding="utf-8") as f_ja, \
     open(pt_path, encoding="utf-8") as f_pt:

    for ja, pt in zip(f_ja, f_pt):
        ja = ja.strip()
        pt = pt.strip()

        if ja and pt:
            out.write(ja + "\n")
            out.write(pt + "\n")

print("spm_input.txt criado com sucesso")

# TREINAR SENTENCEPIECE

spm.SentencePieceTrainer.train(
    input=spm_input,
    model_prefix=model_prefix,
    vocab_size=16000,
    model_type="unigram",
    pad_id=0,
    bos_id=1,
    eos_id=2,
    unk_id=3
)

print("SentencePiece treinado com sucesso!")
print("Arquivos gerados:")
print(" - data/spm/spm.model")
print(" - data/spm/spm.vocab")
