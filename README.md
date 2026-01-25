# Tradutor-IA
IA que traduz texto em Japones para Portugues

```bash
IC_TRADUCAO_IA/
│
├── data/
│   ├── ja-pt.txt 
│   │
│   ├── processed/
│   │   ├── tokenized_pairs.json
│   │
│   └── spm/
│       ├── spm_input.txt
│       ├── spm.model
│       └── spm.vocab
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── preprocess.py
│   ├── train.py
│   └── spm.model.py
│
├── checkpoints/
│   └── transformer_ja_pt.pth
│
│── translate.py
├── requirements.txt
├── README.md
└── .gitignore
```
spm_train.py -> preprocess.py -> train.py

py -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
