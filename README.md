# Word2vec-From-Scratch

Minimal Word2Vec implementation (Skip-gram with Negative Sampling) in NumPy.

## Requirements

- Python 3.9+ (3.10+ recommended)
- Dependencies from `requirements.txt`

## Quick start

1. (Optional, recommended) create a virtual environment:

```bash
python -m venv .venv
```

Activation:

- Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

- macOS/Linux:

```bash
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run training + a small evaluation demo:

```bash
python word2vec.py
```

The script reads `text_example.txt`, trains embeddings and prints example nearest and furthest neighbors for a test word.

## Data

`text_example.txt` contains text from Project Gutenberg ("The Return of Sherlock Holmes") including the license header.
