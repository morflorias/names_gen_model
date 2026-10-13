# names_gen_model

Character-level LSTM (PyTorch) that generates hybrid product names conditioned on
two or more categories at once (e.g. `cosmetics + food` for *coconut oil*), with an
optional start-token prefix. Comes with a CLI and a Streamlit web UI.

## Features

- Generate names for any combination of 2+ categories
- Optional start-token prefix
- Sampling (temperature + top-k) and beam search
- Evaluation metrics and t-SNE embedding visualization
- Streamlit web UI with training history charts

## Quick start

```bash
pip install -r requirements.txt

# train (uses data/corpus_pairs.csv; a pre-trained checkpoint is included in checkpoints/)
python main.py train

# generate hybrid names
python main.py generate --categories cosmetics,food --n 5

# list available start tokens
python main.py generate --categories cosmetics,food --list-start-tokens

# launch the web UI
python main.py web
```

## Project structure

| Path                    | Description                                   |
|-------------------------|-----------------------------------------------|
| `main.py`               | CLI entry point (`train`/`generate`/`eval`/`tsne`/`web`) |
| `src/`                  | Model, tokenizer, dataset, training, generation, evaluation |
| `app/streamlit_app.py`  | Streamlit web UI                              |
| `data/`                 | Corpora and data-generation scripts           |
| `checkpoints/`          | Trained model, tokenizer, training history    |
| `release/`              | Packaged standalone build                     |
| `scripts/`              | Release build and run helpers                 |

## License

MIT — see [LICENSE](LICENSE).
