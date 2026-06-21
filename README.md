# MLX Encoder-Decoder Transformer for Translation

This repository contains a rewritten and extended sequence-to-sequence translation notebook:

- `s2sTransformer.ipynb`: original TensorFlow/Keras notebook
- `s2sTransformer_mlx_c.ipynb`: rewritten MLX version with a more explicit training and inference pipeline
- `saved_translation_model/`: saved MLX checkpoint, tokenizer files, and model configuration
- `spm_models/`: SentencePiece tokenizer models and tokenizer training text used by the notebook

The project builds a small English-to-French encoder-decoder Transformer. It is intended as a technical portfolio example showing practical understanding of modern neural translation systems: tokenization, attention, masking, training loops, validation, decoding, and model serialization.

## Project Motivation

The original notebook demonstrated the basic sequence-to-sequence translation workflow using TensorFlow/Keras abstractions. The rewritten notebook keeps the same learning objective but rebuilds the system in MLX, making the implementation more transparent and more suitable for Apple Silicon experimentation.

Rather than relying on high-level Keras methods such as `model.fit()`, Keras `TextVectorization`, and Keras model serialization, the MLX version exposes the core mechanics directly:

- data loading and filtering
- subword tokenizer training
- encoder, decoder, and target sequence construction
- Transformer block implementation
- attention masking
- masked loss and token accuracy
- optimizer updates
- validation and early stopping
- beam search decoding
- portable checkpoint save/load

## What Was Improved

The rewritten notebook adds several substantial improvements over the original version.

First, it replaces local absolute file paths with a public Hugging Face OPUS dataset. This makes the notebook easier to reproduce on another machine and clearer for reviewers.

Second, it replaces Keras text vectorization with SentencePiece subword tokenization. This is a better fit for translation because rare words, names, punctuation variants, and inflected forms can be represented as smaller learned pieces instead of being dropped or mapped directly to unknown tokens.

Third, it implements the Transformer architecture directly in MLX. The notebook defines positional encodings, encoder blocks, decoder blocks, multi-head self-attention, cross-attention, residual connections, layer normalization, feed-forward layers, and the final vocabulary projection.

Fourth, it makes training explicit. The notebook includes a custom masked cross-entropy loss, masked token-level accuracy, a Transformer-style warmup learning-rate schedule, validation evaluation, early stopping, and history plotting.

Finally, it improves inference and deployment hygiene. Greedy decoding is replaced with beam search, and the trained weights, tokenizer files, and architecture configuration are saved together so the model can be restored without hidden notebook state.

## Highlights

### Encoder-Decoder Transformer

The model follows the standard encoder-decoder pattern used in neural machine translation:

1. The encoder reads the source English sentence.
2. The decoder receives the partially generated French sentence.
3. Decoder self-attention is masked so it cannot see future target tokens.
4. Decoder cross-attention lets each French position attend to the encoded English sentence.
5. A final linear layer predicts the next French token.

### Teacher Forcing

During training, the target sentence is split into two shifted sequences:

- decoder input: starts with a beginning-of-sequence token
- target output: ends with an end-of-sequence token

This teaches the model to predict the next French token at each position while conditioning on the correct previous target tokens.

### Attention Masks

The notebook uses two important masks:

- padding masks prevent the model from attending to artificial padding tokens
- causal masks prevent the decoder from seeing future target tokens

These masks are essential for making training behavior match inference behavior.

### Beam Search

At inference time, the model generates translations autoregressively. Instead of selecting only the single highest-probability token at every step, beam search keeps multiple candidate translations alive and ranks them using length-normalized log probability.

## Repository Structure

```text
.
├── README.md
├── s2sTransformer.ipynb
├── s2sTransformer_mlx_c.ipynb
├── spm_models/
│   ├── en_train.txt
│   ├── fr_train.txt
│   ├── sp_en.model
│   ├── sp_en.vocab
│   ├── sp_fr.model
│   └── sp_fr.vocab
└── saved_translation_model/
    ├── config.json
    ├── sp_en.model
    ├── sp_fr.model
    └── transformer_weights.npz
```

The generated model artifacts are intentionally included. `saved_translation_model/` lets reviewers reload the trained model without retraining from scratch, while `spm_models/` preserves the SentencePiece tokenizers and tokenizer training text used during preprocessing.

The tokenizer models are duplicated in `saved_translation_model/` because a deployable checkpoint should contain the exact tokenizers required by its weights. The separate `spm_models/` directory remains useful for showing and rerunning the tokenizer training step in the notebook.

## Requirements

The main notebook uses:

- Python 3
- MLX
- NumPy
- pandas
- Plotly
- Hugging Face `datasets`
- SentencePiece

Example installation:

```bash
pip install mlx numpy pandas plotly datasets sentencepiece pyarrow tqdm
```

MLX is designed for Apple Silicon Macs. On non-Apple-Silicon systems, the original TensorFlow/Keras notebook may be easier to run.

## How to Run

Open `s2sTransformer_mlx_c.ipynb` in Jupyter, VS Code, or another notebook environment and run the cells from top to bottom.

The notebook will:

1. load the English/French dataset
2. train SentencePiece tokenizers
3. build tokenized batches
4. train the Transformer
5. plot training metrics
6. run example translations
7. save the model, tokenizer files, and config
8. reload the saved checkpoint for inference

Training time depends on hardware, dataset size, and the configured number of epochs.

## Using the Included Checkpoint

The repository includes a saved checkpoint in `saved_translation_model/`. The final notebook section demonstrates how to reload it by reconstructing the Transformer architecture from `config.json`, loading `transformer_weights.npz`, and restoring the matching SentencePiece tokenizers.