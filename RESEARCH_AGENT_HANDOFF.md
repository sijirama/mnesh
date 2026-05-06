# Mnesh Handoff Summary

## Goal

This document summarizes the model changes, training outcomes, and diagnostics completed after the auxiliary-loss retrain plan was introduced, so follow-up research can start from the current state instead of re-deriving prior work.

## Starting Point

Initial problem observed:

- The decoder produced nearly identical outputs across different sessions.
- The encoder did produce different session vectors, but the decoder largely ignored them.
- Frequent commands, especially git commands, dominated generation.

Root cause identified:

- The earlier decoder conditioning path concatenated the seed to every decoder step.
- Under teacher forcing, the decoder learned a shortcut and ignored the seed.

## Architecture Changes Implemented

### 1. Decoder conditioning fix

Implemented:

- Removed per-step seed concatenation.
- Decoder now consumes only token embeddings.
- Seed is used to initialize decoder hidden state `h0`.

Effect:

- This forced all encoder information to flow through the hidden-state initialization path.

### 2. Teacher-forcing alignment fix

Implemented:

- Training and evaluation now use shifted decoder inputs and shifted decoder targets.
- `decoder_input = target_ids[:, :-1]`
- `decoder_target = target_ids[:, 1:]`

Effect:

- Training now matches autoregressive inference behavior correctly.

### 3. Regularization and encoder cleanup

Implemented:

- `padding_idx=0` for token embeddings.
- Dropout added to embeddings and recurrent summaries.
- Inner GRU now packs padded sequences.
- Packed lengths are computed from token ids, not from embeddings.
- Outer GRU is bidirectional.
- Context projector uses layer norm and dropout.

Effect:

- Better sequence handling and cleaner encoder signal.

### 4. Auxiliary next-cmd-type supervision

Implemented:

- Added `MneshCmdTypeHead`.
- Dataset now returns `target_cmd_type`.
- Training optimizes:
  - command generation loss
  - weighted auxiliary cmd-type classification loss
- Added:
  - `AdamW`
  - label smoothing for token loss
  - warmup + cosine scheduler
  - cmd-type validation accuracy reporting

Effect:

- Session representation started carrying more type information, but early runs showed the decoder still falling back to frequent commands.

### 5. Attention pooling over session command states

Implemented:

- Outer bidirectional GRU now returns per-command outputs instead of only one final summary.
- Added `MneshAttentionPool` over the 10 command states.
- Pooled session vector is used for:
  - `cmd_type_head`
  - seed projection into decoder
- Inference tools now print attention weights for debugging.

Effect:

- This improved conditioning noticeably, especially for docker sessions.

### 6. Residual session MLP

Implemented:

- Added `MneshSessionRefiner`, a residual MLP with:
  - linear
  - GELU
  - dropout
  - linear
  - residual add
  - layer norm
- Applied after attention pooling and before:
  - `cmd_type_head`
  - context/seed projection

Why:

- Attention pooling improved which commands matter.
- The residual MLP is intended to make the pooled session vector more expressive before mapping it into decoder seed space.
- This is a low-risk way to add nonlinear capacity without deepening the recurrent stack or decoder.

## Training Runs and Outcomes

### Run before attention pooling

Best validation loss:

- `2.1874`

Observed behavior:

- Session type predictions improved somewhat.
- Decoder still collapsed often to frequent git outputs.

Example behavior:

- Docker and python sessions could have correct top cmd-type predictions.
- Greedy decoding still often returned `git add .`.

Conclusion:

- Encoder carried some signal.
- Decoder coupling to that signal was still too weak.

### Run with attention pooling

Best validation loss:

- `2.1792`

Observed behavior from `model.temp_inference_eval`:

- Docker:
  - top cmd type: `docker 0.5071`
  - greedy output: `docker compose up -d`
- Python:
  - top cmd type: `python 0.5429`
  - greedy output still collapsed to `git add .`
- Sysadmin:
  - top cmd types moved toward `system/process/ssh`
  - generation remained noisy
- Frontend:
  - improved slightly
  - greedy output became `git status`

Conclusion:

- Attention pooling helped.
- Conditioning is better but incomplete.
- Decoder fallback to high-frequency commands still exists, especially outside the strongest contexts.

## Key Diagnostics

### Cmd-type head is not dead

Observed previously:

- head weights had nontrivial variance
- only a small fraction of weights were near zero

Conclusion:

- The auxiliary head is learning something.
- The main issue is not a dead classifier head.

### Encoder signal exists

Evidence:

- Docker and python sessions produced clearly different cmd-type predictions.
- Attention weights also differed across sessions.

Conclusion:

- The encoder is not totally failing.
- The remaining weakness is the mapping from session representation to robust token generation.

## Current Model State

Current architecture now includes:

- inner GRU with packed padded sequences
- bidirectional outer GRU
- attention pooling over command states
- residual MLP session refiner
- context projector into decoder seed
- auxiliary cmd-type supervision

## Training Script Behavior

Updated behavior:

- Epoch-end beacon notifications are enabled.
- Final completion beacon notification is enabled.
- Automatic RunPod stop at the end of training has been commented out.

## Current Evaluation Utility

Inference evaluation helper moved to:

- `model/temp_inference_eval.py`

Run with:

```bash
PYTHONPATH=. python3 -m model.temp_inference_eval
```

## Practical Interpretation

Summary of what has been learned so far:

1. The original architecture was not fundamentally wrong.
2. The first major issue was decoder conditioning collapse.
3. After fixing conditioning, the remaining issue became weak session-to-decoder coupling under dataset imbalance.
4. Auxiliary cmd-type loss helped the representation but was not enough on its own.
5. Attention pooling improved conditioning, especially for docker sessions.
6. Residual MLP was added next to strengthen the pooled session representation before seed projection and decoding.

## Suggested Next Research Focus

Questions worth investigating next:

1. Whether the residual MLP improves greedy decoding for python/sysadmin/front-end sessions.
2. Whether cmd-type accuracy improves materially with the current architecture.
3. Whether the next major bottleneck is still decoder fallback to high-frequency outputs.
4. Whether the task itself should eventually become two-stage:
   - predict command family / intent
   - then generate within that narrower space

If the residual MLP still does not sufficiently improve generation, the next likely areas to examine are:

- stronger decoder conditioning mechanisms
- multi-task or two-stage generation
- retrieval/reranking hybrids instead of pure next-command generation
