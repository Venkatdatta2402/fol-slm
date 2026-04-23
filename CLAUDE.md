# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Conda environment: `fol-slm`

```bash
conda activate fol-slm
pip install -r requirements.txt
```

## Commands

**Train (v1 single decoder):**
```bash
python scripts/train_final.py --config configs/final.yaml
```

**Train v12 two-decoder architecture:**
```bash
# Phase 1: translation decoder only
python scripts/train_translation.py --config configs/v12_translation.yaml

# Phase 2: proof decoder only, freeze translation decoder
python scripts/train_proof.py --config configs/v12_proof.yaml

# Phase 3: cls head only, freeze both decoders
python scripts/train_cls_head.py \
    --checkpoint outputs/v12_proof/checkpoint_final.pt \
    --config configs/v12_proof.yaml \
    --out-dir outputs/cls_head_v12/
```

**Train v13 stacked 8-layer decoder:**
```bash
python scripts/train_v13.py --config configs/v13.yaml
```

**Train cls head on frozen decoder (any checkpoint):**
```bash
python scripts/train_cls_head.py \
    --checkpoint outputs/final_v8/checkpoint_final.pt \
    --config configs/final.yaml \
    --lr 1e-3 \
    --max-steps 3000 \
    --out-dir outputs/cls_head_v8/
```

**Optuna hyperparameter sweep:**
```bash
python scripts/optuna_sweep.py
python scripts/sweep_pilot.py   # lightweight pilot before full sweep
```

**Generate outputs at inference:**
```bash
python scripts/generate.py --checkpoint outputs/checkpoint_final.pt --config configs/base_config.yaml
```

**Evaluate a checkpoint:**
```bash
python scripts/evaluate.py \
    --checkpoint outputs/final_v8/checkpoint_final.pt \
    --config configs/final.yaml \
    --input-file data/processed/test.jsonl \
    --max-samples 200 \
    --output-file outputs/eval_v8_200.jsonl

# With cls head for answer prediction:
python scripts/evaluate.py \
    --checkpoint outputs/final_v8/checkpoint_final.pt \
    --config configs/final.yaml \
    --input-file data/processed/test.jsonl \
    --max-samples 200 \
    --output-file outputs/eval_cls_head_200.jsonl \
    --use-cls-head \
    --cls-head-checkpoint outputs/cls_head_v8/checkpoint_final.pt

# Evaluate v12 full pipeline (translation + proof decoder):
python scripts/evaluate_v12.py \
    --proof-checkpoint outputs/v12_proof/checkpoint_final.pt \
    --config configs/v12_proof.yaml \
    --input-file data/processed/test.jsonl \
    --max-samples 200 \
    --output-file outputs/eval_v12_200.jsonl

# Evaluate translation decoder only:
python scripts/evaluate_translation.py \
    --checkpoint outputs/v12_translation_4L512/checkpoint_final.pt \
    --config configs/v12_translation.yaml \
    --input-file data/processed/test.jsonl \
    --max-samples 200
```

**Analyze evaluation results (confusion matrices + case files):**
```bash
python scripts/analyze_failures.py \
    --eval-file outputs/eval_v8_200.jsonl \
    --test-file data/processed/test.jsonl \
    --out-dir outputs/analysis_v8
```

**Fine-tune last decoder layer(s) with confusion-pair balanced data:**
```bash
python scripts/finetune_unknown.py \
    --checkpoint outputs/final_v8/checkpoint_final.pt \
    --config configs/final.yaml \
    --base-lr 1e-4 \
    --max-steps 5000 \
    --unfreeze-at 99999 \
    --out-dir outputs/finetune_v8_balanced/
```

**Preprocess ProofWriter data:**
```bash
python scripts/preprocess_proofwriter.py                  # full dataset
python scripts/preprocess_proofwriter.py --pilot          # QDep=1 only (sweep piloting)
python scripts/preprocess_proofwriter.py --min-qdep 1     # exclude trivial QDep=0
```

**Run tests:**
```bash
pytest
```

## Architecture

### V1: Single-decoder (FOLModel)

`src/model/model.py` — `FOLModel`

```
T5 Encoder (frozen, 768d)
    ↓ cross-attn
Decoder (4L, 512d) — generates full sequence: FOL premises + question + proof chain
    ↓ hidden state at <extra_id_4>
Cls Head (512→3) → True / False / Unknown
```

The decoder generates the full sequence in one pass. Answer token was in labels in v7/v8 (helped representations but introduced phrasing shortcut). Removed in v10/v11 (cleaner but cls head can't separate True/False without that signal).

**Proof self-attention mask** (v11): at proof positions (after `<extra_id_3>`), self-attention is blocked from attending to FOL premises (before `<extra_id_2>`). Proof steps can only attend to FOL question + previously generated proof steps. Built by `build_proof_self_attn_mask()` in `src/utils/attention.py`.

### V2: Two-decoder (FOLModelV2)

`src/model/model.py` — `FOLModelV2`

```
T5 Encoder (frozen, 768d)
    ↓ cross-attn
Translation Decoder (4L, 512d)
    input: [<extra_id_1> FOL_premises <extra_id_2> FOL_question]  ← truncated at <extra_id_3>
    learns: NL → structured FOL (lexical mapping, no proof)
    ↓ hidden states (B, T_trans, 512d) — pure FOL representations, no proof/answer context
Proof Decoder (4L, 512d)
    input: [<extra_id_3> proof_step_1 ...]  ← truncated at <extra_id_4>
    cross-attends: translation decoder hidden states (structured FOL only, no NL)
    learns: FOL reasoning chain (no NL phrasing, no answer token)
    ↓ hidden state at last proof position
Cls Head (512→3) → True / False / Unknown
```

**Known issue**: Proof decoder collapses at inference due to train/inference mismatch in hidden state format and padding mask convention (`int64 1=real/0=pad` during training vs `bool` at inference). Fix applied in `evaluate_v12.py`: re-run translation decoder in one shot on generated sequence + use `torch.ones(..., dtype=torch.long)` as padding mask.

**V12 findings**: Proof decoder trained successfully (val_loss 0.33) but generates hallucinated proofs — cross-attention spreads uniformly instead of focusing on relevant premises. Curriculum hard switch at step 5k caused catastrophic spike (val_loss 0.40 → 1.85). Translation decoder achieved 99.7% premises-FOL (fuzzy), 100% question.

### V3: Two-decoder with T5 cross-attention (FOLModelV3) ← current approach

`src/model/model.py` — `FOLModelV3`

Sequential two-phase training like V2, but proof decoder cross-attends directly to T5 encoder (not translation hidden states):

```
T5 Encoder (frozen, 768d)
    ↓ cross-attn (premises only, masked at <extra_id_0>)
Translation Decoder (4L, 512d)                    Proof Decoder (4L, 512d)
    Phase 1: trained on NL→FOL                        Phase 2: trained on proof chain
    input: NL premises + question                      input: [<extra_id_2>, FOL_question,
    output: FOL premises + question                             <extra_id_3>, gold_proof_{0..t-1}]
                                                       self-attn: FOL question + prior proof steps
                                                           (no mask needed — FOL premises never
                                                            in decoder input)
                                                       cross-attn: T5 encoder, premises only
                                                           (masked at <extra_id_0>)
    ↓ hidden state at <extra_id_4>
Cls Head (512→3) → True / False / Unknown
```

**Key design principles:**
- Proof decoder input starts at `<extra_id_2>` — FOL question tokens (from translation decoder) then gold proof tokens
- FOL premises are never in the proof decoder input → no self-attn mask needed, clean by construction
- Cross-attn to T5 encoder premises only (768d, 12L pretrained) — richer than translation hidden states (512d)
- Negation shortcut lives in NL question, not NL premises → premises-only cross-attn removes it
- No train/inference mismatch: proof decoder never touches translation hidden states
- At inference: run translation decoder to get FOL question tokens, feed as prefix to proof decoder

**Gradual curriculum (Phase 2):** QDep≤1 (5k) → QDep≤2 (10k) → QDep≤3 (15k) → full (25k)

**`build_premises_cross_attn_mask()`** in `src/utils/attention.py`:
- (B, S_enc) bool mask, True=ignore positions at/after `<extra_id_0>` in encoder input
- Applied directly to PyTorch MHA (no inversion needed)

### Data Pipeline

**V1/V3:** `ReasoningDataset` + `FOLCollator`
- Labels: answer token masked to -100 (decoder trains on proof only, not answer)
- Full sequence in decoder_input_ids

**V2:** `FOLDatasetV2` + `FOLCollatorV2`
- `trans_decoder_input_ids`: target truncated before `<extra_id_3>` — translation decoder never sees proof
- `trans_labels`: shifted, last label = `<extra_id_3>`
- `proof_decoder_input_ids`: from `<extra_id_3>` up to (not incl.) `<extra_id_4>`
- `proof_labels`: shifted, last label = `<extra_id_4>`
- `trans_padding_mask`: int64, 1=real/0=pad — proof decoder cross-attention key_padding_mask
  **Important**: decoder inverts this (`== 0`) internally. At inference use `torch.ones(..., dtype=torch.long)` not bool zeros.

**Weighted sampler** (all versions):
- `get_sample_weights()` boosts only `pos_False` and `neg_True` (~11×) to match dominant class frequency
- All other classes (incl. Unknown) stay at natural rate — do NOT penalise Unknown

### Model Components

- **`FrozenT5Encoder`** (`src/model/encoder.py`): T5EncoderModel, all frozen, returns `last_hidden_state` (B, S, 768)
- **`TransformerDecoder`** (`src/model/decoder.py`): generic decoder used for all versions. `last_hidden` (B, T, d_model) stored after each forward. Accepts `proof_self_attn_mask`, `proof_cross_attn_mask`, `proof_layer_start` parameters.
- **`src/utils/attention.py`**:
  - `cross_attention_entropy()`: mean entropy of cross-attn distributions. Use `w.clamp(min=1e-9)` — do NOT use additive epsilon pattern
  - `build_proof_self_attn_mask()`: blocks FOL premises from self-attention at proof positions. Returns (B*n_heads, T, T) additive mask. Includes standard causal masking.
  - `build_premises_cross_attn_mask()`: restricts cross-attn to NL premises (before `<extra_id_0>`). Returns (B, S_enc) bool mask. True=ignore.
  - `build_fol_mask()`: float mask for entropy diagnostics (1.0 at FOL premise positions)

### Training Details

- **Precision**: bf16 preferred — same range as fp32. Loss always in float32 (`logits.float()`)
- **Optimizer**: AdamW, weight_decay=3e-3, lr=1.9028e-3 (from Optuna Phase 3)
- **Warmup**: 16.7% of max_steps
- **Curriculum (V3)**: gradual QDep≤1 (5k) → QDep≤2 (10k) → QDep≤3 (15k) → full (25k)
- **Masking labels**: sets positions to -100 — CrossEntropyLoss ignores them. Model still sees all tokens in decoder_input_ids; only gradient signal is blocked.

### Cls Head

`answer_cls_head = nn.Linear(decoder_dim, 3)` — True=0, False=1, Unknown=2

Trained in isolation on frozen decoder hidden states. Takes hidden state at `<extra_id_4>` position.

Key finding: cls head only works if decoder representations are separable:
- V8 (answer in labels): 99.5% val accuracy — perfectly separable
- V10 (answer removed from labels): ~78% unstable — True/False not separable
- V11 (proof self-attn mask + no answer): 86.9% — partially separable, still oscillating

## Dataset

### Source

**ProofWriter** (OWA splits) — multi-step logical reasoning dataset by Allen AI. Downloaded from Kaggle (`mathurinache/proofwriter`).

Files used:
```
data/proofwriter/
  depth-2/   meta-train.jsonl   meta-dev.jsonl   meta-test.jsonl
  depth-3/   meta-train.jsonl   meta-dev.jsonl   meta-test.jsonl
  depth-3ext/ meta-train.jsonl  meta-dev.jsonl   meta-test.jsonl
```

Preprocessed output (ready for training):
```
data/processed/
  train.jsonl   (~229k examples)
  dev.jsonl     (~33k examples)
  test.jsonl    (~66k examples)
```

### Data Format

Each line in `data/processed/*.jsonl`:
```json
{
  "premises": "Anne is kind. Bob is furry. ...",
  "logic":    "<extra_id_1>\nKind(anne)\n...\n<extra_id_2>\nGreen(anne)\n<extra_id_3>\n...\n<extra_id_4>\nTrue",
  "qdep":     1,
  "answer":   "True",
  "source":   "AttNeg-OWA-D2-1865"
}
```

### Decoder Target Structure (`logic` field)

Sentinels: `<extra_id_1>` (premises FOL), `<extra_id_2>` (question FOL), `<extra_id_3>` (proof), `<extra_id_4>` (answer).

```
<extra_id_1>
Kind(anne)
forall x (Kind(x) -> Furry(x))
<extra_id_2>
Green(anne)                                         ← FOL question
<extra_id_3>
Kind(anne) and forall x (Kind(x) -> Furry(x)) -> therefore Furry(anne)
Furry(anne) and forall x (Furry(x) -> Green(x)) -> therefore Green(anne)
<extra_id_4>
True
```

For Unknown (failure chain):
```
<extra_id_3>
forall x (Big(x) and Round(x) -> White(x)) <- Rough(fiona) -> Big(fiona) <- [no base fact]
Cannot be determined from given premises.
<extra_id_4>
Unknown
```

**Encoder input format** (V3): `"{premises} <extra_id_0> {question_nl}"` — `<extra_id_0>` separates premises from question in NL. Proof layers' cross-attn is restricted to positions before `<extra_id_0>`.

### Data Distribution

```
Total train: 229,832
  pos_True    (non-negated, True):    58,034  (25.3%)
  neg_False   (negated, False):       57,984  (25.2%)
  pos_Unknown (non-negated, Unknown): 51,808  (22.5%)
  neg_Unknown (negated, Unknown):     51,808  (22.5%)
  pos_False   (non-negated, False):    5,124   (2.2%)  ← underrepresented, boosted 11×
  neg_True    (negated, True):         5,074   (2.2%)  ← underrepresented, boosted 11×
```

### Preprocessing

```bash
python scripts/preprocess_proofwriter.py
```

Rule-based translator — 100% coverage. ProofWriter's NL maps deterministically to FOL.
Unknown proofs: parses `proofs` failure trace string → FOL failure chain `rule1_fol <- rule2_fol <- [no base fact]`.

## Evaluation

### Metrics

- **Metric 1**: Answer accuracy overall + per-QDep breakdown
- **Metric 2a**: Premises-FOL match (sorted, predicate-normalised, WordNet + edit-distance fuzzy matching)
- **Metric 2b**: Proof match (LCS of `therefore X` steps, normalised by gold length)
- **Confusion matrices**: 3×3 (gold × pred) for counts, premises score, proof score — split by overall / negated / non-negated

### evaluate.py

Supports `--use-cls-head --cls-head-checkpoint` for answer prediction via cls head instead of greedy decoding. Always use `--max-samples 200` for quick checks.

### evaluate_v12.py

Full v12 pipeline evaluation: chains translation decoder → proof decoder. Uses teacher-forced hidden states (re-runs full generated sequence through translation decoder in one shot) and `torch.ones(..., dtype=torch.long)` padding mask to match training format.

### evaluate_translation.py

Translation decoder only. Uses fuzzy scoring from `evaluate.py` (`score_premises_fol` + `_split_fol_statements`). Scores: 99.7% premises-FOL, 100% question at step 20k (4L/512d).

### analyze_failures.py

Joins gold data by `premises` string (NOT `source` — multiple questions share same source ID). Outputs 9 case files (one per gold×pred cell) + confusion_matrices.txt split by negated/non-negated.

## Training History

| Run | Checkpoint | Key changes | True | False | Unknown | Overall |
|-----|-----------|-------------|------|-------|---------|---------|
| v7 | `outputs/final/` | cls head (coeff=0.05, delayed 10k), full proof traces | 88.9% | 77.8% | 68.5% | **75.5%** |
| v8 | `outputs/final_v8/` | richer Unknown proofs, auto-calibrated cls coeff | 92.6% | 90.7% | 56.5% | 74.5% |
| v9 | `outputs/final_v9/` | no cls head, buggy equal-weight sampler | 94.4% | 88.9% | 46.7% | 71.0% |
| v10 | `outputs/final_v10/` | fixed weighted sampler, answer masked from labels | — | — | — | — |
| v11 | `outputs/final_v11/` | + proof self-attn mask (blocks FOL premises at proof positions) | — | — | — | — |
| v12 | `outputs/v12_proof/` | two-decoder: collapsed at inference (padding mask bug + curriculum spike) | — | — | — | — |
| v13 | `outputs/v13/` | two-decoder: proof decoder cross-attends T5 encoder (premises only), input=[FOL_question+gold_proof] | — | — | — | — |

**v7 has the best Unknown accuracy; v8 has the best True/False accuracy.**

### Cls Head Results (trained separately on frozen decoder)

| Base model | Val acc | True | False | Unknown | Notes |
|---|---|---|---|---|---|
| v8 | **99.5%** | 99.1% | 99.7% | 100.0% | Answer in labels → perfectly separable |
| v10 | ~78% unstable | oscillating | oscillating | 99.7% | No answer signal → not separable |
| v11 | 86.9% | 80.4% | 72.6% | 99.7% | Proof mask helps but still oscillating |

### Key Findings

- **Phrasing shortcut**: model learns positive → True, negative → False. Root cause: pos_False and neg_True are 11× underrepresented.
- **Confusion pairs**: negated → False/Unknown confused; non-negated → True/Unknown confused.
- **Negation shortcut lives in question, not premises**: `<extra_id_0>` separates NL premises from NL question in encoder input — restricting proof layer cross-attn to premises removes the shortcut.
- **Answer in labels**: makes cls head work perfectly but bakes shortcut into proof representations.
- **Answer removed from labels**: cleaner proofs but cls head can't separate True/False.
- **Proof self-attn mask**: blocks FOL premises from self-attention at proof positions — forces proof to be driven by question + derived steps. Improved proof quality (v11: 86.9%).
- **V12 failure**: hard curriculum switch caused val_loss spike (0.40→1.85); padding mask format mismatch (`int64` training vs `bool` inference) caused proof decoder collapse.
- **V13 rationale**: proof decoder cross-attends T5 encoder premises only (richer 768d signal, no shortcut); decoder input = FOL question + gold proof (no self-attn mask needed, premises never in input); gradual curriculum prevents spike.

## Checkpoints

```
outputs/final/checkpoint_final.pt                  ← v7 (75.5%, best Unknown 68.5%)
outputs/final_v8/checkpoint_final.pt               ← v8 (74.5%, best True/False)
outputs/final_v9/checkpoint_final.pt               ← v9 (71.0%)
outputs/final_v10/checkpoint_final.pt              ← v10 (proof-only labels, fixed sampler)
outputs/final_v11/checkpoint_final.pt              ← v11 (+ proof self-attn mask)
outputs/cls_head_v8/checkpoint_final.pt            ← cls head on v8 (99.5% val)
outputs/cls_head_v11/checkpoint_final.pt           ← cls head on v11 (86.9% val)
outputs/v12_translation_4L512/checkpoint_final.pt  ← v12 translation decoder (99.7% premises, 100% question)
outputs/v12_proof/checkpoint_final.pt              ← v12 proof decoder (collapsed — cross-attn issue)
outputs/v13/checkpoint_final.pt                    ← v13 proof decoder cross-attends T5 encoder premises (in progress)
```
