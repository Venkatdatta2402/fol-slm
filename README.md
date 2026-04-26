# FOL SLM — Natural Language → First-Order Logic

A small language model that translates plain-English premises and questions into First-Order Logic (FOL), then uses a [clingo](https://potassco.org/clingo/) ASP solver to derive a verifiable answer with a step-by-step proof.

**HuggingFace Space**: [Venkatdatta/fol-slm-demo](https://huggingface.co/spaces/Venkatdatta/fol-slm-demo)  
**Model**: [Venkatdatta/fol-slm](https://huggingface.co/Venkatdatta/fol-slm)  
**Dataset**: [Venkatdatta/fol-data](https://huggingface.co/datasets/Venkatdatta/fol-data)

---

## Architecture

```
NL premises + question
        │
        ▼  (joined as: "{premises} <extra_id_0> {question}")
T5-base Encoder  (frozen, 768d, 12L)
        │  cross-attention
        ▼
Translation Decoder  (4L, 512d)
        │  autoregressive generation — stops at <extra_id_3>
        ▼
FOL premises + FOL question
        │
        ▼
clingo ASP solver  (symbolic, no neural component)
        │
        ▼
Answer: True / False / Unknown  +  proof chain
```

- **Encoder**: `t5-base`, all weights frozen throughout training
- **Translation Decoder**: 4-layer 512d Transformer trained to map NL → structured FOL using T5's sentinel tokens (`<extra_id_1..3>`) as section markers
- **Reasoner**: clingo — deterministic, verifiable, no learned parameters

The FOL subset used is Horn-clause-like: entity attributes and universal rules (`forall x (P(x) -> Q(x))`), no existential quantifiers, no nested quantifiers.

---

## Setup

```bash
conda create -n fol-slm python=3.11
conda activate fol-slm
pip install -r requirements.txt
```

---

## Data

### Source

ProofWriter OWA splits (depth-2, depth-3, depth-3ext), downloaded from [Kaggle](https://www.kaggle.com/datasets/mathurinache/proofwriter).

Expected layout after download:
```
data/proofwriter/
  depth-2/    meta-train.jsonl  meta-dev.jsonl  meta-test.jsonl
  depth-3/    meta-train.jsonl  meta-dev.jsonl  meta-test.jsonl
  depth-3ext/ meta-train.jsonl  meta-dev.jsonl  meta-test.jsonl
```

### Preprocessing

```bash
python scripts/preprocess_proofwriter.py
```

Outputs `data/processed/train.jsonl`, `dev.jsonl`, `test.jsonl` (~229k / 33k / 66k examples).

Each example gets two transformations:
1. **FOL annotation** — rule-based NL → FOL translator (100% coverage). Premises, question, and proof chain are all converted to FOL form.
2. **Vocabulary substitution** — entity and predicate names are replaced per-question with random draws from NLTK/WordNet pools (7,372 names, 13,006 adjectives, 7,463 verbs), forcing the model to learn structural FOL mapping rather than memorising surface names.

### Data format

Each line in `data/processed/*.jsonl`:
```json
{
  "premises": "Anne is kind. If someone is kind then they are furry. <extra_id_0> Anne is furry.",
  "logic":    "<extra_id_1>\nKind(anne)\nforall x (Kind(x) -> Furry(x))\n<extra_id_2>\nFurry(anne)\n<extra_id_3>\nKind(anne) and forall x (Kind(x) -> Furry(x)) -> therefore Furry(anne)\n<extra_id_4>\nTrue",
  "qdep":     1,
  "answer":   "True",
  "source":   "depth-2/meta-train-1234"
}
```

`premises` is the full encoder input — NL facts and rules, then `<extra_id_0>`, then the NL question.  
`logic` is the full decoder target with sentinel markers: `<extra_id_1>` (FOL premises), `<extra_id_2>` (FOL question), `<extra_id_3>` (proof chain), `<extra_id_4>` (answer).

### Class distribution (train)

| Class | Count | % |
|-------|-------|---|
| pos_True (non-negated → True) | 58,034 | 25.3% |
| neg_False (negated → False) | 57,984 | 25.2% |
| pos_Unknown | 51,808 | 22.5% |
| neg_Unknown | 51,808 | 22.5% |
| pos_False (non-negated → False) | 5,124 | 2.2% |
| neg_True (negated → True) | 5,074 | 2.2% |

`pos_False` and `neg_True` are underrepresented ~11× — a weighted sampler boosts them to match the dominant class frequency during training.

---

## Training

```bash
python scripts/train_translation.py --config configs/v12_translation.yaml
```

Key settings:
- **Optimizer**: AdamW, lr=1.9028e-3, weight_decay=3e-3
- **Schedule**: linear decay with 3,340 warmup steps (16.7% of 20k total)
- **Batch**: 16 × 4 grad accum = effective batch 64
- **Precision**: bfloat16 (loss computed in float32)
- **Curriculum**: QDep≤2 for first 5k steps, then full dataset — prevents early overfitting on hard examples

---

## Hyperparameter Sweep

```bash
python scripts/optuna_sweep.py --config configs/v12_translation.yaml --n-trials 20
```

Searches: `lr`, `weight_decay`, `label_smoothing`, `dropout`, `warmup_fraction`, `layers`, `heads`.  
Each trial runs 3,000 steps (15% of full training). MedianPruner activates after 5 startup trials.  
Results saved to `outputs/optuna_study.db`.

---

## Evaluation

```bash
python scripts/evaluate_translation.py \
    --checkpoint outputs/v12_translation_4L512/checkpoint_final.pt \
    --config configs/v12_translation.yaml \
    --input-file data/processed/test.jsonl \
    --batch-size 64
```

Metrics:
- **Premises-FOL**: fuzzy match (sorted, predicate-normalised, WordNet + edit-distance)
- **Question FOL**: exact match

---

## Runtime Pipeline

```python
from scripts.pipeline import load_model, run

load_model("outputs/v12_translation_4L512/checkpoint_final.pt")

result = run(
    nl_premises="Anne is kind. If someone is kind then they are furry. If someone is furry then they are green.",
    nl_question="Anne is green.",
)

print(result["answer"])        # "True"
print(result["fol_premises"])  # list of FOL premise strings
print(result["fol_question"])  # "Green(anne)"
for step in result["proof"]:
    print(step)
```

---

## Results

Evaluated on the full test split (66,556 examples, unseen vocabulary):

| Metric | Score |
|--------|-------|
| Premises-FOL accuracy (fuzzy) | **85.8%** |
| Question FOL exact match | **91.1%** (60,621 / 66,556) |

---

## Citation

```bibtex
@misc{fol-slm-2026,
  author = {Venkat Datta Bommena},
  title  = {FOL SLM: Natural Language to First-Order Logic with Symbolic Reasoning},
  year   = {2026},
  url    = {https://huggingface.co/Venkatdatta/fol-slm}
}
```

Data source:
```bibtex
@misc{mathurinache-proofwriter-kaggle,
  author = {mathurinache},
  title  = {ProofWriter},
  year   = {2021},
  url    = {https://www.kaggle.com/datasets/mathurinache/proofwriter},
  note   = {Kaggle dataset}
}
```
