"""Gradio demo for FOL SLM — NL → FOL → clingo symbolic reasoning.

Hosted on Hugging Face Spaces. Loads the translation decoder checkpoint
from the HF Hub model repo on first inference.
"""

import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path so `scripts` and `src` are importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr
from huggingface_hub import hf_hub_download

# ── Model loading (once on first call) ───────────────────────────────────────

_loaded = False

REPO_ID  = "Venkatdatta/fol-slm"
CKPT_FILE = "checkpoint_final.pt"

def _ensure_loaded():
    global _loaded
    if _loaded:
        return
    ckpt_path = hf_hub_download(repo_id=REPO_ID, filename=CKPT_FILE)
    from scripts.pipeline import load_model
    load_model(ckpt_path, config_path="configs/v12_translation.yaml")
    _loaded = True


# ── Inference ─────────────────────────────────────────────────────────────────

def predict(nl_premises: str, nl_question: str):
    if not nl_premises.strip() or not nl_question.strip():
        return "", "⚠️ Please fill in both fields.", ""

    try:
        _ensure_loaded()

        from scripts.pipeline import run
        result = run(nl_premises, nl_question)

        answer = result["answer"]
        if answer == "True":
            answer_display = "✅ True"
        elif answer == "False":
            answer_display = "❌ False"
        else:
            answer_display = "❓ Unknown"

        proof_text = "\n".join(result["proof"]) if result["proof"] else "(no proof steps)"
        fol_premises = result["fol_premises"] if isinstance(result["fol_premises"], str) else "\n".join(result["fol_premises"])
        fol_text   = fol_premises + "\n\nQuestion: " + result["fol_question"]

        return fol_text, answer_display, proof_text

    except Exception as e:
        import traceback
        return "", "❌ Error", traceback.format_exc()


# ── UI ────────────────────────────────────────────────────────────────────────

EXAMPLES = [
    [
        "Anne is kind. Bob is furry. If someone is kind then they are furry. If someone is furry then they are green.",
        "Anne is green.",
    ],
    [
        "Anne is kind. Bob is cold. If someone is kind and cold then they are smart. If someone is smart then they are young.",
        "Bob is young.",
    ],
    [
        "Anne is kind. Bob is furry.",
        "Anne is furry.",
    ],
]

with gr.Blocks(title="FOL SLM — Logical Reasoning") as demo:
    gr.Markdown(
        """
        # FOL SLM — Natural Language → Symbolic Reasoning

        Enter a set of **premises** (facts and rules in plain English) and a **question**.
        The model translates them into First-Order Logic, then uses a symbolic reasoner (clingo/ASP)
        to derive the answer with a step-by-step proof.
        """
    )

    with gr.Row():
        premises_box = gr.Textbox(
            label="Premises",
            placeholder="Anne is kind. If someone is kind then they are furry. ...",
            lines=5,
        )
        question_box = gr.Textbox(
            label="Question",
            placeholder="Anne is furry.",
            lines=5,
        )

    submit_btn = gr.Button("Run", variant="primary")

    fol_box   = gr.Textbox(label="FOL Translation", interactive=False, lines=6)
    answer_box = gr.Textbox(label="Answer", interactive=False)
    proof_box  = gr.Textbox(label="Proof", interactive=False, lines=5)

    submit_btn.click(
        fn=predict,
        inputs=[premises_box, question_box],
        outputs=[fol_box, answer_box, proof_box],
    )

    gr.Examples(
        examples=EXAMPLES,
        inputs=[premises_box, question_box],
        outputs=[fol_box, answer_box, proof_box],
        fn=predict,
        cache_examples=False,
    )

    gr.Markdown(
        """
        ---
        **Model**: T5-base encoder (frozen) + custom 4-layer 512d translation decoder,
        trained on ProofWriter (OWA) with randomised entity/predicate substitution.
        **Reasoner**: [clingo](https://potassco.org/clingo/) ASP solver with provenance tracking.
        """
    )

    demo.load(
        fn=None,
        js="() => { document.querySelectorAll('textarea').forEach(el => el.spellcheck = false); }",
    )

if __name__ == "__main__":
    demo.launch()
