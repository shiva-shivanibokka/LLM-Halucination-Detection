"""
app.py

Gradio frontend for the LLM Hallucination Detection system.

Tabs:
  1. Source Documents  — ingest text / PDF / URL into the vector store
  2. Analyze Response  — paste any LLM response and analyze it sentence by sentence
  3. Full Demo         — pick a question, generate grounded vs ungrounded, compare both

Run locally:
    uvicorn api.main:app --reload --port 8000 &
    python app.py
"""

import os
import textwrap
from typing import Optional

import gradio as gr
import requests as http_requests
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# API base URL — point at the running FastAPI server
# ---------------------------------------------------------------------------
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")


# ---------------------------------------------------------------------------
# Helpers — thin wrappers around the FastAPI endpoints
# ---------------------------------------------------------------------------


def _post(path: str, json: dict = None, files: dict = None) -> dict:
    try:
        if files:
            r = http_requests.post(f"{API_BASE}{path}", files=files, timeout=120)
        else:
            r = http_requests.post(f"{API_BASE}{path}", json=json, timeout=120)
        r.raise_for_status()
        return r.json()
    except http_requests.HTTPError as e:
        detail = e.response.json().get("detail", str(e)) if e.response else str(e)
        raise gr.Error(f"API error: {detail}")
    except Exception as e:
        raise gr.Error(f"Connection error — is the FastAPI server running? {e}")


def _delete(path: str) -> dict:
    try:
        r = http_requests.delete(f"{API_BASE}{path}", timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise gr.Error(str(e))


def _chunk_count() -> int:
    try:
        r = http_requests.get(f"{API_BASE}/health", timeout=10)
        return r.json().get("chunks_in_store", 0)
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Label colours
# ---------------------------------------------------------------------------

LABEL_COLOURS = {
    "GROUNDED": "#2d9e5f",
    "UNGROUNDED": "#e07b00",
    "CONTRADICTED": "#cc2f2f",
    "PARTIALLY_GROUNDED": "#e07b00",
    "HALLUCINATED": "#cc2f2f",
}

LABEL_ICONS = {
    "GROUNDED": "✅",
    "UNGROUNDED": "⚠️",
    "CONTRADICTED": "❌",
    "PARTIALLY_GROUNDED": "⚠️",
    "HALLUCINATED": "❌",
}


def _colour(label: str) -> str:
    return LABEL_COLOURS.get(label, "#888888")


def _icon(label: str) -> str:
    return LABEL_ICONS.get(label, "❓")


# ---------------------------------------------------------------------------
# Format analysis results as readable Markdown
# ---------------------------------------------------------------------------


def _format_analysis(analysis: dict, response_text: str) -> str:
    overall = analysis["overall_label"]
    score = analysis["overall_hallucination_score"]
    total = analysis["total_sentences"]
    grounded = analysis["grounded_count"]
    ungrounded = analysis["ungrounded_count"]
    contradicted = analysis["contradicted_count"]

    icon = _icon(overall)
    lines = [
        f"## {icon} Overall: **{overall}**",
        f"**Hallucination Score:** `{score:.2%}` &nbsp;|&nbsp; "
        f"**Sentences:** {total} total — "
        f"✅ {grounded} grounded, ⚠️ {ungrounded} ungrounded, ❌ {contradicted} contradicted",
        "",
        "---",
        "### Sentence-by-sentence breakdown",
        "",
    ]

    for i, sr in enumerate(analysis["sentence_results"], 1):
        s_icon = _icon(sr["label"])
        score_bar = "█" * int(sr["hallucination_score"] * 10) + "░" * (
            10 - int(sr["hallucination_score"] * 10)
        )
        lines += [
            f"**{i}. {s_icon} `{sr['label']}`** &nbsp; score: `{sr['hallucination_score']:.2%}`",
            f"> {sr['sentence']}",
            "",
            f"&nbsp;&nbsp;`[{score_bar}]` &nbsp; "
            f"entailment: `{sr['entailment_score']:.3f}` &nbsp; "
            f"contradiction: `{sr['contradiction_score']:.3f}` &nbsp; "
            f"neutral: `{sr['neutral_score']:.3f}`",
        ]
        if sr["best_source_chunk"]:
            chunk_preview = textwrap.shorten(
                sr["best_source_chunk"], width=150, placeholder="…"
            )
            lines += [
                f"&nbsp;&nbsp;*Best matching source chunk (similarity `{sr['best_source_similarity']:.3f}`):*",
                f"&nbsp;&nbsp;> {chunk_preview}",
            ]
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tab 1 — Source Documents
# ---------------------------------------------------------------------------


def ingest_text_fn(text: str) -> str:
    if not text.strip():
        return "Please enter some text."
    result = _post("/ingest/text", json={"text": text, "label": "pasted_text"})
    return f"✅ Stored **{result['chunks_stored']}** chunks. Total in store: **{result['total_chunks']}**."


def ingest_pdf_fn(file) -> str:
    if file is None:
        return "Please upload a PDF."
    with open(file.name, "rb") as f:
        pdf_bytes = f.read()
    fname = os.path.basename(file.name)
    result = _post("/ingest/pdf", files={"file": (fname, pdf_bytes, "application/pdf")})
    return f"✅ Stored **{result['chunks_stored']}** chunks from **{fname}**. Total: **{result['total_chunks']}**."


def ingest_url_fn(url: str) -> str:
    if not url.strip():
        return "Please enter a URL."
    result = _post("/ingest/url", json={"url": url.strip()})
    return f"✅ Stored **{result['chunks_stored']}** chunks from URL. Total: **{result['total_chunks']}**."


def reset_fn() -> str:
    _delete("/reset")
    return "🗑️ All source documents cleared."


def status_fn() -> str:
    n = _chunk_count()
    return f"📚 Source store contains **{n}** chunks."


# ---------------------------------------------------------------------------
# Tab 2 — Analyze a Response
# ---------------------------------------------------------------------------


def analyze_fn(response_text: str) -> str:
    if not response_text.strip():
        return "Please paste an LLM response to analyze."
    result = _post("/analyze", json={"response": response_text})
    return _format_analysis(result, response_text)


# ---------------------------------------------------------------------------
# Tab 3 — Full Demo
# ---------------------------------------------------------------------------


def full_demo_fn(
    question: str,
    provider: str,
    openai_key: str,
    anthropic_key: str,
) -> tuple[str, str, str, str]:
    if not question.strip():
        raise gr.Error("Please enter a question.")

    payload = {
        "question": question.strip(),
        "provider": provider.lower(),
        "openai_api_key": openai_key.strip() or None,
        "anthropic_api_key": anthropic_key.strip() or None,
    }
    result = _post("/full-pipeline", json=payload)

    grounded_md = _format_analysis(
        result["grounded_analysis"], result["grounded_response"]
    )
    ungrounded_md = _format_analysis(
        result["ungrounded_analysis"], result["ungrounded_response"]
    )

    return (
        result["grounded_response"],
        result["ungrounded_response"],
        grounded_md,
        ungrounded_md,
    )


# ---------------------------------------------------------------------------
# Build the Gradio UI
# ---------------------------------------------------------------------------


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="LLM Hallucination Detection",
        theme=gr.themes.Soft(),
        css="""
            .label-grounded { color: #2d9e5f; font-weight: bold; }
            .label-ungrounded { color: #e07b00; font-weight: bold; }
            .label-contradicted { color: #cc2f2f; font-weight: bold; }
            footer { display: none !important; }
        """,
    ) as demo:
        gr.Markdown(
            """
            # LLM Hallucination Detection & Guardrails System
            **NLI-based hallucination detection using DeBERTa-v3.**
            Flags unsupported or contradicted claims in LLM responses sentence by sentence.

            > **How it works:** Source documents are embedded with sentence-transformers and stored in ChromaDB.
            > For each sentence in the LLM response, the top-K most relevant source chunks are retrieved
            > and passed through a DeBERTa-v3 NLI model. Sentences are classified as
            > ✅ GROUNDED, ⚠️ UNGROUNDED, or ❌ CONTRADICTED based on entailment scores.
            """
        )

        # ---------------------------------------------------------------
        # Tab 1 — Source Documents
        # ---------------------------------------------------------------
        with gr.Tab("📚 Source Documents"):
            gr.Markdown(
                "### Load source documents for grounding\nIngest text, a PDF, or a web page URL. All content is chunked and embedded into the vector store."
            )

            with gr.Row():
                status_box = gr.Markdown(
                    value=f"📚 Source store contains **{_chunk_count()}** chunks."
                )
                refresh_btn = gr.Button("🔄 Refresh status", size="sm")

            refresh_btn.click(fn=status_fn, outputs=status_box)

            with gr.Tabs():
                with gr.Tab("Paste Text"):
                    text_input = gr.Textbox(
                        label="Source text",
                        placeholder="Paste any document, article, or passage here...",
                        lines=10,
                    )
                    text_btn = gr.Button("Ingest Text", variant="primary")
                    text_out = gr.Markdown()
                    text_btn.click(
                        fn=ingest_text_fn, inputs=text_input, outputs=text_out
                    )

                with gr.Tab("Upload PDF"):
                    pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
                    pdf_btn = gr.Button("Ingest PDF", variant="primary")
                    pdf_out = gr.Markdown()
                    pdf_btn.click(fn=ingest_pdf_fn, inputs=pdf_input, outputs=pdf_out)

                with gr.Tab("Scrape URL"):
                    url_input = gr.Textbox(
                        label="URL",
                        placeholder="https://en.wikipedia.org/wiki/...",
                    )
                    url_btn = gr.Button("Scrape & Ingest", variant="primary")
                    url_out = gr.Markdown()
                    url_btn.click(fn=ingest_url_fn, inputs=url_input, outputs=url_out)

            gr.Markdown("---")
            reset_btn = gr.Button("🗑️ Clear all source documents", variant="stop")
            reset_out = gr.Markdown()
            reset_btn.click(fn=reset_fn, outputs=reset_out)

        # ---------------------------------------------------------------
        # Tab 2 — Analyze a Response
        # ---------------------------------------------------------------
        with gr.Tab("🔍 Analyze Response"):
            gr.Markdown(
                "### Analyze any LLM response\n"
                "Paste an LLM-generated response below. The pipeline will check each "
                "sentence against your loaded source documents and flag hallucinations."
            )
            response_input = gr.Textbox(
                label="LLM Response to analyze",
                placeholder="Paste the LLM response here...",
                lines=8,
            )
            analyze_btn = gr.Button("Analyze for Hallucinations", variant="primary")
            analysis_out = gr.Markdown(label="Analysis Results")
            analyze_btn.click(
                fn=analyze_fn, inputs=response_input, outputs=analysis_out
            )

        # ---------------------------------------------------------------
        # Tab 3 — Full Demo
        # ---------------------------------------------------------------
        with gr.Tab("⚡ Full Demo"):
            gr.Markdown(
                "### Grounded vs Ungrounded — side by side\n"
                "Enter a question. The pipeline will generate two responses — one grounded "
                "in your source documents, one without any grounding — then analyze both. "
                "This demonstrates the hallucination risk of ungrounded LLM responses.\n\n"
                "**Source documents must be loaded first (Tab 1).**"
            )

            with gr.Row():
                provider_dd = gr.Dropdown(
                    choices=["OpenAI", "Anthropic"],
                    value="OpenAI",
                    label="LLM Provider",
                    scale=1,
                )
                openai_key_input = gr.Textbox(
                    label="OpenAI API Key",
                    placeholder="sk-... (or set OPENAI_API_KEY env var)",
                    type="password",
                    scale=2,
                )
                anthropic_key_input = gr.Textbox(
                    label="Anthropic API Key",
                    placeholder="sk-ant-... (or set ANTHROPIC_API_KEY env var)",
                    type="password",
                    scale=2,
                )

            question_input = gr.Textbox(
                label="Question",
                placeholder="e.g. What are the main findings of the study?",
                lines=2,
            )
            demo_btn = gr.Button("Generate & Analyze Both Responses", variant="primary")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### ✅ Grounded Response (uses source docs)")
                    grounded_response_box = gr.Textbox(
                        label="Response", lines=6, interactive=False
                    )
                    grounded_analysis_box = gr.Markdown(label="Analysis")

                with gr.Column():
                    gr.Markdown("#### ⚠️ Ungrounded Response (no source docs)")
                    ungrounded_response_box = gr.Textbox(
                        label="Response", lines=6, interactive=False
                    )
                    ungrounded_analysis_box = gr.Markdown(label="Analysis")

            demo_btn.click(
                fn=full_demo_fn,
                inputs=[
                    question_input,
                    provider_dd,
                    openai_key_input,
                    anthropic_key_input,
                ],
                outputs=[
                    grounded_response_box,
                    ungrounded_response_box,
                    grounded_analysis_box,
                    ungrounded_analysis_box,
                ],
            )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ui = build_ui()
    ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
