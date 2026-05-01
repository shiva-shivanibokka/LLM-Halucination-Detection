"""
app.py

Gradio frontend for the LLM Eval Platform.

Tabs:
  1. New Benchmark  — upload a PDF, name it, auto-generate test cases, done
  2. My Benchmarks  — view existing benchmarks, manage test cases, add manually
  3. Run Eval       — pick a benchmark and model, launch a run, watch progress
  4. Results        — browse run history, per-question breakdown, domain scores
  5. Compare Models — diff two runs side by side

Run locally:
    uvicorn api.main:app --reload --port 8000
    python app.py
"""

import io
import os
import time

import gradio as gr
import requests as http_requests
from dotenv import load_dotenv

load_dotenv()

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

PROVIDER_MODELS: dict[str, list[str]] = {
    "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
    "anthropic": ["claude-opus-4-5", "claude-sonnet-4-5", "claude-haiku-4-5"],
    "groq": [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ],
    "mistral": ["mistral-large-latest", "mistral-small-latest", "open-mistral-7b"],
    "gemini": ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
    "ollama": [
        "llama3.2",
        "llama3.2:1b",
        "mistral",
        "deepseek-r1:1.5b",
        "deepseek-r1:7b",
        "phi3",
        "phi3:mini",
        "gemma3",
        "gemma3:4b",
    ],
}

PROVIDER_DISPLAY_TO_KEY: dict[str, str] = {k.upper(): k for k in PROVIDER_MODELS}
PROVIDER_DISPLAY_LABELS: list[str] = list(PROVIDER_DISPLAY_TO_KEY.keys())
NO_KEY_PROVIDERS = {"ollama"}

KEY_PLACEHOLDER: dict[str, str] = {
    "openai": "sk-...  (or set OPENAI_API_KEY in .env)",
    "anthropic": "sk-ant-...  (or set ANTHROPIC_API_KEY in .env)",
    "groq": "gsk_...  (or set GROQ_API_KEY in .env)",
    "mistral": "...  (or set MISTRAL_API_KEY in .env)",
    "gemini": "AI...  (or set GOOGLE_API_KEY in .env)",
    "ollama": "No API key needed — runs locally",
}

DOMAINS = ["general", "medical", "legal", "finance", "technical", "scientific", "other"]


def _get(path: str, params: dict = None) -> dict | list:
    try:
        r = http_requests.get(f"{API_BASE}{path}", params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except http_requests.HTTPError as e:
        detail = e.response.json().get("detail", str(e)) if e.response else str(e)
        raise gr.Error(f"API error: {detail}")
    except Exception as e:
        raise gr.Error(f"Connection error — is the FastAPI server running? {e}")


def _post(path: str, json: dict = None, files: dict = None) -> dict | list:
    try:
        if files:
            r = http_requests.post(f"{API_BASE}{path}", files=files, timeout=300)
        else:
            r = http_requests.post(f"{API_BASE}{path}", json=json, timeout=300)
        r.raise_for_status()
        return r.json()
    except http_requests.HTTPError as e:
        detail = e.response.json().get("detail", str(e)) if e.response else str(e)
        raise gr.Error(f"API error: {detail}")
    except Exception as e:
        raise gr.Error(f"Connection error — is the FastAPI server running? {e}")


def _delete(path: str) -> None:
    try:
        r = http_requests.delete(f"{API_BASE}{path}", timeout=30)
        r.raise_for_status()
    except http_requests.HTTPError as e:
        detail = e.response.json().get("detail", str(e)) if e.response else str(e)
        raise gr.Error(f"API error: {detail}")
    except Exception as e:
        raise gr.Error(str(e))


def _resolve_provider(display: str) -> str:
    return PROVIDER_DISPLAY_TO_KEY.get(display, display.lower())


def _api_key(raw: str) -> str | None:
    s = raw.strip()
    return None if not s or s == "Not required" else s


def _read_pdf(file_path: str) -> str:
    """Extract text from an uploaded PDF file."""
    from pypdf import PdfReader

    reader = PdfReader(file_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def update_models(display_provider: str):
    key = _resolve_provider(display_provider)
    models = PROVIDER_MODELS.get(key, [])
    # Always return both choices AND value together so Gradio never
    # tries to validate the old value against the new choices list.
    return gr.update(choices=models, value=models[0] if models else None)


def update_key_field(display_provider: str):
    key = _resolve_provider(display_provider)
    is_local = key in NO_KEY_PROVIDERS
    return gr.update(
        placeholder=KEY_PLACEHOLDER.get(key, "API key"),
        interactive=not is_local,
        value="" if not is_local else "Not required",
    )


def _benchmark_choices() -> list[str]:
    try:
        benchmarks = _get("/benchmarks")
        return [
            f"{b['id']} — {b['name']} ({b['case_count']} cases)" for b in benchmarks
        ]
    except Exception:
        return []


def _run_choices() -> list[str]:
    try:
        runs = _get("/runs")
        result = []
        for r in runs:
            score = (
                f"{r['avg_score']:.2f}" if r["avg_score"] is not None else r["status"]
            )
            result.append(
                f"Run {r['id']} | {r['model']} | {r['benchmark_name']} | {score}"
            )
        return result
    except Exception:
        return []


def _parse_benchmark_id(choice: str) -> int:
    return int(choice.split(" — ")[0])


def _parse_run_id(choice: str) -> int:
    return int(choice.split(" | ")[0].replace("Run ", ""))


# ---------------------------------------------------------------------------
# New Benchmark tab — the simple flow
# ---------------------------------------------------------------------------


def create_from_pdf_fn(
    pdf_file,
    benchmark_name: str,
    num_questions: int,
    domain: str,
    source_type: str,
    display_provider: str,
    model: str,
    api_key_raw: str,
):
    if pdf_file is None:
        return "Upload a PDF first.", "", gr.update()
    if not benchmark_name.strip():
        return "Give the benchmark a name.", "", gr.update()

    # Read PDF text locally
    try:
        doc_text = _read_pdf(pdf_file.name)
    except Exception as e:
        return f"Could not read PDF: {e}", "", gr.update()

    if len(doc_text.strip()) < 100:
        return (
            "PDF appears to be empty or unreadable (scanned image PDFs are not supported).",
            "",
            gr.update(),
        )

    # Create the benchmark
    try:
        bm = _post(
            "/benchmarks",
            {"name": benchmark_name.strip(), "description": f"Auto-generated from PDF"},
        )
    except gr.Error as e:
        return str(e), "", gr.update()

    benchmark_id = bm["id"]

    # Generate test cases
    provider = _resolve_provider(display_provider)
    try:
        result = _post(
            f"/benchmarks/{benchmark_id}/generate-cases",
            {
                "reference_text": doc_text,
                "num_cases": int(num_questions),
                "domain": domain,
                "source_type": source_type,
                "provider": provider,
                "model": model,
                "api_key": _api_key(api_key_raw),
            },
        )
    except gr.Error as e:
        return str(e), "", gr.update()

    questions_preview = "\n".join(
        f"{i + 1}. {q}" for i, q in enumerate(result["questions"])
    )
    status = (
        f"Benchmark '{benchmark_name.strip()}' created with {result['generated']} test cases.\n"
        f"Go to Run Eval to test a model against it."
    )
    return status, questions_preview, gr.update(choices=_benchmark_choices())


# ---------------------------------------------------------------------------
# My Benchmarks tab
# ---------------------------------------------------------------------------


def refresh_benchmarks_fn():
    return gr.update(choices=_benchmark_choices())


def delete_benchmark_fn(choice: str):
    if not choice:
        return "Select a benchmark first.", gr.update()
    bid = _parse_benchmark_id(choice)
    _delete(f"/benchmarks/{bid}")
    return "Benchmark deleted.", gr.update(choices=_benchmark_choices())


def load_cases_fn(choice: str):
    if not choice:
        return "Select a benchmark to see its test cases.", ""
    bid = _parse_benchmark_id(choice)
    cases = _get(f"/benchmarks/{bid}/cases")
    if not cases:
        return "No test cases yet.", ""
    lines = [
        f"[{c['id']}] [{c['domain'].upper()}] [{c['source_type'].upper()}] {c['question']}"
        for c in cases
    ]
    return f"{len(cases)} test cases loaded", "\n".join(lines)


def add_case_fn(
    choice: str, question: str, reference_text: str, domain: str, source_type: str
):
    if not choice:
        return "Select a benchmark first."
    if not question.strip() or not reference_text.strip():
        return "Both question and reference text are required."
    bid = _parse_benchmark_id(choice)
    _post(
        f"/benchmarks/{bid}/cases",
        {
            "question": question.strip(),
            "reference_text": reference_text.strip(),
            "domain": domain,
            "source_type": source_type,
        },
    )
    return "Test case added."


def delete_case_fn(case_id_str: str):
    try:
        case_id = int(case_id_str.strip())
    except ValueError:
        return "Enter a valid case ID (the number in brackets when you load cases)."
    _delete(f"/cases/{case_id}")
    return f"Case {case_id} deleted."


# ---------------------------------------------------------------------------
# Run Eval tab
# ---------------------------------------------------------------------------


def start_run_fn(
    benchmark_choice: str,
    display_provider: str,
    model: str,
    api_key_raw: str,
    entail: float,
    contradict: float,
    grounded_ceil: float,
    partial_ceil: float,
):
    if not benchmark_choice:
        return "Select a benchmark first.", gr.update()
    bid = _parse_benchmark_id(benchmark_choice)
    provider = _resolve_provider(display_provider)
    result = _post(
        "/runs",
        {
            "benchmark_id": bid,
            "provider": provider,
            "model": model,
            "api_key": _api_key(api_key_raw),
            "entail_threshold": entail,
            "contradict_threshold": contradict,
            "grounded_ceiling": grounded_ceil,
            "partial_ceiling": partial_ceil,
        },
    )
    run_id = result["run_id"]

    for _ in range(240):
        time.sleep(5)
        try:
            run = _get(f"/runs/{run_id}")
            completed = run.get("completed_cases", 0)
            total = run.get("total_cases", 1)
            status = run.get("status", "running")
            pct = completed / total * 100 if total else 0
            lines = [
                f"Status: {status.upper()} — {completed}/{total} questions scored ({pct:.0f}%)"
            ]
            if run.get("avg_score") is not None:
                lines.append(f"Hallucination score: {run['avg_score']:.0%}")
                lines.append(f"Grounded rate: {run['grounded_pct']:.0%}")
            if status in ("completed", "failed"):
                if status == "completed":
                    lines.append(
                        "Done. Go to the Results tab to see the full breakdown."
                    )
                break
        except Exception:
            break

    return "\n".join(lines), gr.update(choices=_run_choices())


# ---------------------------------------------------------------------------
# Results tab
# ---------------------------------------------------------------------------


def load_run_results_fn(run_choice: str):
    if not run_choice:
        return "Select a run first.", "", ""
    run_id = _parse_run_id(run_choice)
    run = _get(f"/runs/{run_id}")
    results = _get(f"/runs/{run_id}/results")
    domains = _get(f"/runs/{run_id}/domains")

    summary_lines = [
        f"Model: {run['model']}   Benchmark: {run['benchmark_name']}",
        f"Status: {run['status'].upper()}   {run.get('completed_cases', 0)}/{run.get('total_cases', 0)} questions",
        f"Hallucination score: {run['avg_score']:.0%}"
        if run["avg_score"] is not None
        else "Score: N/A",
        f"Grounded rate: {run['grounded_pct']:.0%}"
        if run["grounded_pct"] is not None
        else "",
    ]

    domain_lines = []
    for d in domains:
        domain_lines.append(
            f"{d['domain'].upper()}: {d['avg_score']:.0%} hallucination | "
            f"{d['grounded']} grounded / {d['hallucinated']} hallucinated / {d['total']} total"
        )

    case_lines = []
    for r in results:
        bar = "█" * int(r["hallucination_score"] * 10) + "░" * (
            10 - int(r["hallucination_score"] * 10)
        )
        case_lines.append(
            f"Q: {r['question']}\n"
            f"   Verdict: {r['overall_label']}  [{bar}]  {r['hallucination_score']:.0%} hallucination\n"
            f"   Answer:  {r['response'][:150]}{'...' if len(r['response']) > 150 else ''}\n"
        )

    return "\n".join(summary_lines), "\n".join(domain_lines), "\n".join(case_lines)


# ---------------------------------------------------------------------------
# Compare tab
# ---------------------------------------------------------------------------


def compare_runs_fn(choice_a: str, choice_b: str):
    if not choice_a or not choice_b:
        return "Select both runs to compare."
    if choice_a == choice_b:
        return "Select two different runs."
    run_a = _parse_run_id(choice_a)
    run_b = _parse_run_id(choice_b)
    data = _get("/compare", {"run_a": run_a, "run_b": run_b})

    ra = data["run_a"]
    rb = data["run_b"]
    delta = data["overall_delta"]
    direction = "better" if delta < 0 else "worse" if delta > 0 else "unchanged"

    lines = [
        f"Run A: {ra['model']}  on  {ra['benchmark_name']}",
        f"Run B: {rb['model']}  on  {rb['benchmark_name']}",
        "",
        f"Hallucination score   A: {data['avg_score_a']:.0%}   B: {data['avg_score_b']:.0%}",
        f"Overall delta: {delta:+.0%}  (Run B is {direction})",
        f"Improved: {data['improved_count']}   Regressed: {data['regressed_count']}   Stable: {data['stable_count']}",
    ]

    # Source type split
    lines += ["", "Score by document source:"]
    for label, st_key in [
        ("Internal (reliable)", "internal"),
        ("Public (contamination risk)", "public"),
    ]:
        sa = data["source_type_scores_a"].get(st_key)
        sb = data["source_type_scores_b"].get(st_key)
        if sa or sb:
            a_str = f"{sa['avg_score']:.0%}" if sa else "n/a"
            b_str = f"{sb['avg_score']:.0%}" if sb else "n/a"
            n = (sa["total"] if sa else 0) or (sb["total"] if sb else 0)
            lines.append(f"  {label} ({n} questions): A={a_str}  B={b_str}")
            if st_key == "public":
                lines.append(
                    "  (Public scores may be inflated — model may have seen this content in training)"
                )

    lines += ["", "Per-question breakdown:"]
    for c in data["per_case"]:
        flag = " [PUBLIC]" if c["source_type"] == "public" else ""
        lines.append(
            f"  {c['question'][:90]}{flag}\n"
            f"    A: {c['label_a']} {c['score_a']:.0%}   B: {c['label_b']} {c['score_b']:.0%}   {c['verdict'].upper()} ({c['delta']:+.0%})"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="LLM Eval Platform",
        theme=gr.themes.Soft(),
        css="footer { display: none !important; }",
    ) as demo:
        gr.Markdown("# LLM Eval Platform")
        gr.Markdown(
            "Test any LLM for hallucination on your own documents. "
            "Upload a PDF, generate questions from it, run a model, see where it makes things up."
        )

        # Global LLM settings — used everywhere
        with gr.Group():
            gr.Markdown(
                "**LLM Settings** — applies to question generation and eval runs"
            )
            with gr.Row():
                g_provider = gr.Dropdown(
                    choices=PROVIDER_DISPLAY_LABELS,
                    value="OPENAI",
                    label="Provider",
                    scale=1,
                )
                g_model = gr.Dropdown(
                    choices=PROVIDER_MODELS["openai"],
                    value=PROVIDER_MODELS["openai"][0],
                    label="Model",
                    scale=2,
                    allow_custom_value=True,  # prevents validation error during provider switch
                )
                g_api_key = gr.Textbox(
                    label="API Key",
                    placeholder=KEY_PLACEHOLDER["openai"],
                    type="password",
                    scale=3,
                )
            g_provider.change(fn=update_models, inputs=g_provider, outputs=g_model)
            g_provider.change(fn=update_key_field, inputs=g_provider, outputs=g_api_key)

        with gr.Accordion(
            "Detection Thresholds — leave as default unless you know what you're doing",
            open=False,
        ):
            gr.Markdown(
                "These control how strict the hallucination scorer is. "
                "Higher entailment threshold = harder for a sentence to be called grounded. "
                "The defaults work well for most cases."
            )
            with gr.Row():
                entail_s = gr.Slider(
                    0.1,
                    0.9,
                    value=0.5,
                    step=0.05,
                    label="Entailment threshold (how confident the model must be that the sentence is supported)",
                )
                contradict_s = gr.Slider(
                    0.1,
                    0.9,
                    value=0.5,
                    step=0.05,
                    label="Contradiction threshold (how confident before marking as contradicted)",
                )
            with gr.Row():
                grounded_s = gr.Slider(
                    0.1,
                    0.6,
                    value=0.3,
                    step=0.05,
                    label="Overall grounded ceiling (overall score below this = GROUNDED verdict)",
                )
                partial_s = gr.Slider(
                    0.3,
                    0.9,
                    value=0.6,
                    step=0.05,
                    label="Overall partial ceiling (below this = PARTIALLY GROUNDED, above = HALLUCINATED)",
                )

        with gr.Tabs():
            # ------------------------------------------------------------------
            with gr.Tab("New Benchmark"):
                gr.Markdown(
                    "The fastest way to get started. Upload a PDF, give it a name, "
                    "and the system will automatically generate factual questions from it. "
                    "Those questions become your benchmark — a reusable test set you can run any model against."
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        nb_pdf = gr.File(label="Upload PDF", file_types=[".pdf"])
                        nb_name = gr.Textbox(
                            label="Benchmark name", placeholder="e.g. File System QA"
                        )
                        nb_num = gr.Slider(
                            3,
                            30,
                            value=10,
                            step=1,
                            label="Number of test questions to generate",
                        )
                        with gr.Row():
                            nb_domain = gr.Dropdown(
                                choices=DOMAINS, value="general", label="Domain"
                            )
                            nb_source_type = gr.Radio(
                                choices=["internal", "public"],
                                value="internal",
                                label="Document source",
                                info="Internal = private doc. Public = Wikipedia, papers, etc.",
                            )
                        nb_create_btn = gr.Button(
                            "Create benchmark from PDF", variant="primary", size="lg"
                        )

                    with gr.Column(scale=1):
                        nb_status = gr.Textbox(
                            label="Status", lines=3, interactive=False
                        )
                        nb_preview = gr.Textbox(
                            label="Generated questions preview",
                            lines=12,
                            interactive=False,
                        )
                        nb_bm_update = gr.State()

                nb_create_btn.click(
                    fn=create_from_pdf_fn,
                    inputs=[
                        nb_pdf,
                        nb_name,
                        nb_num,
                        nb_domain,
                        nb_source_type,
                        g_provider,
                        g_model,
                        g_api_key,
                    ],
                    outputs=[nb_status, nb_preview, nb_bm_update],
                )

            # ------------------------------------------------------------------
            with gr.Tab("My Benchmarks"):
                gr.Markdown(
                    "View and manage your benchmarks. "
                    "You can also add test cases manually here if you prefer to write your own questions."
                )

                with gr.Row():
                    bm_list = gr.Dropdown(
                        choices=_benchmark_choices(),
                        label="Select a benchmark",
                        scale=3,
                        interactive=True,
                    )
                    bm_refresh_btn = gr.Button("Refresh", size="sm", scale=0)
                    bm_delete_btn = gr.Button(
                        "Delete", size="sm", variant="stop", scale=0
                    )

                bm_action_out = gr.Markdown()
                bm_refresh_btn.click(fn=refresh_benchmarks_fn, outputs=bm_list)
                bm_delete_btn.click(
                    fn=delete_benchmark_fn,
                    inputs=bm_list,
                    outputs=[bm_action_out, bm_list],
                )

                with gr.Row():
                    cases_load_btn = gr.Button("Load test cases", size="sm")
                    cases_count_out = gr.Markdown()
                cases_display = gr.Textbox(
                    label="Test cases", lines=10, interactive=False
                )
                cases_load_btn.click(
                    fn=load_cases_fn,
                    inputs=bm_list,
                    outputs=[cases_count_out, cases_display],
                )

                gr.Markdown("**Add a test case manually**")
                with gr.Row():
                    with gr.Column():
                        man_question = gr.Textbox(
                            label="Question",
                            lines=2,
                            placeholder="e.g. What is the block size in the file system?",
                        )
                        man_ref = gr.Textbox(
                            label="Reference text (the document passage that contains the correct answer)",
                            lines=5,
                            placeholder="Paste the relevant section of your document here...",
                        )
                        with gr.Row():
                            man_domain = gr.Dropdown(
                                choices=DOMAINS, value="general", label="Domain"
                            )
                            man_source_type = gr.Radio(
                                choices=["internal", "public"],
                                value="internal",
                                label="Source type",
                            )
                        man_add_btn = gr.Button("Add test case", variant="primary")
                        man_add_out = gr.Markdown()
                        man_add_btn.click(
                            fn=add_case_fn,
                            inputs=[
                                bm_list,
                                man_question,
                                man_ref,
                                man_domain,
                                man_source_type,
                            ],
                            outputs=man_add_out,
                        )

                gr.Markdown("**Delete a test case**")
                with gr.Row():
                    del_case_id = gr.Textbox(
                        label="Case ID (shown in brackets when you load cases)", scale=1
                    )
                    del_case_btn = gr.Button("Delete case", variant="stop", scale=0)
                del_case_out = gr.Markdown()
                del_case_btn.click(
                    fn=delete_case_fn, inputs=del_case_id, outputs=del_case_out
                )

            # ------------------------------------------------------------------
            with gr.Tab("Run Eval"):
                gr.Markdown(
                    "Pick a benchmark and click Start. "
                    "The system sends each question to the selected model, gets its answer, "
                    "and scores every sentence against the reference document using the NLI model. "
                    "Results are saved and available in the Results tab."
                )
                with gr.Row():
                    run_bm_choice = gr.Dropdown(
                        choices=_benchmark_choices(),
                        label="Benchmark to evaluate",
                        scale=3,
                    )
                    run_bm_refresh = gr.Button("Refresh", size="sm", scale=0)
                run_bm_refresh.click(
                    fn=lambda: gr.update(choices=_benchmark_choices()),
                    outputs=run_bm_choice,
                )

                run_btn = gr.Button("Start Eval Run", variant="primary", size="lg")
                run_status_out = gr.Textbox(
                    label="Progress", lines=5, interactive=False
                )
                run_refresh_state = gr.State()

                run_btn.click(
                    fn=start_run_fn,
                    inputs=[
                        run_bm_choice,
                        g_provider,
                        g_model,
                        g_api_key,
                        entail_s,
                        contradict_s,
                        grounded_s,
                        partial_s,
                    ],
                    outputs=[run_status_out, run_refresh_state],
                )

            # ------------------------------------------------------------------
            with gr.Tab("Results"):
                gr.Markdown(
                    "Browse all completed eval runs. "
                    "Select a run and load it to see the overall score, domain breakdown, "
                    "and what the model said for each question."
                )
                with gr.Row():
                    results_run_choice = gr.Dropdown(
                        choices=_run_choices(), label="Select run", scale=3
                    )
                    results_refresh_btn = gr.Button("Refresh", size="sm", scale=0)
                results_refresh_btn.click(
                    fn=lambda: gr.update(choices=_run_choices()),
                    outputs=results_run_choice,
                )

                load_results_btn = gr.Button("Load results", variant="primary")
                results_summary = gr.Textbox(
                    label="Summary", lines=4, interactive=False
                )
                results_domains = gr.Textbox(
                    label="Domain breakdown", lines=6, interactive=False
                )
                results_cases = gr.Textbox(
                    label="Per-question results", lines=24, interactive=False
                )
                load_results_btn.click(
                    fn=load_run_results_fn,
                    inputs=results_run_choice,
                    outputs=[results_summary, results_domains, results_cases],
                )

            # ------------------------------------------------------------------
            with gr.Tab("Compare Models"):
                gr.Markdown(
                    "Run the same benchmark against two different models, then compare here. "
                    "The report shows which model hallucinated more, which questions flipped, "
                    "and flags any results from public documents that may not be trustworthy."
                )
                with gr.Row():
                    compare_a = gr.Dropdown(
                        choices=_run_choices(), label="Run A (baseline model)", scale=2
                    )
                    compare_b = gr.Dropdown(
                        choices=_run_choices(),
                        label="Run B (model to compare)",
                        scale=2,
                    )
                    compare_refresh = gr.Button("Refresh", size="sm", scale=0)
                compare_refresh.click(
                    fn=lambda: (
                        gr.update(choices=_run_choices()),
                        gr.update(choices=_run_choices()),
                    ),
                    outputs=[compare_a, compare_b],
                )
                compare_btn = gr.Button("Compare", variant="primary")
                compare_out = gr.Textbox(
                    label="Comparison report", lines=30, interactive=False
                )
                compare_btn.click(
                    fn=compare_runs_fn,
                    inputs=[compare_a, compare_b],
                    outputs=compare_out,
                )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
    )
