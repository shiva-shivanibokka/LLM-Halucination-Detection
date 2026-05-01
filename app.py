"""
app.py

Gradio frontend for the LLM Eval Platform.

Tabs:
  1. Benchmarks   — create benchmarks, add/import/generate test cases
  2. Run Eval     — pick a benchmark and model, launch a run, watch progress
  3. Results      — browse run history, view per-question breakdown and domain scores
  4. Compare      — diff two runs side by side

Run locally:
    uvicorn api.main:app --reload --port 8000
    python app.py
"""

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


def _post(path: str, json: dict = None) -> dict | list:
    try:
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


def update_models(display_provider: str):
    key = _resolve_provider(display_provider)
    models = PROVIDER_MODELS.get(key, [])
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


def create_benchmark_fn(name: str, description: str):
    if not name.strip():
        return "Benchmark name is required.", gr.update()
    result = _post(
        "/benchmarks", {"name": name.strip(), "description": description.strip()}
    )
    return f"Benchmark '{result['name']}' created (ID {result['id']}).", gr.update(
        choices=_benchmark_choices()
    )


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
        return "Select a benchmark first.", ""
    bid = _parse_benchmark_id(choice)
    cases = _get(f"/benchmarks/{bid}/cases")
    if not cases:
        return "No test cases yet.", ""
    lines = [f"[{c['id']}] [{c['domain'].upper()}] {c['question']}" for c in cases]
    return f"{len(cases)} test cases", "\n".join(lines)


def add_case_fn(
    choice: str, question: str, reference_text: str, domain: str, source_type: str
):
    if not choice:
        return "Select a benchmark first."
    if not question.strip() or not reference_text.strip():
        return "Question and reference text are required."
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


def bulk_import_fn(choice: str, csv_text: str):
    if not choice:
        return "Select a benchmark first."
    if not csv_text.strip():
        return "Paste CSV content above."
    bid = _parse_benchmark_id(choice)
    result = _post(f"/benchmarks/{bid}/cases/bulk", {"csv_text": csv_text})
    return f"Imported {result['added']} test cases."


def generate_cases_fn(
    choice: str,
    reference_text: str,
    num_cases: int,
    domain: str,
    source_type: str,
    display_provider: str,
    model: str,
    api_key_raw: str,
):
    if not choice:
        return "Select a benchmark first.", ""
    if not reference_text.strip():
        return "Paste a reference document first.", ""
    bid = _parse_benchmark_id(choice)
    provider = _resolve_provider(display_provider)
    result = _post(
        f"/benchmarks/{bid}/generate-cases",
        {
            "reference_text": reference_text.strip(),
            "num_cases": int(num_cases),
            "domain": domain,
            "source_type": source_type,
            "provider": provider,
            "model": model,
            "api_key": _api_key(api_key_raw),
        },
    )
    questions_text = "\n".join(
        f"{i + 1}. {q}" for i, q in enumerate(result["questions"])
    )
    return f"Generated and saved {result['generated']} test cases.", questions_text


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
    status_lines = [f"Run {run_id} started. {result['message']}"]

    for _ in range(240):
        time.sleep(5)
        try:
            run = _get(f"/runs/{run_id}")
            completed = run.get("completed_cases", 0)
            total = run.get("total_cases", 1)
            status = run.get("status", "running")
            pct = completed / total * 100 if total else 0
            status_lines = [
                f"Status: {status.upper()} — {completed}/{total} cases ({pct:.0f}%)"
            ]
            if run.get("avg_score") is not None:
                status_lines.append(f"Avg hallucination score: {run['avg_score']:.2%}")
                status_lines.append(f"Grounded rate: {run['grounded_pct']:.0%}")
            if status in ("completed", "failed"):
                break
        except Exception:
            break

    return "\n".join(status_lines), gr.update(choices=_run_choices())


def load_run_results_fn(run_choice: str):
    if not run_choice:
        return "Select a run first.", "", ""
    run_id = _parse_run_id(run_choice)
    run = _get(f"/runs/{run_id}")
    results = _get(f"/runs/{run_id}/results")
    domains = _get(f"/runs/{run_id}/domains")

    summary_lines = [
        f"Run {run_id} — {run['model']} on {run['benchmark_name']}",
        f"Status: {run['status'].upper()}",
        f"Avg hallucination score: {run['avg_score']:.2%}"
        if run["avg_score"] is not None
        else "Score: N/A",
        f"Grounded rate: {run['grounded_pct']:.0%}"
        if run["grounded_pct"] is not None
        else "",
        f"{run.get('completed_cases', 0)}/{run.get('total_cases', 0)} cases completed",
    ]

    domain_lines = ["Domain breakdown:"]
    for d in domains:
        domain_lines.append(
            f"  {d['domain'].upper()}: avg {d['avg_score']:.2%} | "
            f"{d['grounded']} grounded / {d['hallucinated']} hallucinated / {d['total']} total"
        )

    case_lines = []
    for r in results:
        bar = "█" * int(r["hallucination_score"] * 10) + "░" * (
            10 - int(r["hallucination_score"] * 10)
        )
        case_lines.append(
            f"[{r['domain'].upper()}] {r['question']}\n"
            f"  {r['overall_label']} [{bar}] {r['hallucination_score']:.0%}\n"
            f"  Response: {r['response'][:120]}{'...' if len(r['response']) > 120 else ''}\n"
        )

    return "\n".join(summary_lines), "\n".join(domain_lines), "\n".join(case_lines)


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
        f"Run A: {ra['model']} on {ra['benchmark_name']}",
        f"Run B: {rb['model']} on {rb['benchmark_name']}",
        "",
        f"Overall avg score  A: {data['avg_score_a']:.2%}   B: {data['avg_score_b']:.2%}",
        f"Overall delta: {delta:+.2%}  (Run B is {direction})",
        f"Improved: {data['improved_count']}   Regressed: {data['regressed_count']}   Stable: {data['stable_count']}",
    ]

    # Source type reliability breakdown
    lines += ["", "Reliability breakdown by document source:"]
    for label, st_key in [
        ("Internal documents (reliable)", "internal"),
        ("Public documents (contamination risk)", "public"),
    ]:
        sa = data["source_type_scores_a"].get(st_key)
        sb = data["source_type_scores_b"].get(st_key)
        if sa or sb:
            score_a_str = f"{sa['avg_score']:.2%}" if sa else "n/a"
            score_b_str = f"{sb['avg_score']:.2%}" if sb else "n/a"
            total = (sa["total"] if sa else 0) or (sb["total"] if sb else 0)
            lines.append(f"  {label} ({total} cases)")
            lines.append(f"    A: {score_a_str}   B: {score_b_str}")
            if st_key == "public":
                lines.append(
                    "    Note: public document scores may be inflated by training data memorization."
                )

    lines += ["", "Per-question breakdown:"]
    for c in data["per_case"]:
        src_flag = " [PUBLIC]" if c["source_type"] == "public" else ""
        lines.append(
            f"  [{c['domain'].upper()}]{src_flag} {c['question'][:80]}\n"
            f"    A: {c['label_a']} {c['score_a']:.0%}   "
            f"B: {c['label_b']} {c['score_b']:.0%}   "
            f"{c['verdict'].upper()} ({c['delta']:+.2%})"
        )

    return "\n".join(lines)


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="LLM Eval Platform",
        theme=gr.themes.Soft(),
        css="""
            #page-subtitle { color: #888; font-size: 0.9em; margin-top: 0; }
            footer { display: none !important; }
        """,
    ) as demo:
        gr.Markdown("# LLM Eval Platform")
        gr.Markdown(
            "Create benchmarks, run models against them, and compare results across model versions.",
            elem_id="page-subtitle",
        )

        with gr.Group():
            gr.Markdown("**LLM Settings**")
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
                )
                g_api_key = gr.Textbox(
                    label="API Key",
                    placeholder=KEY_PLACEHOLDER["openai"],
                    type="password",
                    scale=3,
                )
            g_provider.change(fn=update_models, inputs=g_provider, outputs=g_model)
            g_provider.change(fn=update_key_field, inputs=g_provider, outputs=g_api_key)

        with gr.Accordion("Detection Thresholds", open=False):
            with gr.Row():
                entail_s = gr.Slider(
                    0.1, 0.9, value=0.5, step=0.05, label="Entailment threshold"
                )
                contradict_s = gr.Slider(
                    0.1, 0.9, value=0.5, step=0.05, label="Contradiction threshold"
                )
            with gr.Row():
                grounded_s = gr.Slider(
                    0.1, 0.6, value=0.3, step=0.05, label="Grounded score ceiling"
                )
                partial_s = gr.Slider(
                    0.3, 0.9, value=0.6, step=0.05, label="Partially grounded ceiling"
                )

        with gr.Tabs():
            with gr.Tab("Benchmarks"):
                gr.Markdown(
                    "A benchmark is a named set of test cases. Each test case is a question "
                    "paired with a reference document — the ground truth the model's answer is checked against."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("**Create benchmark**")
                        bm_name = gr.Textbox(
                            label="Name", placeholder="e.g. Medical QA v1"
                        )
                        bm_desc = gr.Textbox(
                            label="Description", placeholder="Optional"
                        )
                        bm_create_btn = gr.Button("Create", variant="primary")
                        bm_create_out = gr.Markdown()

                    with gr.Column(scale=2):
                        gr.Markdown("**Existing benchmarks**")
                        bm_list = gr.Dropdown(
                            choices=_benchmark_choices(),
                            label="Select benchmark",
                            interactive=True,
                        )
                        with gr.Row():
                            bm_refresh_btn = gr.Button("Refresh", size="sm")
                            bm_delete_btn = gr.Button(
                                "Delete selected", size="sm", variant="stop"
                            )
                        bm_action_out = gr.Markdown()

                bm_create_btn.click(
                    fn=create_benchmark_fn,
                    inputs=[bm_name, bm_desc],
                    outputs=[bm_create_out, bm_list],
                )
                bm_refresh_btn.click(fn=refresh_benchmarks_fn, outputs=bm_list)
                bm_delete_btn.click(
                    fn=delete_benchmark_fn,
                    inputs=bm_list,
                    outputs=[bm_action_out, bm_list],
                )

                gr.Markdown("**Test cases**")
                with gr.Row():
                    cases_load_btn = gr.Button("Load cases", size="sm")
                    cases_count = gr.Markdown()
                cases_display = gr.Textbox(label="Cases", lines=8, interactive=False)
                cases_load_btn.click(
                    fn=load_cases_fn,
                    inputs=bm_list,
                    outputs=[cases_count, cases_display],
                )

                with gr.Tabs():
                    with gr.Tab("Add single case"):
                        tc_question = gr.Textbox(
                            label="Question",
                            placeholder="e.g. What is the block size defined in the file system?",
                            lines=2,
                        )
                        tc_ref = gr.Textbox(
                            label="Reference document",
                            placeholder="Paste the source text the answer must come from...",
                            lines=6,
                        )
                        with gr.Row():
                            tc_domain = gr.Dropdown(
                                choices=DOMAINS,
                                value="general",
                                label="Domain",
                                scale=2,
                            )
                            tc_source_type = gr.Radio(
                                choices=["internal", "public"],
                                value="internal",
                                label="Source type",
                                info="Internal = private doc the LLM hasn't seen. Public = Wikipedia, papers, etc.",
                                scale=1,
                            )
                        tc_add_btn = gr.Button("Add test case", variant="primary")
                        tc_add_out = gr.Markdown()
                        tc_add_btn.click(
                            fn=add_case_fn,
                            inputs=[
                                bm_list,
                                tc_question,
                                tc_ref,
                                tc_domain,
                                tc_source_type,
                            ],
                            outputs=tc_add_out,
                        )

                    with gr.Tab("Bulk import from CSV"):
                        gr.Markdown(
                            "CSV must have columns: `question`, `reference_text`. Optional: `domain`, `source_type` (internal/public).\n\n"
                            "```\nquestion,reference_text,domain,source_type\n"
                            "What is the block size?,The VCB stores block size as 512 bytes...,technical,internal\n```"
                        )
                        bulk_csv = gr.Textbox(
                            label="CSV content",
                            lines=10,
                            placeholder="question,reference_text,domain\n...",
                        )
                        bulk_btn = gr.Button("Import", variant="primary")
                        bulk_out = gr.Markdown()
                        bulk_btn.click(
                            fn=bulk_import_fn,
                            inputs=[bm_list, bulk_csv],
                            outputs=bulk_out,
                        )

                    with gr.Tab("Auto-generate from document"):
                        gr.Markdown(
                            "Paste a reference document and the selected LLM will generate factual questions from it. "
                            "All generated questions will use this document as their reference."
                        )
                        gen_ref = gr.Textbox(
                            label="Reference document",
                            lines=8,
                            placeholder="Paste the source document here...",
                        )
                        with gr.Row():
                            gen_num = gr.Slider(
                                3,
                                30,
                                value=10,
                                step=1,
                                label="Number of questions",
                                scale=2,
                            )
                            gen_domain = gr.Dropdown(
                                choices=DOMAINS,
                                value="general",
                                label="Domain",
                                scale=1,
                            )
                            gen_source_type = gr.Radio(
                                choices=["internal", "public"],
                                value="internal",
                                label="Source type",
                                scale=1,
                            )
                        gen_btn = gr.Button("Generate test cases", variant="primary")
                        gen_out = gr.Markdown()
                        gen_preview = gr.Textbox(
                            label="Generated questions", lines=8, interactive=False
                        )
                        gen_btn.click(
                            fn=generate_cases_fn,
                            inputs=[
                                bm_list,
                                gen_ref,
                                gen_num,
                                gen_domain,
                                gen_source_type,
                                g_provider,
                                g_model,
                                g_api_key,
                            ],
                            outputs=[gen_out, gen_preview],
                        )

            with gr.Tab("Run Eval"):
                gr.Markdown(
                    "Select a benchmark and click Start. The system runs each test case against the selected model, "
                    "scores the response using the NLI pipeline, and saves everything to the database. "
                    "Check the Results tab when the run completes."
                )
                with gr.Row():
                    run_bm_choice = gr.Dropdown(
                        choices=_benchmark_choices(), label="Benchmark", scale=3
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
                run_refresh_out = gr.State()

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
                    outputs=[run_status_out, run_refresh_out],
                )

            with gr.Tab("Results"):
                gr.Markdown(
                    "Browse completed runs. Select a run to see the full per-question breakdown and domain scores."
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
                    label="Run summary", lines=5, interactive=False
                )
                results_domains = gr.Textbox(
                    label="Domain breakdown", lines=6, interactive=False
                )
                results_cases = gr.Textbox(
                    label="Per-question results", lines=20, interactive=False
                )
                load_results_btn.click(
                    fn=load_run_results_fn,
                    inputs=results_run_choice,
                    outputs=[results_summary, results_domains, results_cases],
                )

            with gr.Tab("Compare Models"):
                gr.Markdown(
                    "Select two runs to compare side by side. "
                    "Run both on the same benchmark for a meaningful comparison. "
                    "The report shows which questions improved, regressed, or stayed the same."
                )
                with gr.Row():
                    compare_a = gr.Dropdown(
                        choices=_run_choices(), label="Run A (baseline)", scale=2
                    )
                    compare_b = gr.Dropdown(
                        choices=_run_choices(), label="Run B (challenger)", scale=2
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
