# 📊 FinSight — LLM-Powered Earnings Call Analyser with Interpretable Risk Scoring

An end-to-end LLM application that extracts structured financial intelligence from unstructured earnings call transcripts, and converts qualitative LLM outputs into a fully auditable, deterministic risk score using Operations Research-based optimization.

**Live Demo → [finsight-4b3w.onrender.com](https://finsight-4b3w.onrender.com/)**

---

## What It Does

Paste any earnings call transcript. FinSight:

1. **Extracts** structured financial KPIs (revenue, growth, guidance) via LLM
2. **Classifies** management sentiment with cited, evidence-based reasoning
3. **Identifies** risk factors with mandatory source-quote grounding (hallucination guardrail)
4. **Computes** a 0-100 Risk Index using deterministic, weighted linear scoring — not another LLM call
5. **Solves** an optimal provisioning budget allocation across identified risks using Linear Programming

Every number in the final output traces back to either a direct quote from the transcript, or an explicit, documented weight in the scoring model.

---

## Why This Is Not "Just an LLM Wrapper"

Most LLM demo projects ask the model to do everything — including final scoring. This is a known production anti-pattern: **LLM-generated numeric scores are non-deterministic and unauditable**. Ask an LLM to "rate risk 1-10" twice, you get two different answers with no way to explain the gap.

FinSight uses a **neuro-symbolic architecture**:
```
LLM Layer (probabilistic)          OR Layer (deterministic)
────────────────────────           ─────────────────────────
Extracts risks from text     →     Computes weighted risk index
Classifies sentiment          →     using fixed, documented weights
Grounds every claim in a
direct quote                  →     Solves optimal budget allocation
via Linear Programming (PuLP)

```
The LLM does what it's good at: language understanding and extraction. The OR layer does what it's good at: consistent, explainable, mathematically optimal scoring and allocation. This split is the project's core engineering decision.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Web Framework | Flask 3.0 |
| WSGI Server | Gunicorn |
| LLM Provider | Groq (Llama 3.3 70B) |
| Templating | Jinja2 |
| Optimization | PuLP (Linear Programming, CBC solver) |
| Observability | LangSmith (LLMOps tracing) |
| Deployment | Render |

---

## Architecture

```
User pastes transcript
↓
Flask receives POST request
↓
┌─────────────────────────────────────┐
│  LLM Analysis Chain (3 calls)        │
│  ├─ extract_financial_kpis()         │
│  ├─ analyze_sentiment()              │
│  └─ extract_risk_factors()           │
│  (each traced via LangSmith)         │
└─────────────────────────────────────┘
↓
┌─────────────────────────────────────┐
│  OR Scoring Layer (deterministic)    │
│  ├─ compute_weighted_risk_score()    │
│  └─ optimize_coverage_allocation()   │
│      (PuLP Linear Program)           │
└─────────────────────────────────────┘
↓
Jinja2 renders dashboard
↓
User sees: KPIs, sentiment with evidence,
cited risks, risk index, optimal allocation
```

---

## LLM Engineering Decisions

**Prompt decomposition** — One LLM call per concern (KPIs / sentiment / risks), not one mega-prompt. Smaller, focused prompts have measurably higher instruction-following accuracy, and a failure in one call doesn't break the whole pipeline.

**Citation grounding** — Every risk factor and sentiment classification must include a direct quote from the source transcript. If the model can't produce supporting text, that signals possible hallucination. This is a practical, prompt-level guardrail used before reaching for retrieval-based grounding (see Project 2: CreditLens, a RAG-based version of this problem).

**Structured JSON output** — All LLM calls use `response_format: json_object` to guarantee parseable output, with defensive `try/except` parsing since even constrained generation can occasionally fail mid-output.

**Low temperature (0.1)** — Financial analysis prioritizes consistency over creativity. Low temperature reduces sampling randomness.

---

## OR Engineering Decisions

**Weighted linear scoring over LLM self-scoring** — Risk severity and category each carry explicit, documented weights (`SEVERITY_WEIGHTS`, `RISK_CATEGORY_WEIGHTS` in `scorer.py`). The final Risk Index is fully traceable: any stakeholder can see exactly which risk contributed how many points.

**Confidence-weighted sentiment scoring** — The LLM's own stated confidence scales its contribution to the final score. A low-confidence "bearish" classification swings the index less than a high-confidence one.

**Constrained optimization for provisioning** — `optimize_coverage_allocation()` formulates budget allocation as a Linear Program: minimize unmitigated risk exposure subject to a total budget constraint, solved with PuLP's open-source CBC solver. This produces a mathematically optimal, defensible allocation rather than a heuristic guess.

---

## LLMOps / Observability

Every LLM call and the full analysis chain are instrumented with **LangSmith** tracing via the `@traceable` decorator — zero changes to business logic required. This provides:

- Full prompt/response visibility per call
- Token usage and latency tracking
- Nested chain visualization (parent chain → 3 child LLM calls)
- A debugging dashboard equivalent to what production ML teams use to diagnose pipeline failures

---
---

## Local Setup

```bash
git clone https://github.com/vsh0711/finsight.git
cd finsight
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# .env
echo "GROQ_API_KEY=your_key" >> .env
echo "LANGCHAIN_API_KEY=your_key" >> .env
echo "LANGCHAIN_TRACING_V2=true" >> .env
echo "LANGCHAIN_PROJECT=finsight" >> .env

python app.py
# → http://127.0.0.1:5001
```

Free API keys: [Groq Console](https://console.groq.com) · [LangSmith](https://smith.langchain.com)

---

## Portfolio Context

Built as part of  LLM/RAG/Agent portfolio targeting Data Science / ML Engineering / AI Consulting roles in BFSI and Fintech, demonstrating:

- Production LLM application design (prompt decomposition, structured outputs, hallucination mitigation)
- Neuro-symbolic architecture combining LLM extraction with OR-based deterministic scoring
- LLMOps observability practices (tracing, token/cost monitoring)
- End-to-end deployment (Flask + Gunicorn + Render)

---

## Author

**Vishalini Satheesh**
M.E. CSE (Operational Research) — College of Engineering Guindy, Anna University
[GitHub](https://github.com/vsh0711) · [LinkedIn](https://linkedin.com/in/your-linkedin-here)
