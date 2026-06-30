from llm_client import call_llm, parse_json_response


def extract_financial_kpis(transcript: str) -> dict:
    """
    LLM Call 1: Extract structured financial KPIs mentioned
    in the earnings call transcript.

    Engineering principle: ONE prompt, ONE job.
    Don't ask the LLM to extract KPIs AND analyze sentiment
    AND find risks in a single call. Why?

    1. Smaller, focused prompts have HIGHER accuracy.
       LLMs degrade in instruction-following as task
       complexity within one prompt increases.
    2. If this call fails, only KPI extraction fails —
       not your entire pipeline. Easier to debug.
    3. You can swap/improve this prompt independently
       without touching sentiment logic.

    This is the same Single Responsibility Principle
    from MoodFlix, applied to prompts instead of functions.
    """
    system_prompt = """You are a financial analyst extracting structured data from earnings call transcripts.

Extract ONLY information explicitly stated in the transcript. Do not infer or estimate numbers that aren't mentioned.

Return a JSON object with this exact structure:
{
  "revenue_mentioned": "string or null - revenue figure if mentioned",
  "revenue_growth_yoy": "string or null - YoY growth percentage if mentioned",
  "profit_margin": "string or null",
  "guidance_given": true/false,
  "guidance_summary": "string or null - forward guidance if given",
  "key_metrics": ["list of other specific numbers/metrics mentioned"]
}

If a field is not mentioned in the transcript, use null. Never fabricate numbers."""

    user_prompt = f"Earnings call transcript:\n\n{transcript}\n\nExtract the financial KPIs as specified."

    result = call_llm(system_prompt, user_prompt, json_mode=True)

    if not result["success"]:
        return {"error": result.get("error", "LLM call failed")}

    return parse_json_response(result["content"])


def analyze_sentiment(transcript: str) -> dict:
    """
    LLM Call 2: Classify management sentiment with reasoning.

    Critical design choice: we ask for REASONING alongside
    the classification, not just a label.

    Why this matters for interpretability (your XAI background):
    A bare label like "bullish" is a black-box output —
    you can't audit WHY the model decided that. Asking the
    LLM to cite specific phrases as evidence makes the
    classification auditable, similar to how SHAP makes
    a tree-based model's prediction auditable.

    This is "chain of thought" prompting — forcing the model
    to articulate reasoning improves both accuracy AND
    gives you an explanation for free.
    """
    system_prompt = """You are a financial analyst assessing management sentiment from earnings call language.

Classify the overall sentiment and provide evidence-based reasoning.

Return JSON in this exact structure:
{
  "sentiment": "bullish" | "cautious" | "bearish",
  "confidence": 0.0 to 1.0,
  "reasoning": "2-3 sentence explanation citing specific language patterns",
  "supporting_phrases": ["exact phrases from transcript that support this classification"],
  "tone_indicators": {
    "hedging_language": true/false,
    "forward_looking_confidence": "high" | "medium" | "low"
  }
}

Base your classification ONLY on language used, not on financial figures."""

    user_prompt = f"Earnings call transcript:\n\n{transcript}\n\nAnalyze management sentiment as specified."

    result = call_llm(system_prompt, user_prompt, json_mode=True)

    if not result["success"]:
        return {"error": result.get("error", "LLM call failed")}

    return parse_json_response(result["content"])


def extract_risk_factors(transcript: str) -> dict:
    """
    LLM Call 3: Identify risk factors and forward-looking
    statements that warrant analyst attention.

    Engineering note on hallucination risk:
    This is the highest-risk prompt in the pipeline for
    hallucination, because we're asking the LLM to
    identify IMPLICIT risks, not just extract explicit text.

    Mitigation strategy used here: we explicitly instruct
    the model to quote source text for every risk it flags.
    If the LLM can't produce a supporting quote, that's a
    signal the "risk" might be fabricated — this is a
    practical, prompt-level hallucination guardrail.

    The JSON structure is designed to make it easy to
    filter out risks that don't have supporting quotes.
    """
    system_prompt = """You are a credit risk analyst reviewing earnings call transcripts for risk signals.

Identify risk factors and forward-looking concerns explicitly discussed or implied.

For EVERY risk identified, you MUST include a direct quote from the transcript as evidence. If you cannot find supporting text, do not include that risk.

Return JSON in this exact structure:
{
  "risks_identified": [
    {
      "risk_category": "operational" | "financial" | "regulatory" | "market" | "competitive",
      "description": "brief description of the risk",
      "supporting_quote": "exact quote from transcript",
      "severity": "low" | "medium" | "high"
    }
  ],
  "overall_risk_level": "low" | "medium" | "high"
}"""

    user_prompt = f"Earnings call transcript:\n\n{transcript}\n\nIdentify risk factors as specified."

    result = call_llm(system_prompt, user_prompt, json_mode=True)

    if not result["success"]:
        return {"error": result.get("error", "LLM call failed")}

    return parse_json_response(result["content"])


def run_full_analysis(transcript: str) -> dict:
    """
    Orchestrates all three LLM calls and assembles the
    complete analysis.

    Engineering note: these three calls are currently
    SEQUENTIAL.
    sequential is correct: simpler to debug,
    and total latency (~3-4 seconds) is still acceptable
    for a synchronous Flask request.
    """
    kpis = extract_financial_kpis(transcript)
    sentiment = analyze_sentiment(transcript)
    risks = extract_risk_factors(transcript)

    return {
        "kpis": kpis,
        "sentiment": sentiment,
        "risks": risks
    }