from groq import Groq
from config import Config
from langsmith import traceable
import json
import time

client = Groq(api_key=Config.GROQ_API_KEY)

@traceable(run_type="llm", run_name="groq_llm_call")
def call_llm(system_prompt: str, user_prompt: str, json_mode: bool = False) -> dict:
    """
    Core LLM call wrapper. Every LLM interaction in this
    entire project funnels through this one function.
    
    The @traceable decorator is doing a lot of invisible work:
    - Captures the exact input (system_prompt, user_prompt)
    - Captures the exact output (the LLM's response)
    - Records start/end time automatically
    - Sends all of this to your LangSmith project dashboard
    - Nests this trace under any PARENT @traceable function
      that called it (we'll see this when we trace
      run_full_analysis as a whole "chain")

    This requires ZERO changes to your function logic --
    that's the point of decorators. Observability is added
    as a cross-cutting concern, not woven into business logic.
    
    Why centralize this?
    - Single place to add retry logic, logging, error handling
    - If we switch providers later (Groq → OpenAI → Claude),
      we change ONE file, not every call site
    - This is the same pattern as tmdb_client.py in MoodFlix —
      isolate the external dependency behind one interface

    Parameters:
        system_prompt: Sets the LLM's role/behavior (not visible to user)
        user_prompt: The actual task/question
        json_mode: If True, forces LLM to return valid JSON only

    Returns:
        dict with: content, latency_ms, tokens_used, success
    """
    start_time = time.time()

    try:
        kwargs = {
            "model": Config.MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": Config.TEMPERATURE,
            "max_tokens": Config.MAX_TOKENS,
        }

        # json_mode forces the model to ONLY output valid JSON
        # This is critical for production pipelines — without this,
        # LLMs sometimes add "Here's the analysis:" before the JSON,
        # breaking your json.loads() parsing downstream
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(**kwargs)

        latency_ms = round((time.time() - start_time) * 1000, 2)
        raw_content = response.choices[0].message.content

        result = {
            "content": raw_content,
            "latency_ms": latency_ms,
            "tokens_used": response.usage.total_tokens,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "success": True
        }

        # Manual tracing log — prints to console for now
        # We'll wire this into LangSmith properly in the next step
        print(f"[LLM CALL] tokens={result['tokens_used']} "
              f"latency={latency_ms}ms model={Config.MODEL_NAME}")

        return result

    except Exception as e:
        print(f"[LLM ERROR] {str(e)}")
        return {
            "content": None,
            "latency_ms": round((time.time() - start_time) * 1000, 2),
            "tokens_used": 0,
            "success": False,
            "error": str(e)
        }


def parse_json_response(raw_content: str) -> dict:
    """
    Safely parses LLM's JSON output.

    Why this needs to be defensive:
    Even with json_mode=True, LLMs occasionally produce
    malformed JSON — trailing commas, unescaped quotes,
    or truncated output if max_tokens is hit mid-generation.

    This is THE most common production bug in LLM apps.
    Never trust raw LLM output to be perfectly parseable.
    """
    if not raw_content:
        return {}

    try:
        return json.loads(raw_content)
    except json.JSONDecodeError as e:
        print(f"[JSON PARSE ERROR] {e}")
        print(f"[RAW CONTENT] {raw_content[:500]}")
        return {}