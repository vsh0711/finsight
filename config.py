import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
    # Llama 3 70B — best free model on Groq for reasoning tasks
    # Engineering note: 70B param model, but Groq's custom chips
    # (LPUs) serve it at ~300 tokens/sec — 10x faster than typical
    # GPU inference. This is WHY Groq is viable for production demos.
    MODEL_NAME = "llama-3.3-70b-versatile"
    
    # Temperature controls randomness in output
    # 0.0 = deterministic, always picks highest probability token
    # 1.0 = creative, more random sampling
    # For financial analysis we want LOW temperature —
    # consistency matters more than creativity here
    TEMPERATURE = 0.1
    
    # Max tokens in the LLM's response
    # Engineering note: this is OUTPUT limit, not input.
    # Input (prompt) has a separate, much larger context window
    MAX_TOKENS = 2048
    
    # Llama 3.3 70B context window is 128k tokens
    # Roughly 96,000 words — most earnings calls fit easily
    # without chunking. We'll still build chunking logic
    # because production transcripts (10-Ks, etc) can exceed this
    CONTEXT_WINDOW = 128000
