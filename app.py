from flask import Flask, render_template, request
from config import Config
from analyser import run_full_analysis
from scorer import compute_weighted_risk_score, optimize_coverage_allocation

app = Flask(__name__)
app.config.from_object(Config)


@app.route("/", methods=["GET"])
def index():
    """
    Landing page with transcript input form.
    """
    return render_template("index.html")


@app.route("/analyse", methods=["POST"])
def analyse():
    """
    Core pipeline route.

    Request flow:
    1. Read transcript text from form
    2. Run 3 LLM calls (KPIs, sentiment, risks)
    3. Compute deterministic OR-based risk index
    4. Solve provisioning allocation LP
    5. Render dashboard with everything

    Engineering note: this entire route is synchronous.
    Total latency = sum of all LLM calls + LP solve time,
    roughly 2.5-4 seconds. Acceptable for a single-user
    demo. In Project 5 we'll parallelize the 3 LLM calls
    using async, since they're independent of each other.
    """
    transcript = request.form.get("transcript", "").strip()

    if not transcript:
        return render_template(
            "index.html",
            error="Please paste an earnings call transcript to analyse."
        )

    if len(transcript) < 200:
        return render_template(
            "index.html",
            error="Transcript seems too short for meaningful analysis. Please paste the full transcript."
        )

    # Step 1: Run LLM analysis pipeline
    analysis = run_full_analysis(transcript)

    # Guard: check if any LLM call failed
    if "error" in analysis.get("kpis", {}) or \
       "error" in analysis.get("sentiment", {}) or \
       "error" in analysis.get("risks", {}):
        return render_template(
            "index.html",
            error="Analysis failed — the LLM service may be temporarily unavailable. Please try again."
        )

    # Step 2: Deterministic OR-based scoring
    risk_score = compute_weighted_risk_score(analysis)

    # Step 3: OR provisioning allocation
    # Using a fixed demo budget of 50 crore — in production
    # this would come from the institution's actual
    # provisioning budget for the quarter
    risks_list = analysis.get("risks", {}).get("risks_identified", [])
    allocation = optimize_coverage_allocation(risks_list, total_provisioning_budget=50)

    return render_template(
        "dashboard.html",
        analysis=analysis,
        risk_score=risk_score,
        allocation=allocation,
        transcript_preview=transcript[:300]
    )


@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}, 200


if __name__ == "__main__":
    app.run(debug=True, port=5001)  # different port from MoodFlix