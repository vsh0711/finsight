import pulp


# Severity → numeric weight mapping
# Engineering note: these weights are NOT arbitrary.
# In a real deployment, these would be calibrated against
# historical default data (a credit risk model would inform
# these weights via logistic regression coefficients or
# similar). For this portfolio project, we use defensible
# analyst judgment weights, explicitly documented.
SEVERITY_WEIGHTS = {
    "low": 1,
    "medium": 3,
    "high": 6
}

RISK_CATEGORY_WEIGHTS = {
    "financial": 1.5,      # Direct balance sheet impact - highest weight
    "regulatory": 1.3,     # Can change business model overnight
    "market": 1.1,         # Cyclical, somewhat outside management control
    "operational": 1.0,    # Internal, typically more controllable
    "competitive": 0.8     # Slow-moving, least urgent
}

SENTIMENT_SCORE_MAP = {
    "bullish": 0,
    "cautious": 5,
    "bearish": 10
}


def compute_weighted_risk_score(analysis: dict) -> dict:
    """
    Computes a single interpretable Risk Index (0-100) from
    the LLM's structured outputs, using explicit weighted
    linear scoring -- NOT another LLM call.

    Why not just ask the LLM to score it 0-100?
    1. Non-determinism: same input, different score across runs
    2. No audit trail: a regulator or credit committee cannot
       trace WHY a score is 67 vs 72
    3. No control: you can't easily re-weight "regulatory risk
       matters more this quarter" without re-prompting and
       hoping the LLM listens

    With explicit weighted scoring, every point in the final
    score traces back to a specific risk, its severity, and
    its category weight. This is fully auditable and explainable, which is critical for credit risk.
    """
    risks = analysis.get("risks", {}).get("risks_identified", [])
    sentiment_data = analysis.get("sentiment", {})

    # --- Component 1: Risk Factor Score ---
    risk_score = 0
    risk_breakdown = []

    for risk in risks:
        severity = risk.get("severity", "low")
        category = risk.get("risk_category", "operational")

        severity_weight = SEVERITY_WEIGHTS.get(severity, 1)
        category_weight = RISK_CATEGORY_WEIGHTS.get(category, 1.0)

        contribution = severity_weight * category_weight
        risk_score += contribution

        risk_breakdown.append({
            "description": risk.get("description", ""),
            "category": category,
            "severity": severity,
            "contribution": round(contribution, 2)
        })

    # Normalize risk_score to 0-70 range
    # (we cap contribution since unbounded risk counts would
    # let a transcript with many minor risks outscore one
    # with fewer but severe risks -- diminishing returns curve)
    max_possible_raw = 6 * 1.5 * 8  # heuristic ceiling: 8 high-financial risks
    normalized_risk_score = min(70, (risk_score / max_possible_raw) * 70)

    # --- Component 2: Sentiment Score ---
    sentiment_label = sentiment_data.get("sentiment", "cautious")
    confidence = sentiment_data.get("confidence", 0.5)

    base_sentiment_score = SENTIMENT_SCORE_MAP.get(sentiment_label, 5)
    # Weight sentiment contribution by model's own confidence --
    # a low-confidence "bearish" call shouldn't swing the
    # index as hard as a high-confidence one
    sentiment_score = base_sentiment_score * confidence * 3  # scaled to 0-30 range

    # --- Final Composite Index ---
    final_score = round(normalized_risk_score + sentiment_score, 1)
    final_score = min(100, max(0, final_score))  # clamp to [0, 100]

    return {
        "risk_index": final_score,
        "risk_band": classify_risk_band(final_score),
        "components": {
            "risk_factor_score": round(normalized_risk_score, 1),
            "sentiment_score": round(sentiment_score, 1)
        },
        "risk_breakdown": risk_breakdown,
        "methodology_note": (
            "Risk Index = normalized weighted risk factors (max 70) "
            "+ confidence-weighted sentiment score (max 30). "
            "Weights are analyst-defined; see SEVERITY_WEIGHTS and "
            "RISK_CATEGORY_WEIGHTS for full transparency."
        )
    }


def classify_risk_band(score: float) -> str:
    """
    Converts continuous score to a discrete band --
    mirrors how credit rating agencies present scores
    (e.g., AAA/AA/A bands rather than raw numeric scores)
    for easier analyst consumption.
    """
    if score < 30:
        return "Low Risk"
    elif score < 60:
        return "Moderate Risk"
    else:
        return "High Risk"


def optimize_coverage_allocation(risks: list, total_provisioning_budget: float) -> dict:
    """
    A genuine Operations Research add-on: given a limited
    provisioning budget, allocate it across identified risk
    categories to MINIMIZE expected residual risk exposure.

    This frames provisioning as a constrained optimization
    problem -- exactly the OR-to-ML bridge from your thesis
    work: ML/LLM identifies risks, OR allocates resources
    optimally against them.

    Formulation:
        Decision variables: x_i = budget allocated to risk i
        Objective: minimize sum(severity_weight_i * (1 - x_i/cost_i))
        Constraints: sum(x_i) <= total_budget, x_i >= 0

    This is a simplified knapsack-style allocation -- in a
    full production system you'd model risk reduction as a
    nonlinear function of spend, but linear approximation is
    a defensible starting point and keeps the LP solvable
    with PuLP's open-source CBC solver.
    """
    if not risks:
        return {"allocations": [], "total_allocated": 0, "note": "No risks to allocate against"}

    prob = pulp.LpProblem("Risk_Coverage_Allocation", pulp.LpMinimize)

    # Assume each risk has an associated "mitigation cost"
    # proportional to severity (higher severity = costlier to mitigate)
    severity_cost_map = {"low": 5, "medium": 15, "high": 30}

    risk_vars = {}
    for i, risk in enumerate(risks):
        risk_vars[i] = pulp.LpVariable(f"alloc_{i}", lowBound=0)

    # Objective: minimize unmitigated risk exposure
    # (weighted by severity -- higher severity left unmitigated
    # is penalized more heavily in the objective)
    objective_terms = []
    for i, risk in enumerate(risks):
        severity = risk.get("severity", "low")
        severity_weight = SEVERITY_WEIGHTS.get(severity, 1)
        cost = severity_cost_map.get(severity, 5)

        # Residual risk = severity_weight * (1 - allocated/cost)
        # Linearized: minimize severity_weight - (severity_weight/cost)*allocated
        objective_terms.append(
            severity_weight - (severity_weight / cost) * risk_vars[i]
        )

    prob += pulp.lpSum(objective_terms)

    # Constraint: total allocation cannot exceed budget
    prob += pulp.lpSum(risk_vars.values()) <= total_provisioning_budget

    # Constraint: cannot over-allocate beyond what a risk needs
    for i, risk in enumerate(risks):
        severity = risk.get("severity", "low")
        cost = severity_cost_map.get(severity, 5)
        prob += risk_vars[i] <= cost

    prob.solve(pulp.PULP_CBC_CMD(msg=0))  # msg=0 suppresses solver logs

    allocations = []
    total_allocated = 0
    for i, risk in enumerate(risks):
        allocated = risk_vars[i].varValue or 0
        allocations.append({
            "risk_description": risk.get("description", ""),
            "severity": risk.get("severity", ""),
            "allocated_budget_cr": round(allocated, 2)
        })
        total_allocated += allocated

    return {
        "allocations": allocations,
        "total_allocated": round(total_allocated, 2),
        "total_budget": total_provisioning_budget,
        "solver_status": pulp.LpStatus[prob.status]
    }