"""
fusion/bayesian.py — MFAD Bayesian Log-Odds Fusion
====================================================
Owner: Dev

Combines all per-module anomaly_scores into a single DeepFake Prediction
Score using log-odds Bayesian fusion.

Math
----
Each module score is treated as P(fake | evidence from that module).
Prior: P(fake) = 0.5 (uninformative — no image-level prior applied).

For each module i:
    log_odds_i = weight_i × log( score_i / (1 - score_i) )

Total:
    log_odds_total = Σ log_odds_i

Final score:
    final_score = sigmoid(log_odds_total) = 1 / (1 + exp(-log_odds_total))

Confidence interval:
    Bootstrap resampling over module scores (1000 samples).
    95% CI = [2.5th percentile, 97.5th percentile] of bootstrap distribution.

Weights
-------
Sourced from published AUC values in contracts.py / §6 of reference report.
geometry:     0.15  (§5.1)
gan_artefact: 0.25  (§5.2 — EfficientNet-B4)
frequency:    0.25  (§5.2 — FFT)
texture:      0.20  (§5.3)
vlm:          0.25  (§5.4)
biological:   0.15  (§5.5)
metadata:     0.15  (§5.6)

NOTE: Weights intentionally do NOT sum to 1.0.
      They are reliability multipliers on the log-odds axis,
      not a probability simplex. This is standard in log-odds ensembles.

Decision thresholds (from contracts.py DEEPFAKE_THRESHOLD / AUTHENTIC_THRESHOLD):
    final_score >= 0.70  → DEEPFAKE
    final_score <= 0.35  → AUTHENTIC
    otherwise            → UNCERTAIN

Interpretation bands:
    >= 0.90  → Very High Confidence
    >= 0.75  → High Confidence
    >= 0.55  → Moderate Confidence
    else     → Low Confidence

Output schema: matches contracts.py FUSION_KEYS exactly.
"""

from __future__ import annotations

import math
import sys
import logging
from pathlib import Path
from typing import Optional

import numpy as np

# ── path setup so contracts import works from any cwd ─────────────────────────
_PROJECT_DIR = str(Path(__file__).resolve().parent.parent)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from contracts import (
    FUSION_KEYS,
    FUSION_WEIGHTS,
    DEEPFAKE_THRESHOLD,
    AUTHENTIC_THRESHOLD,
    validate,
)

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Number of bootstrap resamples for 95% CI
_BOOTSTRAP_N = 1000

# Score clamping — avoid log(0) and log(inf)
_EPS = 1e-6
_SCORE_MIN = _EPS
_SCORE_MAX = 1.0 - _EPS

# Weights from contracts.py FUSION_WEIGHTS (§6 of reference report)
MODULE_WEIGHTS: dict[str, float] = FUSION_WEIGHTS

# Model performance constants — fixed, image-independent (§6)
MODEL_AUC_ROC        = 0.983
FALSE_POSITIVE_RATE  = 0.021
CALIBRATION_ECE      = 0.014
DECISION_THRESHOLD   = DEEPFAKE_THRESHOLD


# ── Core math ────────────────────────────────────────────────────────────────

def _clamp(score: float) -> float:
    """Clamp score to (_EPS, 1-_EPS) to keep log finite."""
    return max(_SCORE_MIN, min(_SCORE_MAX, float(score)))


def _log_odds(score: float) -> float:
    """Convert probability to log-odds: log(p / (1-p))."""
    p = _clamp(score)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    """Sigmoid: 1 / (1 + exp(-x)). Numerically stable."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def _fuse_scores(scores: dict[str, float]) -> float:
    """
    Core log-odds Bayesian fusion.

    Args:
        scores: {module_name: anomaly_score} — only modules present in
                MODULE_WEIGHTS are used; extra keys are silently ignored.

    Returns:
        final_score in [0, 1].
    """
    log_odds_total = 0.0
    modules_used = 0

    for module, weight in MODULE_WEIGHTS.items():
        score = scores.get(module)
        if score is None:
            log.debug("fusion: module '%s' absent — skipped", module)
            continue
        log_odds_total += weight * _log_odds(score)
        modules_used += 1

    if modules_used == 0:
        log.warning("fusion: no valid module scores — returning 0.5")
        return 0.5

    return _sigmoid(log_odds_total)


def _bootstrap_ci(
    scores: dict[str, float],
    n: int = _BOOTSTRAP_N,
    ci: float = 0.95,
) -> list[float]:
    """
    Compute confidence interval via bootstrap resampling over modules.

    For each bootstrap iteration:
        1. Resample (with replacement) from the available module scores.
        2. Compute the fused score from the resample.
    Returns [lower, upper] at the requested CI level.

    Args:
        scores: {module_name: anomaly_score}
        n:      number of bootstrap resamples
        ci:     confidence level (default 0.95 → 95% CI)

    Returns:
        [lower_bound, upper_bound] rounded to 4 decimal places.
    """
    rng = np.random.default_rng(seed=42)   # deterministic for reproducibility
    available = [
        (module, weight, scores[module])
        for module, weight in MODULE_WEIGHTS.items()
        if module in scores
    ]

    if len(available) < 2:
        # Can't bootstrap with < 2 modules — return degenerate CI
        base = _fuse_scores(scores)
        return [round(base, 4), round(base, 4)]

    n_modules = len(available)
    bootstrap_scores = []

    for _ in range(n):
        # Resample indices with replacement
        indices = rng.integers(0, n_modules, size=n_modules)
        resampled = {}
        seen: dict[str, list[float]] = {}

        for idx in indices:
            module, weight, score = available[idx]
            seen.setdefault(module, []).append(score)

        # For modules that appear multiple times, take the mean score
        for module, score_list in seen.items():
            resampled[module] = float(np.mean(score_list))

        bootstrap_scores.append(_fuse_scores(resampled))

    alpha = 1.0 - ci
    lo = float(np.percentile(bootstrap_scores, 100 * alpha / 2))
    hi = float(np.percentile(bootstrap_scores, 100 * (1.0 - alpha / 2)))
    return [round(lo, 4), round(hi, 4)]


def _interpret(final_score: float) -> str:
    """Map final_score to a human-readable confidence band."""
    if final_score >= 0.90:
        return "Very High Confidence"
    if final_score >= 0.75:
        return "High Confidence"
    if final_score >= 0.55:
        return "Moderate Confidence"
    return "Low Confidence"


def _decide(final_score: float) -> str:
    """Apply decision thresholds from contracts.py."""
    if final_score >= DEEPFAKE_THRESHOLD:
        return "DEEPFAKE"
    if final_score <= AUTHENTIC_THRESHOLD:
        return "AUTHENTIC"
    return "UNCERTAIN"


# ── Public API ────────────────────────────────────────────────────────────────

def bayesian_fusion(
    per_module_scores: dict[str, float],
    compute_ci: bool = True,
) -> dict:
    """
    Main entry point. Called by fusion_node in master_agent.py.

    Accepts a flat dict of {module_name: anomaly_score} from all agents
    and returns a FUSION_KEYS-compliant output dict.

    Expected module keys (any subset is acceptable — missing modules
    are skipped with a debug log):
        geometry, gan_artefact, frequency, texture,
        vlm, biological, metadata

    Args:
        per_module_scores: dict of {module_name: anomaly_score (float 0-1)}
        compute_ci:        whether to run bootstrap CI (set False for unit tests)

    Returns:
        dict matching contracts.py FUSION_KEYS

    Raises:
        ValueError if contracts.validate() fails (missing output keys).
    """
    log.info("bayesian_fusion | modules received: %s", list(per_module_scores.keys()))

    # ── 1. Filter and validate scores ──────────────────────────────────────
    # Skip None, NaN, and out-of-range values
    clamped = {}
    skipped = []
    
    for module, score in per_module_scores.items():
        # Skip None values
        if score is None:
            skipped.append(f"{module}=None")
            continue
        
        # Skip NaN values
        if isinstance(score, float) and math.isnan(score):
            skipped.append(f"{module}=NaN")
            continue
        
        # Skip invalid range (should be 0-1)
        if not isinstance(score, (int, float)) or score < 0 or score > 1:
            skipped.append(f"{module}=invalid({score})")
            continue
        
        # Valid score — clamp and add
        clamped[module] = _clamp(score)
    
    if skipped:
        log.warning("bayesian_fusion | skipped modules: %s", ", ".join(skipped))

    # ── 2. Fuse ────────────────────────────────────────────────────────────
    final_score = round(_fuse_scores(clamped), 6)

    # ── 3. Confidence interval ─────────────────────────────────────────────
    if compute_ci:
        confidence_interval = _bootstrap_ci(clamped)
    else:
        # Heuristic CI when bootstrapping is skipped (unit tests / speed)
        spread = 0.04 * (1.0 - abs(2 * final_score - 1.0))
        confidence_interval = [
            round(max(0.0, final_score - spread), 4),
            round(min(1.0, final_score + spread), 4),
        ]

    # ── 4. Decision + interpretation ──────────────────────────────────────
    decision       = _decide(final_score)
    interpretation = _interpret(final_score)

    log.info(
        "bayesian_fusion | final=%.4f  decision=%s  CI=%s",
        final_score, decision, confidence_interval,
    )

    # ── 5. Build output dict (FUSION_KEYS) ─────────────────────────────────
    output = {
        "final_score":         final_score,
        "confidence_interval": confidence_interval,
        "per_module_scores":   dict(per_module_scores),  # pass through original (unclamped)
        "decision":            decision,
        "interpretation":      interpretation,
        "model_auc_roc":       MODEL_AUC_ROC,
        "false_positive_rate": FALSE_POSITIVE_RATE,
        "calibration_ece":     CALIBRATION_ECE,
        "decision_threshold":  DECISION_THRESHOLD,
    }

    validate(output, FUSION_KEYS, "bayesian_fusion")
    return output


# ── CLI for quick testing ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    # Example scores from reference case DFA-2025-TC-00471
    test_scores = {
        "geometry":     0.884,
        "gan_artefact": 0.967,
        "frequency":    0.912,
        "texture":      0.895,
        "vlm":          0.931,
        "biological":   0.826,
        "metadata":     0.973,
    }

    result = bayesian_fusion(test_scores)
    print(json.dumps(result, indent=2))
    print(f"\nVerdict: {result['decision']}  ({result['final_score']*100:.1f}%)")
    print(f"95% CI : {result['confidence_interval']}")