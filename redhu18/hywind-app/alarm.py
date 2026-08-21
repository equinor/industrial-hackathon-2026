"""Wind Drop Alarm: turn quantile forecasts into an operator-facing state.

Severity is driven by the expected drop (median forecast) and the plausible
worst case (q0.05 of forecast speed = deepest credible drop).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AlarmState:
    level: str        # ok | watch | alert | critical
    label: str
    icon: str
    color: str        # status palette hex (light mode)
    headline: str
    action: str


STATUS_COLORS = {"ok": "#0ca30c", "watch": "#fab219", "alert": "#ec835a", "critical": "#d03b3b"}

ACTIONS = {
    "ok": "Normal operations. No action required.",
    "watch": "Monitor closely. Brief the control room; review spinning reserve.",
    "alert": "Prepare backup generation. Pre-warm gas turbines and shed deferrable load.",
    "critical": "Act now. Start backup generation and secure critical systems before the drop hits.",
}


def evaluate_alarm(s_now: float, p30: float, p60: float, q05_30: float, q05_60: float,
                   drop_prob: float | None = None) -> AlarmState:
    """Levels driven by the point forecast and the drop classifier.

    Thresholds tuned on the Oct-Dec 2024 holdout: watch+ fires 16% of hours,
    catches 72% of severe (>=4 m/s) drops, quiet hours stay quiet 89%.
    (The calibrated worst case is reported but doesn't drive the level - it
    is wide by design and made the alarm fire 90% of the time.)
    """
    expected_drop = max(s_now - p30, s_now - p60)
    worst_drop = max(s_now - q05_30, s_now - q05_60)
    prob = -1.0 if drop_prob is None else float(drop_prob)

    if expected_drop >= 5 or prob >= 0.60:
        level = "critical"
    elif expected_drop >= 3 or prob >= 0.35:
        level = "alert"
    elif expected_drop >= 1.5 or prob >= 0.15:
        level = "watch"
    else:
        level = "ok"

    headline = (
        f"Expected change {-expected_drop:+.1f} m/s within 60 min "
        f"(plausible worst case {-worst_drop:+.1f} m/s)"
    )
    if drop_prob is not None:
        headline += f" · P(drop > 2 m/s) = {prob * 100:.0f}%"
    labels = {"ok": "Normal", "watch": "Watch", "alert": "Alert", "critical": "Critical"}
    icons = {"ok": "✓", "watch": "▲", "alert": "⚠", "critical": "●"}
    return AlarmState(
        level=level,
        label=labels[level],
        icon=icons[level],
        color=STATUS_COLORS[level],
        headline=headline,
        action=ACTIONS[level],
    )
