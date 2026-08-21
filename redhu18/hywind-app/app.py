"""Hywind Tampen Wind Intelligence — hackathon dashboard (v2 UI).

Light/dark theme, click-driven explorer with sticky map, animated model
comparison, prev/next event replay in the operator view.

    .venv/bin/python app/app.py     # http://127.0.0.1:8506
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from alarm import evaluate_alarm  # noqa: E402

DERIVED = ROOT / "data"
OUTPUTS = ROOT / "data"

# ---------------------------------------------------------------- theme tokens
TOKENS = {
    "light": dict(
        card="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
        grid="#e1e0d9", baseline="#c3c2b7", blue="#2a78d6",
        wash="rgba(42,120,214,0.10)",
        assets={
            "HYT-HY09": "#2a78d6", "Statfjord_A": "#eb6834", "Statfjord_B": "#1baf7a",
            "Gullfaks_C": "#eda100", "Snorre_A": "#e87ba4", "Snorre_B": "#008300",
            "Visund": "#4a3aa7", "Other turbines": "#898781",
        },
    ),
    "dark": dict(
        card="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
        grid="#2c2c2a", baseline="#383835", blue="#3987e5",
        wash="rgba(57,135,229,0.14)",
        assets={
            "HYT-HY09": "#3987e5", "Statfjord_A": "#d95926", "Statfjord_B": "#199e70",
            "Gullfaks_C": "#c98500", "Snorre_A": "#d55181", "Snorre_B": "#008300",
            "Visund": "#9085e9", "Other turbines": "#898781",
        },
    ),
}
MODEL_LABELS = {
    "persistence": "Persistence (benchmark)", "ridge": "Ridge",
    "lgbm": "LightGBM", "xgb": "XGBoost", "ensemble": "Ensemble (LGB+XGB)",
}
MODEL_ORDER = ["persistence", "ridge", "lgbm", "xgb", "ensemble"]

NETWORK_POS = {
    "Snorre_A": (-8, 26), "Snorre_B": (-4, 34), "Visund": (18, 6),
    "Statfjord_A": (-30, -10), "Statfjord_B": (-34, -15), "Gullfaks_C": (1, -24),
}

# ---------------------------------------------------------------- data
speeds10 = pd.read_parquet(DERIVED / "ui_speeds_10min.parquet")
speeds1 = pd.read_parquet(DERIVED / "ui_speeds_1min_holdout.parquet")
drops = pd.read_parquet(DERIVED / "ui_drop_events.parquet")
metrics = json.loads((OUTPUTS / "holdout_metrics.json").read_text())
preds = pd.read_parquet(OUTPUTS / "holdout_predictions.parquet")

CALIB_PATH = OUTPUTS / "band_calibration.json"
BAND_CALIB = json.loads(CALIB_PATH.read_text()) if CALIB_PATH.exists() else None

TURBINES = [c for c in speeds10.columns if c.startswith("HYT-") and c != "HYT-HY09"]
speeds10["Other turbines"] = speeds10[TURBINES].mean(axis=1)

corr_2024 = speeds10.loc["2024"].corr()["HYT-HY09"]

p60 = preds[preds["horizon"] == 60].set_index("origin")
p30 = preds[preds["horizon"] == 30].set_index("origin")
common = p60.index.intersection(p30.index)
p60, p30 = p60.loc[common], p30.loc[common]
change = (p60["actual"] - p60["s_now"]).sort_values()
EPISODES = list(change.index[:28]) + list(change.abs().sort_values().index[:4])


def _alarm_track_record() -> dict:
    """Alarm operating stats on the full unseen holdout (vectorized rule)."""
    sn = p60["s_now"]
    exp_drop = np.maximum(sn - p30["ensemble"], sn - p60["ensemble"])
    prob = p60["drop_prob"].fillna(0) if "drop_prob" in p60.columns else pd.Series(0, index=p60.index)
    real = (p60["actual"] - sn) <= -2.0
    severe = (p60["actual"] - sn) <= -4.0
    watch = (exp_drop >= 1.5) | (prob >= 0.15)
    return {
        "n": int(len(p60)),
        "severe_recall": 100 * (watch & severe).sum() / max(int(severe.sum()), 1),
        "precision": 100 * (watch & real).sum() / max(int(watch.sum()), 1),
        "lift": ((watch & real).sum() / max(int(watch.sum()), 1)) / max(real.mean(), 1e-9),
        "quiet": 100 * (~watch & ~real).sum() / max(int((~real).sum()), 1),
    }


TRACK = _alarm_track_record()


def calibrated_band(h, point, q05, q95):
    if not BAND_CALIB:
        return q05, q95
    cfg = BAND_CALIB.get(f"h{h}")
    lo_raw, hi_raw = min(q05, point), max(q95, point)
    if cfg["lo"].get("method") == "add":
        lo = max(lo_raw - cfg["lo"]["param"], 0.0)
    else:
        lo = max(point - cfg["lo"].get("param", cfg["lo"]["k"]) * (point - lo_raw), 0.0)
    if cfg["hi"].get("method") == "add":
        hi = hi_raw + cfg["hi"]["param"]
    else:
        hi = point + cfg["hi"].get("param", cfg["hi"]["k"]) * (hi_raw - point)
    return lo, hi


def base_fig(theme: str, height: int = 340) -> go.Figure:
    t = TOKENS[theme]
    fig = go.Figure()
    fig.update_layout(
        template=None, height=height, paper_bgcolor=t["card"], plot_bgcolor=t["card"],
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', color=t["ink2"], size=12),
        margin=dict(l=52, r=18, t=8, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=12)),
        hoverlabel=dict(bgcolor=t["card"], bordercolor=t["grid"], font=dict(color=t["ink"], size=12)),
    )
    fig.update_xaxes(gridcolor=t["grid"], linecolor=t["baseline"], zeroline=False, tickcolor=t["baseline"])
    fig.update_yaxes(gridcolor=t["grid"], linecolor=t["baseline"], zeroline=False, tickcolor=t["baseline"])
    return fig


def tile(label, value, delta="", good=False):
    kids = [html.Div(label, className="label"), html.Div(value, className="value")]
    if delta:
        kids.append(html.Div(delta, className="delta good" if good else "delta"))
    return html.Div(kids, className="tile")


# ---------------------------------------------------------------- data explorer
ASSET_META = {
    "HYT-HY09": ("Target turbine", 0.0),
    **{name: (f"Platform · {((x * x + y * y) ** 0.5):.0f} km out", (x * x + y * y) ** 0.5)
       for name, (x, y) in NETWORK_POS.items()},
    "Other turbines": ("Mean of the 10 neighbours", 0.5),
}


def sensor_buttons(selected, theme):
    t = TOKENS[theme]
    btns = []
    for a, (desc, _) in ASSET_META.items():
        btns.append(
            html.Button(
                [
                    html.Span(className="sensor-dot", style={"background": t["assets"][a]}),
                    html.Span(a.replace("_", " ")),
                    html.Span(desc, className="meta"),
                ],
                id={"type": "sensor-btn", "asset": a},
                className="sensor-btn selected" if a == selected else "sensor-btn",
            )
        )
    return btns


def explorer_map(selected, theme):
    t = TOKENS[theme]
    fig = base_fig(theme, 330)
    fig.update_layout(margin=dict(l=10, r=10, t=6, b=10))
    fig.add_scatter(
        x=[0, 1.5, 2, 0.5, -0.5, 0], y=[6, 5, -5, -7, -4, 6], mode="lines",
        line=dict(width=2, color=t["assets"]["HYT-HY09"]), fill="toself",
        fillcolor=t["wash"], name="Hywind Tampen", hoverinfo="name", showlegend=False,
    )
    hy_sel = selected in ("HYT-HY09", "Other turbines")
    fig.add_scatter(
        x=[-0.4], y=[-2], mode="markers+text",
        marker=dict(size=16 if hy_sel else 11, color=t["assets"]["HYT-HY09"],
                    line=dict(width=3 if hy_sel else 2, color=t["card"])),
        text=["HY09"], textposition="middle right",
        textfont=dict(size=12.5, color=t["ink"]), showlegend=False,
        hovertemplate="HYT-HY09 — forecast target<extra></extra>",
    )
    for name, (x, y) in NETWORK_POS.items():
        sel = name == selected
        fig.add_scatter(
            x=[x], y=[y], mode="markers+text",
            marker=dict(size=17 if sel else 11, color=t["assets"][name],
                        line=dict(width=3 if sel else 2, color=t["card"])),
            text=[name.replace("_", " ")], textposition="bottom center",
            textfont=dict(size=12 if sel else 11, color=t["ink"] if sel else t["ink2"]),
            showlegend=False,
            hovertemplate=f"{name.replace('_', ' ')} · ~{(x * x + y * y) ** 0.5:.0f} km<extra></extra>",
        )
    if selected in NETWORK_POS:  # signal path: bold arrow from sensor to farm
        x, y = NETWORK_POS[selected]
        dist = (x * x + y * y) ** 0.5
        mean_w = float(speeds10[selected].loc["2024"].mean())
        lead = dist * 1000 / max(mean_w, 1) / 60
        fig.add_annotation(
            x=0, y=0, ax=x, ay=y, xref="x", yref="y", axref="x", ayref="y",
            arrowhead=2, arrowsize=1.3, arrowwidth=2.5, arrowcolor=t["assets"][selected],
            opacity=0.85,
        )
        fig.add_annotation(
            x=x * 0.62, y=y * 0.62, text=f"~{lead:.0f} min", showarrow=False,
            font=dict(size=12.5, color=t["ink"]), bgcolor=t["card"],
            bordercolor=t["assets"][selected], borderwidth=1.5, borderpad=4,
        )
    fig.update_layout(
        xaxis=dict(range=[-45, 32], scaleanchor="y", constrain="domain",
                   showticklabels=False, title=None),
        yaxis=dict(range=[-34, 42], showticklabels=False, title=None),
    )
    return fig


def explorer_series(selected, window, theme):
    t = TOKENS[theme]
    if window == "all":
        view = speeds10
    elif window == "2024":
        view = speeds10.loc["2024"]
    else:
        view = speeds10.loc[speeds10.index.max() - pd.Timedelta(window):]
    fig = base_fig(theme, 240)
    fig.update_layout(margin=dict(l=46, r=12, t=6, b=34))
    if selected != "HYT-HY09":
        fig.add_scatter(x=view.index, y=view[selected], mode="lines",
                        name=selected.replace("_", " "),
                        line=dict(width=2, color=t["assets"][selected]))
    fig.add_scatter(x=view.index, y=view["HYT-HY09"], mode="lines", name="HYT-HY09",
                    line=dict(width=2, color=t["assets"]["HYT-HY09"]))
    fig.update_layout(hovermode="x unified", yaxis_title="Wind speed, m/s")
    return fig


def data_tab(theme, selected="Statfjord_A", window="30D"):
    t = TOKENS[theme]
    sel_speed = speeds10[selected]
    mean_2024 = float(sel_speed.loc["2024"].mean())
    corr = float(corr_2024.get(selected, 1.0))
    dist = ASSET_META[selected][1]
    lead = f"~{dist * 1000 / max(mean_2024, 1) / 60:.0f} min" if dist > 1 else "—"
    stats = [
        tile("Selected sensor", selected.replace("_", " "), ASSET_META[selected][0]),
        tile("Mean wind 2024", f"{mean_2024:.1f} m/s"),
        tile("Correlation with HY09", f"{corr:.3f}", "10-min means, 2024"),
        tile("Signal travel time", lead, "distance ÷ mean wind" if dist > 1 else "co-located"),
    ]
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H3("The sensor network"),
                                    html.P("34 signals · 1-min sampling · Jun 2023 – Dec 2024. "
                                           "Click a sensor — everything follows.", className="hint"),
                                    html.Div(sensor_buttons(selected, theme),
                                             className="sensor-list", id="sensor-list"),
                                ],
                                className="card",
                            ),
                            html.Div(stats, className="tiles two-col", id="sensor-stats"),
                            html.Details(
                                [
                                    html.Summary(f"The 12 hardest wind drops (worst "
                                                 f"{float(drops['drop_60min'].min()):+.1f} m/s in 60 min)"),
                                    html.Table(
                                        [
                                            html.Thead(html.Tr([html.Th("Origin (UTC)"), html.Th("Before"),
                                                                html.Th("Δ 60 min")])),
                                            html.Tbody(
                                                [
                                                    html.Tr([html.Td(f"{r.origin:%Y-%m-%d %H:%M}"),
                                                             html.Td(f"{r.speed_at_origin:.1f}"),
                                                             html.Td(f"{r.drop_60min:+.1f}")])
                                                    for r in drops.head(12).itertuples()
                                                ]
                                            ),
                                        ],
                                        className="mini",
                                    ),
                                ],
                                className="card", style={"padding": "12px 22px"},
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H3("Early-warning ring", style={"marginBottom": "0"}),
                                    html.P("The arrow is the signal path: upwind readings arrive at the "
                                           "farm after distance ÷ wind speed.", className="hint"),
                                    dcc.Graph(id="explorer-map", figure=explorer_map(selected, theme),
                                              config={"displayModeBar": False}),
                                ],
                                className="card",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.H3("Selected sensor vs HY09",
                                                    style={"marginRight": "auto"}),
                                            dcc.RadioItems(
                                                id="window",
                                                options=[{"label": l, "value": v} for l, v in
                                                         [("30 d", "30D"), ("90 d", "90D"),
                                                          ("2024", "2024"), ("All", "all")]],
                                                value=window, inline=True, className="radio-inline",
                                            ),
                                        ],
                                        style={"display": "flex", "alignItems": "baseline",
                                               "gap": "12px", "flexWrap": "wrap"},
                                    ),
                                    dcc.Graph(id="explorer-series",
                                              figure=explorer_series(selected, window, theme),
                                              config={"displayModeBar": False}),
                                ],
                                className="card",
                            ),
                        ],
                        className="split-right",
                    ),
                ],
                className="split",
            ),
        ]
    )


# ---------------------------------------------------------------- model comparison
def model_metric(h, model, key):
    return metrics[h]["models"][model][key]


def race_fig(horizon, regime, theme):
    t = TOKENS[theme]
    key = "rmse" if regime == "all" else "rmse_event"
    vals = [model_metric(f"h{horizon}", m, key) for m in MODEL_ORDER]
    best_i = int(np.argmin(vals))
    order = np.argsort(vals)[::-1]
    names = [MODEL_LABELS[MODEL_ORDER[i]] for i in order]
    v_sorted = [vals[i] for i in order]
    colors = [t["blue"] if i == best_i else t["baseline"] for i in order]
    fig = base_fig(theme, 300)
    fig.update_layout(transition=dict(duration=450, easing="cubic-in-out"))
    fig.add_bar(
        y=names, x=v_sorted, orientation="h", marker_color=colors, width=0.55,
        hovertemplate="%{y}: %{x:.3f} m/s<extra></extra>", showlegend=False,
    )
    for name, v in zip(names, v_sorted):
        fig.add_annotation(y=name, x=v, text=f"{v:.2f}", showarrow=False, xshift=24,
                           font=dict(size=12.5, color=t["ink"]))
    fig.update_layout(xaxis_title="RMSE, m/s (lower is better)",
                      margin=dict(l=170, r=50, t=8, b=40))
    return fig


def models_tab(theme, horizon=60, regime="events"):
    t = TOKENS[theme]
    key = "rmse" if regime == "all" else "rmse_event"
    ens = model_metric(f"h{horizon}", "ensemble", key)
    per = model_metric(f"h{horizon}", "persistence", key)
    gain = (1 - ens / per) * 100
    chip = lambda label, value, group, on: html.Button(
        label, id={"type": f"chip-{group}", "value": value}, className="chip on" if on else "chip")
    return html.Div(
        [
            html.Div(
                [
                    html.Span("Horizon", className="chip-label"),
                    chip("+30 min", 30, "horizon", horizon == 30),
                    chip("+60 min", 60, "horizon", horizon == 60),
                    html.Span("Conditions", className="chip-label", style={"marginLeft": "18px"}),
                    chip("All weather", "all", "regime", regime == "all"),
                    chip("Sudden changes", "events", "regime", regime == "events"),
                ],
                className="chip-row",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div([f"−{gain:.0f}", html.Span("%", className="unit")],
                                     className="headline"),
                            html.Div(
                                f"error vs the persistence benchmark · +{horizon} min · "
                                + ("event periods" if regime == "events" else "all conditions"),
                                className="headline-sub",
                            ),
                            html.Div(f"{ens:.2f} m/s vs {per:.2f} m/s", className="headline-sub"),
                        ],
                        className="card", style={"display": "flex", "flexDirection": "column",
                                                 "justifyContent": "center"},
                    ),
                    html.Div(
                        [
                            html.H3("The model race"),
                            html.P("Strictly out-of-time: trained Jun 2023 → Sep 2024, "
                                   "judged on Oct → Dec 2024. Never shuffled.", className="hint"),
                            dcc.Graph(id="race", figure=race_fig(horizon, regime, theme),
                                      config={"displayModeBar": False}),
                        ],
                        className="card",
                    ),
                ],
                style={"display": "grid", "gridTemplateColumns": "1fr 2fr", "gap": "18px"},
            ),
            html.Div(
                [
                    html.H3("Every number"),
                    html.Table(
                        [
                            html.Thead(html.Tr([html.Th(c) for c in
                                                ["Model", "RMSE +30", "RMSE +60", "Events +30", "Events +60"]])),
                            html.Tbody(
                                [
                                    html.Tr([html.Td(MODEL_LABELS[m]),
                                             html.Td(f"{model_metric('h30', m, 'rmse'):.3f}"),
                                             html.Td(f"{model_metric('h60', m, 'rmse'):.3f}"),
                                             html.Td(f"{model_metric('h30', m, 'rmse_event'):.3f}"),
                                             html.Td(f"{model_metric('h60', m, 'rmse_event'):.3f}")])
                                    for m in MODEL_ORDER
                                ]
                            ),
                        ],
                        className="mini",
                    ),
                    html.P("Ridge wins calm weather and loses events — the trees earn their keep "
                           "exactly when the operator needs them. Bands: 90% interval, honest "
                           "coverage, recalibrated on event-like episodes.", className="footnote"),
                ],
                className="card",
            ),
        ]
    )


# ---------------------------------------------------------------- operator view
def operator_tab(theme, idx=0, reveal=False):
    return html.Div(
        [
            html.Div(
                [
                    html.Button("◀ Previous", id="ep-prev", className="nav-btn"),
                    html.Div(id="event-label", className="event-label"),
                    html.Button("Next event ▶", id="ep-next", className="nav-btn"),
                    dcc.Checklist(
                        id="reveal",
                        options=[{"label": " Show what actually happened", "value": "on"}],
                        value=(["on"] if reveal else []), className="check-inline",
                        style={"marginLeft": "auto"},
                    ),
                ],
                className="event-nav card", style={"padding": "12px 22px"},
            ),
            html.Div(id="alarm-banner"),
            html.Div(id="operator-tiles", className="tiles"),
            html.Div(
                [
                    html.H3("HY09 wind forecast — calibrated 90% band"),
                    html.P("Three hours of 1-minute history, then the bagged LGB+XGB ensemble at "
                           "+30 and +60 min with the same calibrated band the submission carries.",
                           className="hint"),
                    dcc.Graph(id="fan-chart", config={"displayModeBar": False}),
                ],
                className="card",
            ),
        ]
    )


# ---------------------------------------------------------------- shell
app = Dash(__name__, title="Hywind Wind Intelligence", suppress_callback_exceptions=True)
app.layout = html.Div(
    [
        dcc.Location(id="url"),
        dcc.Store(id="theme", data="light"),
        dcc.Store(id="sel-asset", data="Statfjord_A"),
        dcc.Store(id="ep-idx", data=0),
        dcc.Store(id="m-horizon", data=60),
        dcc.Store(id="m-regime", data="events"),
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.Span([html.Span("HYWIND TAMPEN", className="accent"),
                                           " · Wind Intelligence"], className="brand"),
                                html.Span("30/60-min forecasts from the offshore sensor network",
                                          className="tagline"),
                                html.Button("☾ Dark", id="theme-btn", className="theme-btn"),
                            ],
                            className="brand-row",
                        ),
                        dcc.Tabs(
                            id="tabs", value="operator", className="tabs",
                            children=[
                                dcc.Tab(label="Data explorer", value="data", className="tab",
                                        selected_className="tab--selected"),
                                dcc.Tab(label="Model comparison", value="models", className="tab",
                                        selected_className="tab--selected"),
                                dcc.Tab(label="Operator view", value="operator", className="tab",
                                        selected_className="tab--selected"),
                            ],
                        ),
                    ],
                    className="header",
                ),
                html.Div(id="page", className="page"),
            ],
            id="app-root", className="app",
        ),
    ]
)


@app.callback(Output("theme", "data"), Output("theme-btn", "children"),
              Input("theme-btn", "n_clicks"), State("theme", "data"), prevent_initial_call=True)
def flip_theme(_, cur):
    new = "dark" if cur == "light" else "light"
    return new, ("☀ Light" if new == "dark" else "☾ Dark")


@app.callback(Output("app-root", "className"), Input("theme", "data"))
def theme_class(theme):
    return f"app {theme}" if theme == "dark" else "app"


@app.callback(Output("tabs", "value"), Output("theme", "data", allow_duplicate=True),
              Input("url", "search"), prevent_initial_call="initial_duplicate")
def tab_from_url(search):
    tab, theme = "operator", "light"
    if search:
        if "tab=" in search:
            wanted = search.split("tab=")[1].split("&")[0]
            if wanted in ("data", "models", "operator"):
                tab = wanted
        if "theme=dark" in search:
            theme = "dark"
    return tab, theme


@app.callback(Output("page", "children"), Input("tabs", "value"), Input("theme", "data"),
              State("sel-asset", "data"), State("ep-idx", "data"),
              State("m-horizon", "data"), State("m-regime", "data"))
def render_tab(tab, theme, sel, idx, horizon, regime):
    if tab == "data":
        return data_tab(theme, sel or "Statfjord_A")
    if tab == "models":
        return models_tab(theme, horizon or 60, regime or "events")
    return operator_tab(theme)


@app.callback(Output("sel-asset", "data"), Input({"type": "sensor-btn", "asset": ALL}, "n_clicks"),
              State("sel-asset", "data"), prevent_initial_call=True)
def pick_sensor(_, cur):
    if (ctx.triggered_id and isinstance(ctx.triggered_id, dict)
            and ctx.triggered and ctx.triggered[0].get("value")):
        return ctx.triggered_id["asset"]
    return cur


@app.callback(Output("explorer-map", "figure"), Output("explorer-series", "figure"),
              Output("sensor-stats", "children"), Output("sensor-list", "children"),
              Input("sel-asset", "data"), Input("window", "value"), State("theme", "data"),
              prevent_initial_call=True)
def update_explorer(sel, window, theme):
    theme = theme or "light"
    sel = sel or "Statfjord_A"
    sel_speed = speeds10[sel]
    mean_2024 = float(sel_speed.loc["2024"].mean())
    corr = float(corr_2024.get(sel, 1.0))
    dist = ASSET_META[sel][1]
    lead = f"~{dist * 1000 / max(mean_2024, 1) / 60:.0f} min" if dist > 1 else "—"
    stats = [
        tile("Selected sensor", sel.replace("_", " "), ASSET_META[sel][0]),
        tile("Mean wind 2024", f"{mean_2024:.1f} m/s"),
        tile("Correlation with HY09", f"{corr:.3f}", "10-min means, 2024"),
        tile("Signal travel time", lead, "distance ÷ mean wind" if dist > 1 else "co-located"),
    ]
    return (explorer_map(sel, theme), explorer_series(sel, window or "30D", theme),
            stats, sensor_buttons(sel, theme))


@app.callback(Output("m-horizon", "data"), Output("m-regime", "data"),
              Input({"type": "chip-horizon", "value": ALL}, "n_clicks"),
              Input({"type": "chip-regime", "value": ALL}, "n_clicks"),
              State("m-horizon", "data"), State("m-regime", "data"), prevent_initial_call=True)
def pick_chip(_h, _r, horizon, regime):
    trig = ctx.triggered_id
    if isinstance(trig, dict) and ctx.triggered and ctx.triggered[0].get("value"):
        if trig["type"] == "chip-horizon":
            horizon = trig["value"]
        else:
            regime = trig["value"]
    return horizon, regime


@app.callback(Output("page", "children", allow_duplicate=True),
              Input("m-horizon", "data"), Input("m-regime", "data"),
              State("tabs", "value"), State("theme", "data"), prevent_initial_call=True)
def rerender_models(horizon, regime, tab, theme):
    if tab != "models":
        from dash.exceptions import PreventUpdate

        raise PreventUpdate
    return models_tab(theme or "light", horizon or 60, regime or "events")


@app.callback(Output("ep-idx", "data"), Input("ep-prev", "n_clicks"), Input("ep-next", "n_clicks"),
              State("ep-idx", "data"), prevent_initial_call=True)
def step_episode(_p, _n, idx):
    idx = idx or 0
    if not (ctx.triggered and ctx.triggered[0].get("value")):
        return idx
    if ctx.triggered_id == "ep-next":
        return (idx + 1) % len(EPISODES)
    return (idx - 1) % len(EPISODES)


@app.callback(Output("alarm-banner", "children"), Output("operator-tiles", "children"),
              Output("fan-chart", "figure"), Output("event-label", "children"),
              Input("ep-idx", "data"), Input("reveal", "value"), State("theme", "data"))
def update_operator(idx, reveal, theme):
    theme = theme or "light"
    t = TOKENS[theme]
    idx = (idx or 0) % len(EPISODES)
    origin = EPISODES[idx]
    r30, r60 = p30.loc[origin], p60.loc[origin]
    s_now = float(r60["s_now"])

    lo30, hi30 = calibrated_band(30, float(r30["ensemble"]),
                                 float(r30["ensemble_q0.05"]), float(r30["ensemble_q0.95"]))
    lo60, hi60 = calibrated_band(60, float(r60["ensemble"]),
                                 float(r60["ensemble_q0.05"]), float(r60["ensemble_q0.95"]))
    dp = float(r60["drop_prob"]) if "drop_prob" in r60.index and pd.notna(r60["drop_prob"]) else None
    state = evaluate_alarm(s_now, float(r30["ensemble"]), float(r60["ensemble"]), lo30, lo60,
                           drop_prob=dp)

    banner = html.Div(
        [
            html.Div(
                [
                    html.Div(state.icon, className="alarm-icon", style={"background": state.color}),
                    html.Div(
                        [
                            html.Div(f"Wind-Drop Alarm: {state.label}", className="alarm-title"),
                            html.Div(state.headline, className="alarm-headline"),
                            html.Div(state.action, className="alarm-action"),
                        ]
                    ),
                ],
                style={"display": "flex", "gap": "16px", "alignItems": "center"},
            ),
            html.Div(
                f"Track record on {TRACK['n']:,} unseen hours (Oct–Dec 2024): catches "
                f"{TRACK['severe_recall']:.0f}% of severe drops (≥4 m/s) · alarms are "
                f"{TRACK['lift']:.1f}× better than chance · {TRACK['quiet']:.0f}% of quiet "
                f"hours stay alarm-free",
                className="alarm-track",
            ),
        ],
        className="alarm-banner alarm-banner-col", style={"borderLeftColor": state.color},
    )

    tiles = [
        tile("Wind now", f"{s_now:.1f} m/s", f"at {origin:%H:%M} UTC"),
        tile("Expected +30 min", f"{r30['ensemble']:.1f} m/s", f"{r30['ensemble'] - s_now:+.1f} vs now"),
        tile("Expected +60 min", f"{r60['ensemble']:.1f} m/s", f"{r60['ensemble'] - s_now:+.1f} vs now"),
        tile("Worst case +60 min", f"{lo60:.1f} m/s", "q0.05, calibrated"),
    ]
    if "drop_prob" in r60.index and pd.notna(r60["drop_prob"]):
        tiles.append(tile("P(drop > 2 m/s, 60 min)", f"{r60['drop_prob'] * 100:.0f}%",
                          "dedicated drop classifier"))

    hist = speeds1["HYT-HY09"].loc[origin - pd.Timedelta("180min"): origin]
    fig = base_fig(theme, 400)
    band_x = [origin, origin + pd.Timedelta("30min"), origin + pd.Timedelta("60min")]
    fig.add_scatter(x=band_x, y=[s_now, hi30, hi60], mode="lines",
                    line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=band_x, y=[s_now, lo30, lo60], mode="lines", line=dict(width=0),
                    fill="tonexty", fillcolor=t["wash"], name="90% band (calibrated)")
    fig.add_scatter(x=hist.index, y=hist.values, mode="lines", name="History (1-min)",
                    line=dict(width=2, color=t["blue"]))
    fig.add_scatter(
        x=band_x, y=[s_now, r30["ensemble"], r60["ensemble"]],
        mode="lines+markers", name="Forecast (LGB+XGB ensemble)",
        line=dict(width=2, color=t["blue"], dash="dot"),
        marker=dict(size=10, color=t["blue"], line=dict(width=2, color=t["card"])),
    )
    if "on" in (reveal or []):
        actual = speeds1["HYT-HY09"].loc[origin: origin + pd.Timedelta("75min")]
        fig.add_scatter(x=actual.index, y=actual.values, mode="lines", name="What happened",
                        line=dict(width=2, color=t["ink"]))
    fig.add_vline(x=origin, line_width=1, line_color=t["baseline"])
    fig.add_annotation(x=origin, y=1, yref="paper", text="now", showarrow=False,
                       font=dict(size=11, color=t["muted"]), yshift=8)
    # explicit frame per event: no transition artifacts, always well-filled
    y_vals = [float(hist.min()), float(hist.max()), lo30, lo60, hi30, hi60]
    if "on" in (reveal or []):
        y_vals += [float(actual.min()), float(actual.max())]
    pad = max(0.8, (max(y_vals) - min(y_vals)) * 0.08)
    fig.update_layout(
        hovermode="x unified", yaxis_title="Wind speed, m/s",
        xaxis_range=[origin - pd.Timedelta("185min"), origin + pd.Timedelta("70min")],
        yaxis_range=[min(y_vals) - pad, max(y_vals) + pad],
    )

    label = html.Div(
        [
            html.Div(f"{origin:%a %d %b %Y · %H:%M} UTC"),
            html.Div(f"event {idx + 1} of {len(EPISODES)} · actual 60-min change "
                     f"{change.loc[origin]:+.1f} m/s", className="event-count"),
        ]
    )
    return banner, tiles, fig, label


server = app.server  # for gunicorn: gunicorn -w 2 -b 0.0.0.0:8506 app:server

if __name__ == "__main__":
    import os

    app.run(host=os.environ.get("HOST", "0.0.0.0"),
            port=int(os.environ.get("PORT", "8506")), debug=False)
