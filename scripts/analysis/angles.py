"""Joint angle solutions of the two link height equation, explored in a browser dashboard.

A planar two link leg whose upper link makes the angle theta1 with the downward vertical, and whose
lower link makes the angle pi/2 + theta1 - theta2 with it, spans the vertical height

    h = l1 cos(theta1) + l2 cos(pi/2 + theta1 - theta2) = l1 cos(theta1) - l2 sin(theta1 - theta2).

This is one equation in two unknowns, so for fixed h, l1 and l2 its solutions form a curve in the
(theta1, theta2) torus and not a finite set. Three closed forms describe that curve.

* For a fixed theta1, sin(theta1 - theta2) = (l1 cos(theta1) - h) / l2 =: s. A solution exists if
  and only if |s| <= 1, and then theta2 = theta1 - asin(s) or theta2 = theta1 - pi + asin(s), two
  solutions that merge into one at |s| = 1.
* For a fixed theta2, expanding the sine gives h = R cos(theta1 + alpha) with
  R = sqrt(l1^2 + l2^2 + 2 l1 l2 sin(theta2)) and alpha = atan2(l2 cos(theta2), l1 + l2 sin(theta2)),
  hence theta1 = -alpha +- acos(h / R), again two solutions that merge at |h| = R.
* R ranges over [|l1 - l2|, l1 + l2], which fixes the topology of the curve. With l1, l2 > 0 and
  g = |l1 - l2|, s = l1 + l2, the set is empty for |h| > s, a single point for |h| = s, one closed
  loop for g < |h| < s, two curves that touch at theta2 = -pi/2 for |h| = g (a whole line there
  when l1 = l2 and h = 0), and two disjoint curves that each wrap theta2 for |h| < g.

The dashboard draws that curve in the (theta1, theta2) plane with sliders for h, l1 and l2, states
which of the regimes above the current values fall in, and draws the leg for the solutions that
share a chosen theta1 or theta2, so that the count of solutions at that angle is visible and its
residual against the equation is reported.

Usage:
    python scripts/analysis/angles.py [--host 0.0.0.0] [--port 8052]
    python scripts/analysis/angles.py --check

The --check flag runs the numerical self test of the closed forms and needs only numpy. The
dashboard itself needs dash and plotly, the same packages as dashboard.py, which is the reason they
are imported inside build_app and not at module level.
"""

from __future__ import annotations

import argparse
import math

import numpy as np

TWO_PI = 2.0 * math.pi
EPS = 1e-9

# Colours of the two solution branches, fixed so that a branch keeps its identity as the sliders
# move. Branch 0 is theta2 = theta1 - asin(s) and branch 1 is theta2 = theta1 - pi + asin(s).
BRANCH_COLOURS = ("#2a6fdb", "#e0701a")
BRANCH_NAMES = ("theta2 = theta1 - asin(s)", "theta2 = theta1 - pi + asin(s)")


def wrap(x):
    """Wrap an angle, or an array of angles, to [-pi, pi)."""
    return (np.asarray(x) + math.pi) % TWO_PI - math.pi


def height(theta1, theta2, l1, l2):
    """The height h of the equation, evaluated exactly as it is written."""
    return l1 * np.cos(theta1) + l2 * np.cos(math.pi / 2.0 + theta1 - theta2)


def solve_theta2(theta1, h, l1, l2):
    """The theta2 solutions in [-pi, pi) for a given theta1, as a list of (branch, theta2)."""
    s = (l1 * math.cos(theta1) - h) / l2
    if abs(s) > 1.0 + EPS:
        return []
    a = math.asin(max(-1.0, min(1.0, s)))
    out = [(0, float(wrap(theta1 - a)))]
    if abs(abs(s) - 1.0) > EPS:
        out.append((1, float(wrap(theta1 - math.pi + a))))
    return out


def solve_theta1(theta2, h, l1, l2):
    """The theta1 solutions in [-pi, pi) for a given theta2, as a list of (branch, theta1).

    The branch is the one whose theta2 = theta1 - asin(s) or theta1 - pi + asin(s) the pair
    belongs to, told apart by the sign of cos(theta1 - theta2), which is not negative on branch 0.
    Where R = 0, which needs l1 = l2 and theta2 = -pi/2, every theta1 solves the equation if h = 0
    and none does otherwise, and the empty list returned for the first case is the one place
    where this function does not enumerate the solutions.
    """
    a = l1 + l2 * math.sin(theta2)
    b = l2 * math.cos(theta2)
    r = math.hypot(a, b)
    if r < EPS or abs(h) > r + EPS:
        return []
    alpha = math.atan2(b, a)
    d = math.acos(max(-1.0, min(1.0, h / r)))
    thetas = [-alpha + d] if d < EPS or abs(d - math.pi) < EPS else [-alpha + d, -alpha - d]
    return [(0 if math.cos(t - theta2) >= 0.0 else 1, float(wrap(t))) for t in thetas]


def regime(h, l1, l2):
    """Name the topology of the solution set and give the number of solution components."""
    reach, gap, ah = l1 + l2, abs(l1 - l2), abs(h)
    if ah > reach + EPS:
        return "none", "No solution. |h| exceeds the reach l1 + l2, so the leg cannot span this height."
    if abs(ah - reach) <= EPS:
        return "point", "Exactly one solution, modulo 2 pi. The leg is fully extended or fully folded back."
    if abs(ah - gap) <= EPS:
        return "touching", ("Infinitely many solutions. The two curves touch at theta2 = -90 deg, which "
                            "is where |h| equals |l1 - l2|.")
    if ah > gap:
        return "loop", "Infinitely many solutions, forming one closed loop in the (theta1, theta2) torus."
    return "pair", ("Infinitely many solutions, forming two disjoint curves that each wrap the "
                    "theta2 axis. Every theta2 admits two theta1.")


def solution_curves(h, l1, l2, n=3601):
    """The two branches of the solution curve as arrays (theta1, theta2) in degrees.

    Each branch is cut at the +-180 degree seam and at the angles where no solution exists, with
    NaN, so that a line plot does not draw across them. The theta1 grid carries the exact angles at
    which |s| = 1, so that the branches meet there instead of stopping short of each other.
    """
    grid = list(np.linspace(-math.pi, math.pi, n))
    for target in (h + l2, h - l2):
        c = target / l1
        if abs(c) <= 1.0:
            grid += [math.acos(c), -math.acos(c)]
    t1 = np.unique(np.array(grid))
    s = (l1 * np.cos(t1) - h) / l2
    ok = np.abs(s) <= 1.0 + EPS
    a = np.arcsin(np.clip(s, -1.0, 1.0))
    curves = []
    for t2 in (wrap(t1 - a), wrap(t1 - math.pi + a)):
        x = np.degrees(t1)
        y = np.degrees(np.where(ok, t2, np.nan))
        cut = np.where(np.abs(np.diff(y)) > 180.0)[0]
        curves.append((np.insert(x, cut + 1, np.nan), np.insert(y, cut + 1, np.nan)))
    return curves


def leg_points(theta1, theta2, l1, l2):
    """Hip, knee and foot in a frame with y up, the hip at the origin and the legs hanging down.

    Link 1 makes theta1 with the downward vertical and link 2 makes pi/2 + theta1 - theta2, so the
    foot lies a vertical distance h below the hip, matching the equation.
    """
    beta = math.pi / 2.0 + theta1 - theta2
    knee = (l1 * math.sin(theta1), -l1 * math.cos(theta1))
    foot = (knee[0] + l2 * math.sin(beta), knee[1] - l2 * math.cos(beta))
    return (0.0, 0.0), knee, foot


def self_check():
    """Verify the closed forms against the equation, by dense sampling and against brute force."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(2000):
        l1, l2 = rng.uniform(0.05, 1.5, 2)
        h = rng.uniform(-(l1 + l2), l1 + l2)
        t1, t2 = rng.uniform(-math.pi, math.pi, 2)
        for _, th2 in solve_theta2(t1, h, l1, l2):
            worst = max(worst, abs(float(height(t1, th2, l1, l2)) - h))
        for _, th1 in solve_theta1(t2, h, l1, l2):
            worst = max(worst, abs(float(height(th1, t2, l1, l2)) - h))
        for x, y in solution_curves(h, l1, l2, n=401):
            m = ~np.isnan(x)
            res = height(np.radians(x[m]), np.radians(y[m]), l1, l2) - h
            worst = max(worst, float(np.max(np.abs(res))) if res.size else 0.0)
    assert worst < 1e-8, f"closed form residual {worst:.3e}"
    # Count against a brute force grid: the number of theta2 roots at a fixed theta1 is 0, 1 or 2.
    l1, l2, h = 0.4, 0.5, 0.6
    for t1 in np.linspace(-math.pi, math.pi, 73):
        t2 = np.linspace(-math.pi, math.pi, 200001)
        f = height(t1, t2, l1, l2) - h
        brute = int(np.sum(np.sign(f[:-1]) * np.sign(f[1:]) < 0))
        assert brute == len(solve_theta2(t1, h, l1, l2)) or abs(abs((l1 * math.cos(t1) - h) / l2) - 1) < 1e-3
    # The regime thresholds.
    assert regime(1.0, 0.4, 0.5)[0] == "none"
    assert regime(0.9, 0.4, 0.5)[0] == "point"
    assert regime(0.5, 0.4, 0.5)[0] == "loop"
    assert regime(0.1, 0.4, 0.5)[0] == "touching"
    assert regime(0.05, 0.4, 0.5)[0] == "pair"
    print(f"[OK] closed forms agree with the equation, worst residual {worst:.2e}")


def build_app():
    import dash
    import plotly.graph_objects as go
    from dash import Input, Output, dcc, html

    app = dash.Dash(__name__)
    app.title = "Two link height solutions"

    def slider(sid, label, lo, hi, step, value, marks):
        return html.Div([
            html.Label(label, style={"fontWeight": "600"}),
            dcc.Slider(id=sid, min=lo, max=hi, step=step, value=value, marks=marks,
                       tooltip={"placement": "bottom", "always_visible": True}, updatemode="drag"),
        ], style={"marginBottom": "28px"})

    # String keys, since orjson, which plotly uses when installed, rejects numpy scalars as keys and
    # some releases reject floats too. Dash reads a numeric string key as the slider position.
    marks = lambda lo, hi, k: {f"{float(v):g}": f"{float(v):g}" for v in np.linspace(lo, hi, k)}
    controls = html.Div([
        html.H3("Parameters"),
        slider("h", "h, height spanned", -3.0, 3.0, 0.01, 0.6, marks(-3, 3, 7)),
        slider("l1", "l1, upper link", 0.05, 1.5, 0.01, 0.4, marks(0.05, 1.5, 7)),
        slider("l2", "l2, lower link", 0.05, 1.5, 0.01, 0.5, marks(0.05, 1.5, 7)),
        html.H3("Probe"),
        dcc.RadioItems(id="fixed", value="theta1", inline=True,
                       options=[{"label": " fix theta1", "value": "theta1"},
                                {"label": " fix theta2", "value": "theta2"}],
                       style={"marginBottom": "12px"}),
        slider("angle", "Fixed angle, degrees", -180, 180, 0.5, 30, marks(-180, 180, 9)),
        html.Div(id="status", style={"lineHeight": "1.5"}),
    ], style={"flex": "0 0 360px", "minWidth": "300px"})

    graphs = html.Div([
        dcc.Graph(id="curve", style={"flex": "1 1 480px", "minWidth": "320px"}),
        dcc.Graph(id="leg", style={"flex": "1 1 380px", "minWidth": "320px"}),
    ], style={"display": "flex", "flexWrap": "wrap", "flex": "1 1 700px", "gap": "8px"})

    app.layout = html.Div([
        html.H2("h = l1 cos(theta1) + l2 cos(pi/2 + theta1 - theta2)"),
        html.Div([controls, graphs], style={"display": "flex", "flexWrap": "wrap", "gap": "24px"}),
    ], style={"fontFamily": "system-ui, sans-serif", "padding": "16px 24px"})

    @app.callback(Output("curve", "figure"), Output("leg", "figure"), Output("status", "children"),
                  Input("h", "value"), Input("l1", "value"), Input("l2", "value"),
                  Input("fixed", "value"), Input("angle", "value"))
    def update(h, l1, l2, fixed, angle_deg):
        angle = math.radians(angle_deg)
        sols = solve_theta2(angle, h, l1, l2) if fixed == "theta1" else solve_theta1(angle, h, l1, l2)
        pairs = [(angle, v) if fixed == "theta1" else (v, angle) for _, v in sols]
        branches = [br for br, _ in sols]

        fig = go.Figure()
        for k, (x, y) in enumerate(solution_curves(h, l1, l2)):
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=BRANCH_NAMES[k], connectgaps=False,
                                     line=dict(color=BRANCH_COLOURS[k], width=2),
                                     hovertemplate="theta1 %{x:.1f}°<br>theta2 %{y:.1f}°<extra></extra>"))
        if fixed == "theta1":
            fig.add_vline(x=angle_deg, line=dict(color="gray", dash="dot", width=1))
        else:
            fig.add_hline(y=angle_deg, line=dict(color="gray", dash="dot", width=1))
        if pairs:
            fig.add_trace(go.Scatter(
                x=[math.degrees(p[0]) for p in pairs], y=[math.degrees(p[1]) for p in pairs],
                mode="markers", name="probe solutions", showlegend=False,
                marker=dict(size=11, color=[BRANCH_COLOURS[b] for b in branches],
                            line=dict(color="white", width=2))))
        ticks = dict(range=[-180, 180], tickvals=list(range(-180, 181, 90)), zeroline=False)
        fig.update_layout(template="plotly_white", title="Solution set in the joint angle plane",
                          xaxis=dict(title="theta1 (deg)", **ticks),
                          yaxis=dict(title="theta2 (deg)", scaleanchor="x", scaleratio=1, **ticks),
                          legend=dict(orientation="h", y=-0.2), margin=dict(l=60, r=20, t=50, b=40))

        reach = l1 + l2
        leg = go.Figure()
        leg.add_trace(go.Scatter(x=[-reach, reach], y=[-h, -h], mode="lines", name="ground",
                                 line=dict(color="gray", width=2), hoverinfo="skip"))
        for br, (t1, t2) in zip(branches, pairs):
            hip, knee, foot = leg_points(t1, t2, l1, l2)
            leg.add_trace(go.Scatter(x=[hip[0], knee[0], foot[0]], y=[hip[1], knee[1], foot[1]],
                                     mode="lines+markers", name=f"θ1 {math.degrees(t1):.1f}°, θ2 {math.degrees(t2):.1f}°",
                                     line=dict(color=BRANCH_COLOURS[br], width=4), marker=dict(size=9)))
        lim = reach * 1.1
        leg.update_layout(template="plotly_white", title="Leg for the probed angle",
                          xaxis=dict(range=[-lim, lim], title="x"),
                          yaxis=dict(range=[-lim, lim], title="y, hip at 0", scaleanchor="x", scaleratio=1),
                          legend=dict(orientation="h", y=-0.2), margin=dict(l=60, r=20, t=50, b=40))

        residual = max((abs(float(height(*p, l1, l2)) - h) for p in pairs), default=0.0)
        n = len(pairs)
        status = [
            html.P(regime(h, l1, l2)[1]),
            html.P(f"Reach l1 + l2 = {reach:.3f}, gap |l1 - l2| = {abs(l1 - l2):.3f}."),
            html.P(f"Fixing {fixed} = {angle_deg:g}° gives {n} solution{'s' if n != 1 else ''}"
                   + (f", residual {residual:.1e}." if n else ".")),
        ]
        return fig, leg, status

    return app


def main():
    parser = argparse.ArgumentParser(description="Solutions of h = l1 cos(t1) + l2 cos(pi/2 + t1 - t2)")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8052)
    parser.add_argument("--check", action="store_true", help="Run the numerical self test and exit.")
    args = parser.parse_args()
    if args.check:
        self_check()
        return
    build_app().run(debug=False, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
