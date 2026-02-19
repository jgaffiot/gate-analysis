import marimo

__generated_with = "0.19.11"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import base64
    import importlib
    import marimo as mo
    from bokeh.embed import file_html
    from bokeh.layouts import column
    from bokeh.models import Label, Span
    from bokeh.plotting import figure as bk_figure
    from bokeh.resources import CDN
    from gate_analysis.common import generate_synthetic_data, plot_results

    return (
        CDN,
        Label,
        Span,
        base64,
        bk_figure,
        column,
        file_html,
        generate_synthetic_data,
        importlib,
        mo,
        plot_results,
    )


@app.cell(hide_code=True)
def _(base64, file_html, CDN, mo):
    def bk(fig, height=650):
        """Embed a bokeh figure or layout in marimo via a base64-encoded iframe."""
        html = file_html(fig, CDN)
        enc = base64.b64encode(html.encode()).decode()
        return mo.Html(
            f'<iframe src="data:text/html;base64,{enc}" '
            f'width="100%" height="{height}px" frameborder="0" scrolling="no">'
            f"</iframe>"
        )

    return (bk,)


@app.cell(hide_code=True)
def _(mo):
    return mo.md(
        r"""
        # Gate Closing Analysis — Method Comparison

        Seven methods for detecting breakpoints and measuring slopes in pipe gate closing signals.

        **Signal structure:** high plateau → fast linear ramp → slow linear ramp → low plateau

        **True breakpoints:** 2.0 s, 5.0 s, 9.0 s &nbsp;|&nbsp; **True slopes:** −25.0 %/s, −5.0 %/s

        Green dotted lines = ground truth breakpoints. Red dashed lines = detected breakpoints.
"""
    )


# ── Synthetic data ────────────────────────────────────────────────────────────


@app.cell(hide_code=True)
def _(generate_synthetic_data):
    data = generate_synthetic_data()
    return (data,)


@app.cell(hide_code=True)
def _(mo):
    return mo.md("## Raw synthetic data")


@app.cell
def _(bk, data, plot_results):
    return bk(plot_results(data, "Synthetic Gate Closing Data"))


# ── Method 1 ─────────────────────────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo):
    return mo.md(
        r"""
        ## Method 1 — Segmented Regression (Muggeo)
        *Library:* `piecewise-regression` &nbsp;|&nbsp; *Best accuracy / speed trade-off*
        """
    )


@app.cell
def _(bk, data, importlib, plot_results):
    _m = importlib.import_module("gate_analysis.1_segmented_regression")
    _result = _m.segmented_regression(data.time, data.position)
    _segs = _m._build_segments(data, _result)
    return bk(
        plot_results(
            data,
            "Method 1 — Segmented Regression (Muggeo)",
            fitted_segments=_segs,
            detected_breakpoints=_result["breakpoints"],
            estimated_slopes=_result["slopes"],
        )
    )


# ── Method 2 ─────────────────────────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo):
    return mo.md(
        r"""
        ## Method 2 — Bayesian Change-Point Model (PyMC + ADVI)
        *Library:* `pymc` &nbsp;|&nbsp; *Best UQ; approximate Gaussian posterior via variational inference (~5 s)*
        """
    )


@app.cell
def _(bk, data, importlib, plot_results):
    _m = importlib.import_module("gate_analysis.2_bayesian_changepoint")
    _result = _m.bayesian_changepoint(data.time, data.position)
    _segs = _m._build_segments(data, _result)
    return bk(
        plot_results(
            data,
            "Method 2 — Bayesian Change-Point (PyMC + ADVI)",
            fitted_segments=_segs,
            detected_breakpoints=_result["breakpoints"],
            estimated_slopes=_result["slopes"],
        )
    )


# ── Method 3 ─────────────────────────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo):
    return mo.md(
        r"""
        ## Method 3 — CPOP Piecewise Linear (L0 penalty)
        *Library:* numpy only (DP + BIC) &nbsp;|&nbsp; *Accurate, ~4 s*
        """
    )


@app.cell
def _(bk, data, importlib, plot_results):
    _m = importlib.import_module("gate_analysis.3_cpop_piecewise_linear")
    _result = _m.cpop_piecewise_linear(data.time, data.position)
    _segs = _m._build_segments(data, _result)
    return bk(
        plot_results(
            data,
            "Method 3 — CPOP Piecewise Linear (L0 penalty)",
            fitted_segments=_segs,
            detected_breakpoints=_result["breakpoints"],
            estimated_slopes=_result["slopes"],
        )
    )


# ── Method 4 ─────────────────────────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo):
    return mo.md(
        r"""
        ## Method 4 — ruptures + OLS
        *Library:* `ruptures` + `scipy` &nbsp;|&nbsp; *Less accurate (L2 cost)*
        """
    )


@app.cell
def _(bk, data, importlib, plot_results):
    _m = importlib.import_module("gate_analysis.4_ruptures_ols")
    _result = _m.ruptures_ols(data.time, data.position)
    _segs = _m._build_segments(data, _result)
    return bk(
        plot_results(
            data,
            "Method 4 — ruptures + OLS",
            fitted_segments=_segs,
            detected_breakpoints=_result["breakpoints"],
            estimated_slopes=_result["slopes"],
        )
    )


# ── Method 5 ─────────────────────────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo):
    return mo.md(
        r"""
        ## Method 5 — Savitzky-Golay Derivative
        *Library:* `scipy.signal` &nbsp;|&nbsp; *Heuristic, fast*
        """
    )


@app.cell
def _(Label, Span, bk, bk_figure, column, data, importlib):
    _m = importlib.import_module("gate_analysis.5_savitzky_golay")
    _r = _m.savitzky_golay(data.time, data.position)

    _p1 = bk_figure(
        width=1100,
        height=370,
        title="Method 5 — Savitzky-Golay: signal + smoothed",
        y_axis_label="Gate position (%)",
    )
    _p1.scatter(
        data.time, data.position, marker="circle", color="gray", alpha=0.3, size=2
    )
    _p1.line(
        data.time,
        _r["smoothed"],
        line_color="blue",
        line_width=1.5,
        legend_label="SG smoothed",
    )
    for _bp in _r["breakpoints"]:
        _p1.add_layout(
            Span(
                location=_bp,
                dimension="height",
                line_color="red",
                line_dash="dashed",
                line_alpha=0.7,
            )
        )
    for _bp in data.breakpoints:
        _p1.add_layout(
            Span(
                location=_bp,
                dimension="height",
                line_color="green",
                line_dash="dotted",
                line_alpha=0.4,
            )
        )
    _p1.legend.location = "top_right"
    _p1.grid.grid_line_alpha = 0.3

    _p2 = bk_figure(
        width=1100,
        height=310,
        title="First derivative (instantaneous slope)",
        x_axis_label="Time (s)",
        y_axis_label="Derivative (%/s)",
        x_range=_p1.x_range,
    )
    _p2.line(data.time, _r["derivative"], line_color="blue", line_width=1)
    _p2.add_layout(
        Span(location=0, dimension="width", line_color="black", line_width=0.5)
    )
    for _bp in _r["breakpoints"]:
        _p2.add_layout(
            Span(
                location=_bp,
                dimension="height",
                line_color="red",
                line_dash="dashed",
                line_alpha=0.7,
            )
        )
    _p2.add_layout(
        Label(
            x=10,
            y=10,
            x_units="screen",
            y_units="screen",
            text=f"Fast slope: {_r['slopes'][0]:.2f} %/s\nSlow slope: {_r['slopes'][1]:.2f} %/s\nTrue: {data.slopes[0]:.1f}, {data.slopes[1]:.1f} %/s",
            text_font_size="9pt",
            background_fill_color="wheat",
            background_fill_alpha=0.8,
        )
    )
    _p2.grid.grid_line_alpha = 0.3

    return bk(column(_p1, _p2), height=730)


# ── Method 6 ─────────────────────────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo):
    return mo.md(
        r"""
        ## Method 6 — Kalman Filter
        *Library:* `filterpy` &nbsp;|&nbsp; *Heuristic, online-capable*
        """
    )


@app.cell
def _(Label, Span, bk, bk_figure, column, data, importlib):
    _m = importlib.import_module("gate_analysis.6_kalman_filter")
    _r = _m.kalman_filter(data.time, data.position)

    _p1 = bk_figure(
        width=1100,
        height=370,
        title="Method 6 — Kalman Filter: position",
        y_axis_label="Gate position (%)",
    )
    _p1.scatter(
        data.time, data.position, marker="circle", color="gray", alpha=0.3, size=2
    )
    _p1.line(
        data.time,
        _r["filtered_position"],
        line_color="blue",
        line_width=1.5,
        legend_label="Kalman filtered",
    )
    for _bp in _r["breakpoints"]:
        _p1.add_layout(
            Span(
                location=_bp,
                dimension="height",
                line_color="red",
                line_dash="dashed",
                line_alpha=0.7,
            )
        )
    for _bp in data.breakpoints:
        _p1.add_layout(
            Span(
                location=_bp,
                dimension="height",
                line_color="green",
                line_dash="dotted",
                line_alpha=0.4,
            )
        )
    _p1.legend.location = "top_right"
    _p1.grid.grid_line_alpha = 0.3

    _p2 = bk_figure(
        width=1100,
        height=310,
        title="Kalman-estimated velocity (instantaneous slope)",
        x_axis_label="Time (s)",
        y_axis_label="Velocity (%/s)",
        x_range=_p1.x_range,
    )
    _p2.line(data.time, _r["filtered_velocity"], line_color="blue", line_width=1)
    _p2.add_layout(
        Span(location=0, dimension="width", line_color="black", line_width=0.5)
    )
    for _bp in _r["breakpoints"]:
        _p2.add_layout(
            Span(
                location=_bp,
                dimension="height",
                line_color="red",
                line_dash="dashed",
                line_alpha=0.7,
            )
        )
    _p2.add_layout(
        Label(
            x=10,
            y=10,
            x_units="screen",
            y_units="screen",
            text=f"Fast slope: {_r['slopes'][0]:.2f} %/s\nSlow slope: {_r['slopes'][1]:.2f} %/s\nTrue: {data.slopes[0]:.1f}, {data.slopes[1]:.1f} %/s",
            text_font_size="9pt",
            background_fill_color="wheat",
            background_fill_alpha=0.8,
        )
    )
    _p2.grid.grid_line_alpha = 0.3

    return bk(column(_p1, _p2), height=730)


# ── Method 8 ─────────────────────────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo):
    return mo.md(
        r"""
        ## Method 8 — Narrowest-Over-Threshold (NOT)
        *Library:* numpy + scipy only (custom reimplementation) &nbsp;|&nbsp; *Accurate, no prior on number of breakpoints, ~1 s*
        """
    )


@app.cell
def _(bk, data, importlib, plot_results):
    _m = importlib.import_module("gate_analysis.8_not_detection")
    _result = _m.not_detection(data.time, data.position)
    _segs = _m._build_segments(data, _result)
    return bk(
        plot_results(
            data,
            "Method 8 — Narrowest-Over-Threshold (NOT)",
            fitted_segments=_segs,
            detected_breakpoints=_result["breakpoints"],
            estimated_slopes=_result["slopes"],
        )
    )


if __name__ == "__main__":
    app.run()
