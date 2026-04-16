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
    from bokeh.io import show
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
        show,
    )


@app.cell(hide_code=True)
def _(CDN, base64, file_html, mo):
    def bk(fig, height=650):
        """Embed a bokeh figure or layout in marimo via a base64-encoded iframe."""
        html = file_html(fig, CDN)
        enc = base64.b64encode(html.encode()).decode()
        return mo.Html(
            f'<iframe src="data:text/html;base64,{enc}" '
            f'width="100%" height="{height}px" frameborder="0" scrolling="no">'
            f"</iframe>"
        )

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Gate Closing Analysis — Method Comparison

    Seven methods for detecting breakpoints and measuring slopes in pipe gate closing signals.

    **Signal structure:** high plateau → fast linear ramp → slow linear ramp → low plateau

    **True breakpoints:** 2.0 s, 5.0 s, 9.0 s &nbsp;|&nbsp; **True slopes:** −25.0 %/s, −5.0 %/s

    Green dotted lines = ground truth breakpoints. Red dashed lines = detected breakpoints.
    """)
    return


@app.cell(hide_code=True)
def _(generate_synthetic_data):
    data = generate_synthetic_data()
    return (data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Method 0 — Direct Curve Fit (scipy)
    *Library:* `scipy.optimize.curve_fit` &nbsp;|&nbsp; *Simplest and fastest; standard errors on slopes for free*
    """)
    return


@app.cell
def _(data, importlib, plot_results, show):
    _m = importlib.import_module("gate_analysis.0_curve_fit")
    _results, _segs = _m.analyze(data)
    _fig = plot_results(
        data,
        "Method 0: Direct Curve Fit (scipy)",
        fitted_segments=_segs,
        detected_breakpoints={c: r["breakpoints"] for c, r in _results.items()},
        estimated_slopes={c: r["slopes"] for c, r in _results.items()},
    )
    show(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Method 1 — Segmented Regression (Muggeo)
    *Library:* `piecewise-regression` &nbsp;|&nbsp; *Best accuracy / speed trade-off (~0.9 s with n\_boot=10)*
    """)
    return


@app.cell
def _(data, importlib, plot_results, show):
    _m = importlib.import_module("gate_analysis.1_segmented_regression")
    _results, _segs = _m.analyze(data)
    _fig = plot_results(
        data,
        "Method 1: Segmented Regression (Muggeo)",
        fitted_segments=_segs,
        detected_breakpoints={c: r["breakpoints"] for c, r in _results.items()},
        estimated_slopes={c: r["slopes"] for c, r in _results.items()},
    )
    show(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Method 2 — Bayesian MAP + Laplace (scipy)
    *Library:* `scipy.optimize` &nbsp;|&nbsp; *Full Bayesian model; MAP via Powell + Laplace covariance for 95 % CIs (~0.25 s)*
    """)
    return


@app.cell
def _(data, importlib, plot_results, show):
    _m = importlib.import_module("gate_analysis.2_bayesian_changepoint")
    _results, _segs = _m.analyze(data)
    _fig = plot_results(
        data,
        "Method 2: Bayesian Change-Point Model (PyMC + ADVI)",
        fitted_segments=_segs,
        detected_breakpoints={c: r["breakpoints"] for c, r in _results.items()},
        estimated_slopes={c: r["slopes"] for c, r in _results.items()},
    )
    show(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Method 3 — CPOP Piecewise Linear (L0 penalty)
    *Library:* numpy only (DP + BIC) &nbsp;|&nbsp; *Accurate, ~4 s*
    """)
    return


@app.cell
def _(data, importlib, plot_results, show):
    _m = importlib.import_module("gate_analysis.3_cpop_piecewise_linear")
    _results, _segs = _m.analyze(data)
    _fig = plot_results(
        data,
        "Method 3: CPOP-like Continuous Piecewise Linear",
        fitted_segments=_segs,
        detected_breakpoints={c: r["breakpoints"] for c, r in _results.items()},
        estimated_slopes={c: r["slopes"] for c, r in _results.items()},
    )
    show(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Method 4 — ruptures + OLS
    *Library:* `ruptures` + `scipy` &nbsp;|&nbsp; *Less accurate (L2 cost)*
    """)
    return


@app.cell
def _(data, importlib, plot_results, show):
    _m = importlib.import_module("gate_analysis.4_ruptures_ols")
    _results, _segs = _m.analyze(data)
    _fig = plot_results(
        data,
        "Method 4: ruptures + OLS",
        fitted_segments=_segs,
        detected_breakpoints={c: r["breakpoints"] for c, r in _results.items()},
        estimated_slopes={c: r["slopes"] for c, r in _results.items()},
    )
    show(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Method 5 — Savitzky-Golay Derivative
    *Library:* `scipy.signal` &nbsp;|&nbsp; *Heuristic, fast*
    """)
    return


@app.cell
def _(Label, Span, bk_figure, column, data, importlib, show):
    from bokeh.palettes import Category10 as _C10

    _m = importlib.import_module("gate_analysis.5_savitzky_golay")
    _results, _ = _m.analyze(data)
    _colors = _C10[10]
    _time = data.time

    _p1 = bk_figure(
        width=1100,
        height=370,
        title="Method 5 — Savitzky-Golay: signal + smoothed",
        y_axis_label="Gate position (%)",
    )
    for _gi, _col in enumerate(data.gate_columns):
        _c = _colors[_gi % 10]
        _pos = data.df[_col].to_numpy()
        _r = _results[_col]
        _p1.scatter(_time, _pos, marker="circle", color=_c, alpha=0.2, size=2)
        _p1.line(
            _time,
            _r["smoothed"],
            line_color=_c,
            line_width=1.5,
            legend_label=f"{_col} SG smoothed",
        )
        for _bp in _r["breakpoints"]:
            _p1.add_layout(
                Span(
                    location=_bp,
                    dimension="height",
                    line_color=_c,
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
    _info_lines = []
    for _gi, _col in enumerate(data.gate_columns):
        _c = _colors[_gi % 10]
        _r = _results[_col]
        _p2.line(
            _time,
            _r["derivative"],
            line_color=_c,
            line_width=1,
            legend_label=f"{_col} dy/dt",
        )
        for _bp in _r["breakpoints"]:
            _p2.add_layout(
                Span(
                    location=_bp,
                    dimension="height",
                    line_color=_c,
                    line_dash="dashed",
                    line_alpha=0.7,
                )
            )
        for _si, _s in enumerate(_r["slopes"]):
            _info_lines.append(f"{_col} slope {_si + 1}: {_s:.2f} %/s")
    _info_lines.append(f"True: {data.slopes[0]:.1f}, {data.slopes[1]:.1f} %/s")
    _p2.add_layout(
        Span(location=0, dimension="width", line_color="black", line_width=0.5)
    )
    _p2.add_layout(
        Label(
            x=10,
            y=10,
            x_units="screen",
            y_units="screen",
            text="\n".join(_info_lines),
            text_font_size="9pt",
            background_fill_color="wheat",
            background_fill_alpha=0.8,
        )
    )
    _p2.grid.grid_line_alpha = 0.3
    show(column(_p1, _p2))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Method 6 — Kalman Filter
    *Library:* `filterpy` &nbsp;|&nbsp; *Heuristic, online-capable*
    """)
    return


@app.cell
def _(Label, Span, bk_figure, column, data, importlib, show):
    from bokeh.palettes import Category10 as _C10

    _m = importlib.import_module("gate_analysis.6_kalman_filter")
    _results, _ = _m.analyze(data)
    _colors = _C10[10]
    _time = data.time

    _p1 = bk_figure(
        width=1100,
        height=370,
        title="Method 6 — Kalman Filter: position",
        y_axis_label="Gate position (%)",
    )
    for _gi, _col in enumerate(data.gate_columns):
        _c = _colors[_gi % 10]
        _pos = data.df[_col].to_numpy()
        _r = _results[_col]
        _p1.scatter(_time, _pos, marker="circle", color=_c, alpha=0.2, size=2)
        _p1.line(
            _time,
            _r["filtered_position"],
            line_color=_c,
            line_width=1.5,
            legend_label=f"{_col} filtered",
        )
        for _bp in _r["breakpoints"]:
            _p1.add_layout(
                Span(
                    location=_bp,
                    dimension="height",
                    line_color=_c,
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
    _info_lines = []
    for _gi, _col in enumerate(data.gate_columns):
        _c = _colors[_gi % 10]
        _r = _results[_col]
        _p2.line(
            _time,
            _r["filtered_velocity"],
            line_color=_c,
            line_width=1,
            legend_label=_col,
        )
        for _bp in _r["breakpoints"]:
            _p2.add_layout(
                Span(
                    location=_bp,
                    dimension="height",
                    line_color=_c,
                    line_dash="dashed",
                    line_alpha=0.7,
                )
            )
        _info_lines.append(
            f"{_col}: fast={_r['slopes'][0]:.2f}, slow={_r['slopes'][1]:.2f} %/s"
        )
    _info_lines.append(f"True: {data.slopes[0]:.1f}, {data.slopes[1]:.1f} %/s")
    _p2.add_layout(
        Span(location=0, dimension="width", line_color="black", line_width=0.5)
    )
    _p2.add_layout(
        Label(
            x=10,
            y=10,
            x_units="screen",
            y_units="screen",
            text="\n".join(_info_lines),
            text_font_size="9pt",
            background_fill_color="wheat",
            background_fill_alpha=0.8,
        )
    )
    _p2.grid.grid_line_alpha = 0.3
    show(column(_p1, _p2))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Method 8 — Narrowest-Over-Threshold (NOT)
    *Library:* numpy + scipy only (custom reimplementation) &nbsp;|&nbsp; *Accurate, no prior on number of breakpoints, ~1 s*
    """)
    return


@app.cell
def _(data, importlib, plot_results, show):
    _m = importlib.import_module("gate_analysis.8_not_detection")
    _results, _segs = _m.analyze(data)
    _fig = plot_results(
        data,
        "Method 8 — Narrowest-Over-Threshold (NOT)",
        fitted_segments=_segs,
        detected_breakpoints={c: r["breakpoints"] for c, r in _results.items()},
        estimated_slopes={c: r["slopes"] for c, r in _results.items()},
    )
    show(_fig)
    return


if __name__ == "__main__":
    app.run()
