#!/usr/bin/env python3
"""
AprilTag benchmark analyzer.

Reads JSONL output from apriltag-bench and generates an interactive HTML report
with Plotly charts for performance analysis.

Usage:
    python analyze_benchmark.py results.jsonl [--output report.html]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats


# ── Typed Structures ──────────────────────────────────────────────────


class GroupbyVariable(TypedDict):
    key: str
    label: str
    kind: Literal["discrete", "continuous"]


class SeriesData(TypedDict):
    x: list[float]
    y: list[float]
    err: list[float]


class DiscreteChartData(TypedDict):
    kind: Literal["discrete"]
    categories: list[str]
    wall_means: list[float]
    wall_stds: list[float]
    stage_names: list[str]
    stage_data: dict[str, list[float]]
    stage_colors: list[str]


class ContinuousChartData(TypedDict):
    kind: Literal["continuous"]
    series: dict[str, SeriesData]


ChartData = DiscreteChartData | ContinuousChartData


class PairComparison(TypedDict):
    label_a: str
    label_b: str
    table_rows: list[list[str]]
    bar_stages: list[str]
    bar_a: list[float]
    bar_a_err: list[float]
    bar_b: list[float]
    bar_b_err: list[float]


# ── Data Loading ──────────────────────────────────────────────────────


def load_runs(path: str) -> list[dict]:
    """Load benchmark runs from a JSONL file."""
    runs = []
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                runs.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: skipping malformed JSON on line {line_no}: {e}", file=sys.stderr)
    return runs


def _base_label(run: dict, idx: int) -> str:
    """Get a base display label for a run (before disambiguation)."""
    label = run.get("metadata", {}).get("label")
    if label:
        return label
    accel = run.get("config", {}).get("acceleration", "?")
    threads = run.get("config", {}).get("nthreads", "?")
    return f"run-{idx} ({accel}, {threads}t)"


def _disambiguate_labels(runs: list[dict]) -> list[str]:
    """Assign unique labels to runs, appending timestamp when labels collide."""
    base_labels = [_base_label(run, i) for i, run in enumerate(runs)]
    # Count how many times each base label appears
    from collections import Counter
    counts = Counter(base_labels)
    # For duplicates, append a short timestamp to distinguish them
    labels = []
    for base, run in zip(base_labels, runs):
        if counts[base] > 1:
            ts = run.get("metadata", {}).get("timestamp", "")
            # Use date+time portion, drop seconds and 'Z' for brevity
            short_ts = ts.replace("T", " ").rstrip("Z")
            labels.append(f"{base} ({short_ts})")
        else:
            labels.append(base)
    return labels


def flatten_iterations(runs: list[dict]) -> pd.DataFrame:
    """Flatten all iteration data into a single DataFrame for analysis."""
    global _cpu_stage_order, _gpu_stage_order, _cpu_stage_order_cols, _gpu_stage_order_cols, _cputime_stage_order, _cputime_stage_order_cols
    _cpu_stage_order = _extract_stage_order(runs, "stages")
    _cputime_stage_order = _extract_stage_order(runs, "cpu_stages")
    _gpu_stage_order = _extract_stage_order(runs, "gpu_stages")
    _cpu_stage_order_cols = [f"stage_{s}" for s in _cpu_stage_order]
    _cputime_stage_order_cols = [f"cpu_{s}" for s in _cputime_stage_order]
    _gpu_stage_order_cols = [f"gpu_{s}" for s in _gpu_stage_order]
    unique_labels = _disambiguate_labels(runs)
    rows = []
    for run_idx, run in enumerate(runs):
        label = unique_labels[run_idx]
        config = run.get("config", {})
        meta = run.get("metadata", {})
        gpu = meta.get("gpu_device") or {}
        for img in run.get("images", []):
            img_path = img["path"]
            w, h = img.get("width", 0), img.get("height", 0)
            pixels = w * h
            resolution = f"{w}x{h}"
            for iter_idx, it in enumerate(img.get("iterations", [])):
                row = {
                    "run_idx": run_idx,
                    "label": label,
                    "timestamp": meta.get("timestamp", ""),
                    "git_commit": meta.get("git_commit", ""),
                    "acceleration": config.get("acceleration", "unknown"),
                    "nthreads": config.get("nthreads", 1),
                    "allow_concurrency": config.get("allow_concurrency", True),
                    "source_dimensions": config.get("source_dimensions", "dynamic"),
                    "gpu_accelerator": gpu.get("accelerator"),
                    "gpu_name": gpu.get("name"),
                    "gpu_backend": gpu.get("backend"),
                    "gpu_device_type": gpu.get("device_type"),
                    "quad_decimate": config.get("quad_decimate", 1.0),
                    "quad_sigma": config.get("quad_sigma", 0.0),
                    "image": img_path,
                    "width": w,
                    "height": h,
                    "pixels": pixels,
                    "resolution": resolution,
                    "iteration": iter_idx,
                    "wall_time_ms": it.get("wall_time_ms", 0),
                    "nquads": it.get("nquads", 0),
                    "detection_count": len(it.get("detections", [])),
                }
                for stage_name, stage_ms in it.get("stages", {}).items():
                    row[f"stage_{stage_name}"] = stage_ms
                for stage_name, stage_ms in it.get("cpu_stages", {}).items():
                    row[f"cpu_{stage_name}"] = stage_ms
                for stage_name, stage_ms in it.get("gpu_stages", {}).items():
                    row[f"gpu_{stage_name}"] = stage_ms
                rows.append(row)
    return pd.DataFrame(rows)


def _extract_stage_order(runs: list[dict], key: str) -> list[str]:
    """Extract stage names in execution order from the first iteration that has them."""
    for run in runs:
        for img in run.get("images", []):
            for it in img.get("iterations", []):
                stages = it.get(key)
                if stages:
                    return list(stages.keys())
    return []


# Populated by flatten_iterations; preserves execution order from the JSON.
_cpu_stage_order: list[str] = []
_cputime_stage_order: list[str] = []
_gpu_stage_order: list[str] = []
_cpu_stage_order_cols: list[str] = []
_cputime_stage_order_cols: list[str] = []
_gpu_stage_order_cols: list[str] = []


def get_stage_columns(df: pd.DataFrame) -> list[str]:
    """Get CPU stage column names in execution order."""
    if _cpu_stage_order_cols:
        return [c for c in _cpu_stage_order_cols if c in df.columns]
    # Fallback: column order (already insertion-ordered in modern pandas)
    return [c for c in df.columns if c.startswith("stage_")]


def get_cputime_stage_columns(df: pd.DataFrame) -> list[str]:
    """Get CPU (process-time) stage column names in execution order."""
    if _cputime_stage_order_cols:
        return [c for c in _cputime_stage_order_cols if c in df.columns]
    return [c for c in df.columns if c.startswith("cpu_")]


def get_gpu_stage_columns(df: pd.DataFrame) -> list[str]:
    """Get GPU stage column names in execution order."""
    if _gpu_stage_order_cols:
        return [c for c in _gpu_stage_order_cols if c in df.columns]
    return [c for c in df.columns if c.startswith("gpu_") and ' ' in c]


def stage_display_name(col: str) -> str:
    """Convert 'stage_decimate' to 'decimate', 'cpu_foo' to 'foo', or 'gpu_foo' to 'foo'."""
    return col.removeprefix("stage_").removeprefix("cpu_").removeprefix("gpu_")


# ── Chart Builders ────────────────────────────────────────────────────

STAGE_COLORS = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
    "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
]


def build_stage_table_html(df: pd.DataFrame, stage_cols: list[str], title: str) -> str | None:
    """Build a collapsible HTML table of mean, stddev, and % of total for each stage, per run."""
    if not stage_cols:
        return None
    # Only include if there's actual data
    if all(col not in df.columns or df[col].isna().all() for col in stage_cols):
        return None

    labels = []
    # Precompute per-run total (sum of stage means) for percentage calculation
    run_totals: list[float] = []
    for _, group in df.groupby("run_idx"):
        labels.append(group["label"].iloc[0])
        total = sum(
            group[col].dropna().mean()
            for col in stage_cols
            if col in group.columns and len(group[col].dropna()) > 0
        )
        run_totals.append(total)

    header_cells = "".join(
        f'<th colspan="3" style="text-align:center;border-bottom:none">{label}</th>'
        for label in labels
    )
    subheader_cells = "".join(
        '<th style="text-align:right">Mean (ms)</th>'
        '<th style="text-align:right">Stddev</th>'
        '<th style="text-align:right">%</th>'
        for _ in labels
    )

    # Precompute per-run groups (preserve groupby order)
    run_groups = [(group["label"].iloc[0], group) for _, group in df.groupby("run_idx")]

    rows_html = ""
    for col in stage_cols:
        name = stage_display_name(col)
        # Collect means for all runs to find the best (lowest)
        means: list[float | None] = []
        stds: list[float | None] = []
        for _, group in run_groups:
            if col in group.columns:
                vals = group[col].dropna()
                if len(vals) > 0:
                    means.append(float(vals.mean()))
                    stds.append(float(vals.std()))
                else:
                    means.append(None)
                    stds.append(None)
            else:
                means.append(None)
                stds.append(None)

        valid_means = [m for m in means if m is not None]
        best = min(valid_means) if len(valid_means) > 1 else None

        rows_html += f"<tr><td><strong>{name}</strong></td>"
        for run_i, (mean, std) in enumerate(zip(means, stds)):
            if mean is not None:
                pct = mean / run_totals[run_i] * 100 if run_totals[run_i] > 0 else 0
                is_best = best is not None and mean == best
                bold = "font-weight:bold;" if is_best else ""
                rows_html += f'<td style="text-align:right;{bold}">{mean:.3f}</td>'
                rows_html += f'<td style="text-align:right;color:#888">{std:.3f}</td>'
                rows_html += f'<td style="text-align:right">{pct:.1f}%</td>'
            else:
                rows_html += "<td></td><td></td><td></td>"
        rows_html += "</tr>"

    # Total row
    best_total = min(run_totals) if len(run_totals) > 1 else None
    rows_html += '<tr style="border-top:2px solid #999;font-weight:bold"><td>TOTAL</td>'
    for total in run_totals:
        is_best = best_total is not None and total == best_total
        bold = "font-weight:bold;" if is_best else ""
        rows_html += f'<td style="text-align:right;{bold}">{total:.3f}</td><td></td><td style="text-align:right">100%</td>'
    rows_html += "</tr>"

    return f"""
<details style="margin: 12px 0;">
  <summary style="cursor:pointer; font-weight:bold; font-size:15px; padding: 6px 0;">{title}</summary>
  <table class="comparison-table" style="margin-top:8px;">
    <thead>
      <tr><th></th>{header_cells}</tr>
      <tr><th>Stage</th>{subheader_cells}</tr>
    </thead>
    <tbody>{rows_html}</tbody>
  </table>
</details>
"""


def chart_run_overview(df: pd.DataFrame) -> go.Figure:
    """Summary table of all runs."""
    labels = df.groupby("run_idx").first()["label"]
    summaries = []
    for run_idx, group in df.groupby("run_idx"):
        row = {
            "Label": labels[run_idx],
            "Acceleration": group["acceleration"].iloc[0],
            "Threads": group["nthreads"].iloc[0],
            "Decimate": group["quad_decimate"].iloc[0],
            "Images": group["image"].nunique(),
            "Mean Wall (ms)": f"{group['wall_time_ms'].mean():.2f}",
            "Stddev (ms)": f"{group['wall_time_ms'].std():.2f}",
            "Git": (group["git_commit"].iloc[0] or "")[:8],
        }
        if "gpu_name" in group.columns and pd.notna(group["gpu_name"].iloc[0]):
            row["GPU"] = group["gpu_name"].iloc[0]
        summaries.append(row)
    summary_df = pd.DataFrame(summaries)
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(summary_df.columns), align="left"),
        cells=dict(values=[summary_df[c] for c in summary_df.columns], align="left"),
    )])
    fig.update_layout(title="Run Overview", height=200 + 30 * len(summaries))
    return fig


def chart_stage_breakdown(df: pd.DataFrame) -> go.Figure:
    """Stacked bar chart of mean stage times per run."""
    stage_cols = get_stage_columns(df)
    if not stage_cols:
        return go.Figure().update_layout(title="Stage Breakdown (no stage data)")

    labels = []
    stage_means = {col: [] for col in stage_cols}
    for run_idx, group in df.groupby("run_idx"):
        labels.append(group["label"].iloc[0])
        for col in stage_cols:
            stage_means[col].append(group[col].fillna(0).mean() if col in group.columns else 0)

    fig = go.Figure()
    for i, col in enumerate(stage_cols):
        fig.add_trace(go.Bar(
            name=stage_display_name(col),
            x=labels,
            y=stage_means[col],
            marker_color=STAGE_COLORS[i % len(STAGE_COLORS)],
        ))
    fig.update_layout(
        barmode="stack",
        title="Stage Breakdown (mean ms per run)",
        yaxis_title="Time (ms)",
        xaxis_title="Run",
    )
    return fig


def chart_stage_breakdown_pct(df: pd.DataFrame) -> go.Figure:
    """Stacked bar chart of stage proportions per run."""
    stage_cols = get_stage_columns(df)
    if not stage_cols:
        return go.Figure().update_layout(title="Stage Breakdown % (no stage data)")

    labels = []
    stage_pcts = {col: [] for col in stage_cols}
    for run_idx, group in df.groupby("run_idx"):
        labels.append(group["label"].iloc[0])
        # Use wall_time_ms as the denominator — it's always present and represents true total
        total = group["wall_time_ms"].mean()
        for col in stage_cols:
            col_mean = group[col].fillna(0).mean() if col in group.columns else 0
            stage_pcts[col].append(col_mean / total * 100 if total > 0 else 0)

    fig = go.Figure()
    for i, col in enumerate(stage_cols):
        fig.add_trace(go.Bar(
            name=stage_display_name(col),
            x=labels,
            y=stage_pcts[col],
            marker_color=STAGE_COLORS[i % len(STAGE_COLORS)],
        ))
    fig.update_layout(
        barmode="stack",
        title="Stage Breakdown (% of wall time)",
        yaxis_title="% of Wall Time",
        xaxis_title="Run",
    )
    return fig


def chart_gpu_stage_breakdown(df: pd.DataFrame) -> go.Figure | None:
    """Stacked bar chart of mean GPU stage times per run."""
    gpu_cols = get_gpu_stage_columns(df)
    if not gpu_cols:
        return None
    # Only include if there's actual data (not all NaN)
    if all(df[c].isna().all() for c in gpu_cols):
        return None

    labels = []
    stage_means = {col: [] for col in gpu_cols}
    for run_idx, group in df.groupby("run_idx"):
        # Skip runs that have no GPU data (all GPU columns are NaN)
        if all(group[c].isna().all() for c in gpu_cols if c in group.columns):
            continue
        labels.append(group["label"].iloc[0])
        for col in gpu_cols:
            stage_means[col].append(group[col].mean() if col in group.columns else 0)

    if not labels:
        return None

    fig = go.Figure()
    for i, col in enumerate(gpu_cols):
        fig.add_trace(go.Bar(
            name=stage_display_name(col),
            x=labels,
            y=stage_means[col],
            marker_color=STAGE_COLORS[i % len(STAGE_COLORS)],
        ))
    fig.update_layout(
        barmode="stack",
        title="GPU Stage Breakdown (mean ms per run)",
        yaxis_title="Time (ms)",
        xaxis_title="Run",
    )
    return fig


def _compare_pair(df: pd.DataFrame, idx_a: int, idx_b: int) -> PairComparison:
    """Compute comparison data for a single pair of runs."""
    stage_cols = get_stage_columns(df)
    gpu_cols = get_gpu_stage_columns(df)
    df_a = df[df["run_idx"] == idx_a]
    df_b = df[df["run_idx"] == idx_b]
    label_a = df_a["label"].iloc[0]
    label_b = df_b["label"].iloc[0]

    # Table rows
    table_rows = []
    compare_cols = stage_cols + ["wall_time_ms"] + gpu_cols
    for col in compare_cols:
        if col.startswith("gpu_"):
            name = f"GPU: {stage_display_name(col)}"
        elif col.startswith("stage_"):
            name = stage_display_name(col)
        else:
            name = "TOTAL"
        vals_a = df_a[col].dropna().values if col in df_a.columns else np.array([])
        vals_b = df_b[col].dropna().values if col in df_b.columns else np.array([])
        if len(vals_a) == 0 or len(vals_b) == 0:
            continue
        mean_a, std_a = float(vals_a.mean()), float(vals_a.std())
        mean_b, std_b = float(vals_b.mean()), float(vals_b.std())
        diff_pct = (mean_b - mean_a) / mean_a * 100 if mean_a != 0 else 0
        try:
            _, p_value = stats.ttest_ind(vals_a, vals_b, equal_var=False)
            p_value = float(p_value)
        except Exception:
            p_value = 1.0
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        table_rows.append([name, f"{mean_a:.2f} ± {std_a:.2f}", f"{mean_b:.2f} ± {std_b:.2f}",
                           f"{diff_pct:+.1f}%", f"{p_value:.4f}" if p_value >= 0.001 else "<0.001", sig])

    # Bar chart data (CPU stages only)
    bar_stages = [stage_display_name(c) for c in stage_cols]
    bar_a = [float(df_a[c].fillna(0).mean()) for c in stage_cols]
    bar_a_err = [float(df_a[c].fillna(0).std()) for c in stage_cols]
    bar_b = [float(df_b[c].fillna(0).mean()) for c in stage_cols]
    bar_b_err = [float(df_b[c].fillna(0).std()) for c in stage_cols]

    return {
        "label_a": label_a,
        "label_b": label_b,
        "table_rows": table_rows,
        "bar_stages": bar_stages,
        "bar_a": bar_a, "bar_a_err": bar_a_err,
        "bar_b": bar_b, "bar_b_err": bar_b_err,
    }


def build_pairwise_html(df: pd.DataFrame) -> str | None:
    """Build full pairwise comparison HTML with interactive dropdowns."""
    run_indices = sorted(df["run_idx"].unique())
    if len(run_indices) < 2:
        return None

    # Map run_idx -> label
    run_labels = {}
    for run_idx, group in df.groupby("run_idx"):
        run_labels[int(run_idx)] = group["label"].iloc[0]

    # Pre-compute all pairs
    pairs = {}
    for i, idx_a in enumerate(run_indices):
        for idx_b in run_indices[i + 1:]:
            key = f"{int(idx_a)}_{int(idx_b)}"
            pairs[key] = _compare_pair(df, idx_a, idx_b)

    pairs_json = json.dumps(pairs)
    options_html = "\n".join(
        f'<option value="{int(idx)}">{run_labels[int(idx)]}</option>'
        for idx in run_indices
    )

    default_a = int(run_indices[0])
    default_b = int(run_indices[1])

    return f"""
<div style="margin-bottom: 16px;">
  <label><strong>Run A:</strong>
    <select id="pw-select-a" onchange="updatePairwise()" style="padding: 4px 8px; font-size: 14px;">
      {options_html}
    </select>
  </label>
  &nbsp;&nbsp;
  <label><strong>Run B:</strong>
    <select id="pw-select-b" onchange="updatePairwise()" style="padding: 4px 8px; font-size: 14px;">
      {options_html}
    </select>
  </label>
</div>
<div id="pw-chart" class="chart"></div>
<div id="pw-table"></div>
<p><em>Significance: * p&lt;0.05, ** p&lt;0.01, *** p&lt;0.001</em></p>

<script>
(function() {{
  const pairsData = {pairs_json};
  const selA = document.getElementById('pw-select-a');
  const selB = document.getElementById('pw-select-b');
  selA.value = '{default_a}';
  selB.value = '{default_b}';

  // Disable options in each dropdown that match the other's selection
  function syncDropdowns() {{
    const aVal = selA.value, bVal = selB.value;
    for (const opt of selA.options) opt.disabled = (opt.value === bVal);
    for (const opt of selB.options) opt.disabled = (opt.value === aVal);
  }}

  window.updatePairwise = function() {{
    syncDropdowns();
    const a = parseInt(selA.value), b = parseInt(selB.value);
    const chartEl = document.getElementById('pw-chart');
    const tableEl = document.getElementById('pw-table');
    if (a === b) {{
      Plotly.purge(chartEl);
      chartEl.innerHTML = '<p style="color:#888;">Select two different runs.</p>';
      tableEl.innerHTML = '';
      return;
    }}
    const lo = Math.min(a, b), hi = Math.max(a, b);
    const key = lo + '_' + hi;
    const d = pairsData[key];
    if (!d) {{
      Plotly.purge(chartEl);
      chartEl.innerHTML = '<p style="color:#888;">No data for this pair.</p>';
      tableEl.innerHTML = '';
      return;
    }}
    // Clear any message text before Plotly takes over the div
    if (!chartEl.classList.contains('js-plotly-plot')) {{
      chartEl.innerHTML = '';
    }}
    // Flip labels if user selected in reverse order
    const flipLabels = (a > b);
    const labelA = flipLabels ? d.label_b : d.label_a;
    const labelB = flipLabels ? d.label_a : d.label_b;
    const barA = flipLabels ? d.bar_b : d.bar_a;
    const barAErr = flipLabels ? d.bar_b_err : d.bar_a_err;
    const barB = flipLabels ? d.bar_a : d.bar_b;
    const barBErr = flipLabels ? d.bar_a_err : d.bar_b_err;

    // Bar chart
    Plotly.react('pw-chart', [
      {{name: labelA, x: d.bar_stages, y: barA, error_y: {{type:'data', array: barAErr, visible:true}}, type:'bar'}},
      {{name: labelB, x: d.bar_stages, y: barB, error_y: {{type:'data', array: barBErr, visible:true}}, type:'bar'}}
    ], {{barmode:'group', title:'Stage Comparison: ' + labelA + ' vs ' + labelB, yaxis:{{title:'Time (ms)'}}}});

    // Table
    let html = '<table class="comparison-table"><thead><tr>';
    const cols = ['Stage', labelA + ' (ms)', labelB + ' (ms)', 'Diff %', 'p-value', 'Sig'];
    cols.forEach(c => html += '<th>' + c + '</th>');
    html += '</tr></thead><tbody>';
    const rows = flipLabels ? d.table_rows.map(r => {{
      // Swap columns A and B, recalculate diff sign
      const swapMean = r[2], swapStd = r[1];
      // Diff % needs to be recalculated from the text — easier to just show the pre-computed table
      // Since we stored canonical order, just swap display columns
      return [r[0], r[2], r[1], r[3].startsWith('+') ? r[3].replace('+', '-') : r[3].replace('-', '+'), r[4], r[5]];
    }}) : d.table_rows;
    rows.forEach(r => {{
      html += '<tr>';
      r.forEach(c => html += '<td>' + c + '</td>');
      html += '</tr>';
    }});
    html += '</tbody></table>';
    document.getElementById('pw-table').innerHTML = html;
  }};

  // Initial render
  updatePairwise();
}})();
</script>
"""


def chart_timing_histograms(df: pd.DataFrame) -> list[go.Figure]:
    """Histogram of wall time and per-stage distributions, overlaid by run."""
    figs = []
    stage_cols = get_stage_columns(df)

    # Wall time histogram
    fig = go.Figure()
    for run_idx, group in df.groupby("run_idx"):
        fig.add_trace(go.Histogram(
            x=group["wall_time_ms"],
            name=group["label"].iloc[0],
            opacity=0.6,
            nbinsx=40,
        ))
    fig.update_layout(
        barmode="overlay",
        title="Wall Time Distribution",
        xaxis_title="Wall Time (ms)",
        yaxis_title="Count",
    )
    figs.append(fig)

    # Per-stage histograms (only for stages with meaningful variation)
    for col in stage_cols:
        if col not in df.columns:
            continue
        if df[col].std() < 0.001:
            continue
        fig = go.Figure()
        for run_idx, group in df.groupby("run_idx"):
            if col in group.columns:
                fig.add_trace(go.Histogram(
                    x=group[col].dropna(),
                    name=group["label"].iloc[0],
                    opacity=0.6,
                    nbinsx=30,
                ))
        fig.update_layout(
            barmode="overlay",
            title=f"Distribution: {stage_display_name(col)}",
            xaxis_title="Time (ms)",
            yaxis_title="Count",
        )
        figs.append(fig)

    return figs


def _get_groupby_variables(df: pd.DataFrame) -> list[GroupbyVariable]:
    """Discover all available variables for grouping/comparing runs.

    A variable is included only if it has >1 distinct value across the data.
    """
    candidates = [
        # Config parameters
        ("acceleration",      "Backend",             None),
        ("nthreads",          "Threads",             None),
        ("quad_decimate",     "Quad Decimate",       None),
        ("quad_sigma",        "Quad Sigma",          None),
        ("allow_concurrency", "Allow Concurrency",   "discrete"),
        ("source_dimensions", "Source Dimensions",   "discrete"),
        ("gpu_accelerator",   "GPU Accelerator",     "discrete"),
        ("gpu_name",          "GPU Device",          "discrete"),
        ("gpu_backend",       "GPU Backend",         "discrete"),
        # Image properties
        ("resolution",        "Image Resolution",    "discrete"),
        ("pixels",            "Image Pixels",        "continuous"),
        ("image",             "Image Path",          "discrete"),
        # Run metadata
        ("label",             "Run Label",           "discrete"),
        ("run_idx",           "Run Order",           "continuous"),
        ("git_commit",        "Git Commit",          "discrete"),
    ]
    variables = []
    for key, label, force_kind in candidates:
        if key not in df.columns:
            continue
        col = df[key].dropna()
        nunique = col.nunique()
        if nunique < 2:
            continue
        if force_kind:
            kind = force_kind
        elif pd.api.types.is_numeric_dtype(col) and nunique > 5:
            kind = "continuous"
        else:
            kind = "discrete"
        variables.append({"key": key, "label": label, "kind": kind})
    return variables


def _build_variable_chart_data(df: pd.DataFrame, var: GroupbyVariable) -> ChartData:
    """Build chart data for a single groupby variable.

    For discrete: grouped bar chart (mean wall time per category, with stage breakdown).
    For continuous: line chart with error bands (mean wall time vs parameter value).
    """
    key = var["key"]
    stage_cols = get_stage_columns(df)

    if var["kind"] == "discrete":
        categories = sorted(df[key].dropna().unique(), key=str)
        # Wall time bars
        cat_labels = [str(c) for c in categories]
        wall_means = []
        wall_stds = []
        # Stage breakdown
        stage_data = {col: [] for col in stage_cols}
        for cat in categories:
            group = df[df[key] == cat]
            wall_means.append(float(group["wall_time_ms"].mean()))
            wall_stds.append(float(group["wall_time_ms"].std()))
            for col in stage_cols:
                stage_data[col].append(float(group[col].fillna(0).mean()) if col in group.columns else 0)

        return {
            "kind": "discrete",
            "categories": cat_labels,
            "wall_means": wall_means,
            "wall_stds": wall_stds,
            "stage_names": [stage_display_name(c) for c in stage_cols],
            "stage_data": {stage_display_name(c): stage_data[c] for c in stage_cols},
            "stage_colors": STAGE_COLORS[:len(stage_cols)],
        }
    else:
        # Continuous: group by the variable, split series by a secondary grouper
        # Pick the best secondary grouper: use run label
        series = {}
        for run_idx, group in df.groupby("run_idx"):
            label = group["label"].iloc[0]
            agg = group.groupby(key)["wall_time_ms"].agg(["mean", "std"]).reset_index()
            agg = agg.sort_values(key)
            series[label] = {
                "x": [float(v) for v in agg[key]],
                "y": [float(v) for v in agg["mean"]],
                "err": [float(v) if not np.isnan(v) else 0 for v in agg["std"]],
            }
        return {
            "kind": "continuous",
            "series": series,
        }


def build_compare_by_variable_html(df: pd.DataFrame) -> str | None:
    """Build unified 'Compare' by variable tab with a dropdown to select any variable."""
    variables = _get_groupby_variables(df)
    if not variables:
        return None

    # Pre-compute chart data for every variable
    var_data = {}
    for var in variables:
        var_data[var["key"]] = _build_variable_chart_data(df, var)

    var_data_json = json.dumps(var_data)
    var_meta_json = json.dumps(variables)

    options_html = "\n".join(
        f'<option value="{v["key"]}">{v["label"]} ({v["kind"]})</option>'
        for v in variables
    )
    default_key = variables[0]["key"]

    # Secondary Y-axis toggle: wall time vs stage breakdown (for discrete)
    return f"""
<div style="margin-bottom: 16px; display: flex; gap: 16px; align-items: center; flex-wrap: wrap;">
  <label><strong>Group by:</strong>
    <select id="cv-select" onchange="updateCompareVar()" style="padding: 4px 8px; font-size: 14px;">
      {options_html}
    </select>
  </label>
  <label id="cv-mode-label" style="display:none;">
    <strong>View:</strong>
    <select id="cv-mode" onchange="updateCompareVar()" style="padding: 4px 8px; font-size: 14px;">
      <option value="wall">Wall Time</option>
      <option value="stages">Stage Breakdown</option>
    </select>
  </label>
</div>
<div id="cv-chart" class="chart"></div>

<script>
(function() {{
  const varData = {var_data_json};
  const varMeta = {var_meta_json};
  const STAGE_COLORS = {json.dumps(STAGE_COLORS)};
  const sel = document.getElementById('cv-select');
  const modeSel = document.getElementById('cv-mode');
  const modeLabel = document.getElementById('cv-mode-label');
  sel.value = '{default_key}';

  window.updateCompareVar = function() {{
    const key = sel.value;
    const meta = varMeta.find(v => v.key === key);
    const d = varData[key];
    if (!d) return;

    const chartEl = document.getElementById('cv-chart');
    if (!chartEl.classList.contains('js-plotly-plot')) chartEl.innerHTML = '';

    if (d.kind === 'discrete') {{
      modeLabel.style.display = '';
      const mode = modeSel.value;
      if (mode === 'stages' && d.stage_names.length > 0) {{
        // Stacked stage breakdown
        const traces = d.stage_names.map((name, i) => ({{
          name: name,
          x: d.categories,
          y: d.stage_data[name],
          type: 'bar',
          marker: {{color: STAGE_COLORS[i % STAGE_COLORS.length]}},
        }}));
        Plotly.react(chartEl, traces, {{
          barmode: 'stack',
          title: 'Stage Breakdown by ' + meta.label,
          yaxis: {{title: 'Time (ms)'}},
          xaxis: {{title: meta.label}},
        }});
      }} else {{
        // Wall time grouped bar
        Plotly.react(chartEl, [{{
          x: d.categories,
          y: d.wall_means,
          error_y: {{type: 'data', array: d.wall_stds, visible: true}},
          type: 'bar',
          name: 'Wall Time',
        }}], {{
          title: 'Wall Time by ' + meta.label,
          yaxis: {{title: 'Mean Wall Time (ms)'}},
          xaxis: {{title: meta.label}},
        }});
      }}
    }} else {{
      modeLabel.style.display = 'none';
      // Continuous: line chart with one series per run
      const traces = Object.entries(d.series).map(([label, s]) => ({{
        x: s.x,
        y: s.y,
        error_y: {{type: 'data', array: s.err, visible: true}},
        mode: 'lines+markers',
        name: label,
      }}));
      Plotly.react(chartEl, traces, {{
        title: 'Wall Time by ' + meta.label,
        xaxis: {{title: meta.label}},
        yaxis: {{title: 'Mean Wall Time (ms)'}},
      }});
    }}
  }};

  updateCompareVar();
}})();
</script>
"""


# ── HTML Report Generator ────────────────────────────────────────────

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>AprilTag Benchmark Report</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         max-width: 1400px; margin: 0 auto; padding: 20px; background: #fafafa; }}
  h1 {{ color: #333; border-bottom: 2px solid #ddd; padding-bottom: 10px; }}
  h2 {{ color: #555; margin-top: 40px; }}
  .chart {{ background: white; border-radius: 8px; padding: 10px; margin: 20px 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .tabs {{ display: flex; gap: 4px; margin-top: 20px; flex-wrap: wrap; }}
  .tab {{ padding: 8px 16px; background: #e0e0e0; border: none; border-radius: 6px 6px 0 0;
          cursor: pointer; font-size: 14px; }}
  .tab.active {{ background: white; font-weight: bold; }}
  .tab-content {{ display: none; background: white; border-radius: 0 8px 8px 8px;
                  padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .tab-content.active {{ display: block; }}
  .comparison-table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
  .comparison-table th, .comparison-table td {{
    border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
  .comparison-table th {{ background: #f5f5f5; }}
  .comparison-table tr:nth-child(even) {{ background: #fafafa; }}
  .meta {{ color: #888; font-size: 13px; }}
</style>
</head>
<body>
<h1>AprilTag Benchmark Report</h1>
<p class="meta">Generated from {input_file} &mdash; {run_count} run(s)</p>

<div class="tabs">
{tab_headers}
</div>

{tab_contents}

<script>
function showTab(tabId) {{
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
  document.getElementById(tabId).classList.add('active');
  document.querySelector('[onclick*="' + tabId + '"]').classList.add('active');
  // Trigger Plotly resize for any charts in the newly visible tab
  const tabEl = document.getElementById(tabId);
  tabEl.querySelectorAll('.js-plotly-plot').forEach(el => Plotly.Plots.resize(el));
}}
// Show first tab
document.addEventListener('DOMContentLoaded', () => {{
  const firstTab = document.querySelector('.tab');
  if (firstTab) firstTab.click();
}});
</script>
</body>
</html>
"""


def fig_to_div(fig: go.Figure, div_id: str | None = None) -> str:
    """Convert a Plotly figure to an HTML div."""
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)


def generate_report(df: pd.DataFrame, input_file: str) -> str:
    """Generate the full HTML report."""
    tabs = []
    tab_id = 0

    def add_tab(title: str, content_html: str):
        nonlocal tab_id
        tid = f"tab-{tab_id}"
        tabs.append((tid, title, content_html))
        tab_id += 1

    # 1. Overview
    overview_fig = chart_run_overview(df)
    add_tab("Overview", f'<div class="chart">{fig_to_div(overview_fig)}</div>')

    # 2. Stage Breakdown
    breakdown_abs = chart_stage_breakdown(df)
    breakdown_pct = chart_stage_breakdown_pct(df)
    gpu_breakdown = chart_gpu_stage_breakdown(df)
    wall_table = build_stage_table_html(df, get_stage_columns(df), "Wall Time Stages (detail)")
    cpu_table = build_stage_table_html(df, get_cputime_stage_columns(df), "CPU Time Stages (detail)")
    gpu_table = build_stage_table_html(df, get_gpu_stage_columns(df), "GPU Time Stages (detail)")
    breakdown_html = f'<div class="chart">{fig_to_div(breakdown_abs)}</div>'
    if wall_table:
        breakdown_html += wall_table
    if cpu_table:
        breakdown_html += cpu_table
    breakdown_html += f'<div class="chart">{fig_to_div(breakdown_pct)}</div>'
    if gpu_breakdown:
        breakdown_html += f'<div class="chart">{fig_to_div(gpu_breakdown)}</div>'
        if gpu_table:
            breakdown_html += gpu_table
    add_tab("Stage Breakdown", breakdown_html)

    # 3. Pairwise Comparison
    pairwise_html = build_pairwise_html(df)
    if pairwise_html:
        add_tab("Pairwise Comparison", pairwise_html)

    # 4. Timing Distributions
    hist_figs = chart_timing_histograms(df)
    if hist_figs:
        hist_html = "".join(f'<div class="chart">{fig_to_div(f)}</div>' for f in hist_figs)
        add_tab("Distributions", hist_html)

    # 5. Compare by variable (unified: image size, backend, parameters, history)
    compare_html = build_compare_by_variable_html(df)
    if compare_html:
        add_tab("Compare", compare_html)

    # Build HTML
    tab_headers = "\n".join(
        f'<button class="tab{" active" if i == 0 else ""}" onclick="showTab(\'{tid}\')">{title}</button>'
        for i, (tid, title, _) in enumerate(tabs)
    )
    tab_contents = "\n".join(
        f'<div id="{tid}" class="tab-content{" active" if i == 0 else ""}">{html}</div>'
        for i, (tid, _, html) in enumerate(tabs)
    )

    return HTML_TEMPLATE.format(
        input_file=input_file,
        run_count=df["run_idx"].nunique(),
        tab_headers=tab_headers,
        tab_contents=tab_contents,
    )


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AprilTag benchmark analyzer")
    parser.add_argument("input", nargs="+", help="Path(s) to JSONL benchmark results file(s)")
    parser.add_argument("-o", "--output", default="benchmark_report.html",
                        help="Output HTML report path (default: benchmark_report.html)")
    parser.add_argument("--filter-label", help="Only include runs matching this label prefix")
    args = parser.parse_args()

    runs = []
    for path in args.input:
        file_runs = load_runs(path)
        print(f"Loaded {len(file_runs)} run(s) from {path}")
        runs.extend(file_runs)

    if not runs:
        print("No runs found in input file(s).", file=sys.stderr)
        sys.exit(1)

    print(f"Total: {len(runs)} run(s) from {len(args.input)} file(s)")

    df = flatten_iterations(runs)
    if df.empty:
        print("No iteration data found.", file=sys.stderr)
        sys.exit(1)

    if args.filter_label:
        df = df[df["label"].str.startswith(args.filter_label)]
        print(f"Filtered to {df['run_idx'].nunique()} run(s) matching '{args.filter_label}'")

    print(f"Total iterations: {len(df)}")
    print(f"Unique images: {df['image'].nunique()}")
    print(f"Stage columns: {[stage_display_name(c) for c in get_stage_columns(df)]}")

    input_label = ", ".join(args.input) if len(args.input) <= 3 else f"{len(args.input)} files"
    html = generate_report(df, input_label)
    Path(args.output).write_text(html)
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
