#!/usr/bin/env python3
"""
Format a stratified-eval CSV into a more readable comparison table.

Features:
- Keep ML / OR / FR / Ramp in one table block
- Preserve horizon rows: overall / h3 / h6 / h12
- Generate HTML with merged group cells and metric-group headers
- Highlight the best value among methods
- Export flat CSV / Markdown as fallback
"""

from __future__ import annotations

import argparse
import html
from pathlib import Path

import pandas as pd


GROUP_ORDER = ["ML", "OR", "FR", "Ramp", "ALL"]
HORIZON_ORDER = ["overall", "h3", "h6", "h12"]
METRIC_ORDER = ["MAE", "RMSE", "MAPE", "WAPE"]


def log(message: str) -> None:
    print(message, flush=True)


def resolve_existing_path(path_str: str, desc: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{desc} 不存在: {path}")
    return path


def fmt_val(val: float) -> str:
    return f"{val:.4f}" if pd.notna(val) else "NA"


def fmt_delta(delta: float) -> str:
    if pd.isna(delta):
        return "NA"
    if delta > 0:
        return f"+{delta:.4f}"
    return f"{delta:.4f}"


def highlight_best(values: list[float], lower_better: bool = True, markdown: bool = True) -> list[str]:
    formatted = [fmt_val(v) for v in values]
    valid = [(idx, v) for idx, v in enumerate(values) if pd.notna(v)]
    if not valid:
        return formatted
    best_val = min(v for _, v in valid) if lower_better else max(v for _, v in valid)
    best_indices = [idx for idx, v in valid if v == best_val]
    for idx in best_indices:
        formatted[idx] = f"**{formatted[idx]}**" if markdown else f"<strong>{formatted[idx]}</strong>"
    return formatted


def build_readable_table(df: pd.DataFrame, run_order: list[str], include_all: bool) -> pd.DataFrame:
    group_order = GROUP_ORDER if include_all else [g for g in GROUP_ORDER if g != "ALL"]
    df = df[df["group"].isin(group_order)].copy()
    df["group"] = pd.Categorical(df["group"], categories=group_order, ordered=True)
    df["horizon"] = pd.Categorical(df["horizon"], categories=HORIZON_ORDER, ordered=True)
    df = df.sort_values(["group", "horizon", "run"]).reset_index(drop=True)

    index_cols = ["group", "horizon", "num_nodes"]
    wide = df.pivot_table(index=index_cols, columns="run", values=METRIC_ORDER, aggfunc="first")
    wide = wide.swaplevel(0, 1, axis=1)
    desired_cols = pd.MultiIndex.from_product([run_order, METRIC_ORDER])
    wide = wide.reindex(columns=desired_cols)
    wide = wide.reset_index()

    has_two_runs = len(run_order) == 2
    rows = []
    for _, row in wide.iterrows():
        row_dict = {
            "Group": row["group"],
            "Horizon": row["horizon"],
            "Nodes": int(row["num_nodes"]),
        }
        for metric in METRIC_ORDER:
            values = [row[(run, metric)] for run in run_order]
            highlighted = highlight_best(values, lower_better=True, markdown=True)
            for run, val in zip(run_order, highlighted):
                row_dict[f"{run}_{metric}"] = val
            if has_two_runs:
                a = values[0]
                b = values[1]
                row_dict[f"Δ{metric}({run_order[1]}-{run_order[0]})"] = fmt_delta(
                    b - a if pd.notna(a) and pd.notna(b) else float("nan")
                )
        rows.append(row_dict)

    return pd.DataFrame(rows)


def build_html_table(df: pd.DataFrame, run_order: list[str], include_all: bool) -> str:
    group_order = GROUP_ORDER if include_all else [g for g in GROUP_ORDER if g != "ALL"]
    df = df[df["group"].isin(group_order)].copy()
    df["group"] = pd.Categorical(df["group"], categories=group_order, ordered=True)
    df["horizon"] = pd.Categorical(df["horizon"], categories=HORIZON_ORDER, ordered=True)
    df = df.sort_values(["group", "horizon", "run"]).reset_index(drop=True)

    idx_cols = ["group", "horizon", "num_nodes"]
    wide = df.pivot_table(index=idx_cols, columns="run", values=METRIC_ORDER, aggfunc="first")
    wide = wide.swaplevel(0, 1, axis=1)
    desired_cols = pd.MultiIndex.from_product([run_order, METRIC_ORDER])
    wide = wide.reindex(columns=desired_cols)
    wide = wide.reset_index().sort_values(["group", "horizon"]).reset_index(drop=True)

    rowspans = wide.groupby("group").size().to_dict()
    html_parts = [
        "<html><head><meta charset='utf-8'>",
        "<style>",
        "table {border-collapse: collapse; font-family: Arial, sans-serif; font-size: 13px;}",
        "th, td {border: 1px solid #999; padding: 6px 8px; text-align: center;}",
        "th.metric {background: #efefef;}",
        "th.subhead {background: #f8f8f8;}",
        "td.group {font-weight: 700; background: #fafafa;}",
        "td.horizon {font-weight: 600;}",
        "strong {color: #0b7285;}",
        "</style></head><body>",
        "<table>",
        "<thead>",
        "<tr>",
        "<th rowspan='2'>Group</th>",
        "<th rowspan='2'>Horizon</th>",
        "<th rowspan='2'>Nodes</th>",
    ]
    for metric in METRIC_ORDER:
        html_parts.append(f"<th class='metric' colspan='{len(run_order)}'>{html.escape(metric)}</th>")
    html_parts.extend(["</tr>", "<tr>"])
    for _metric in METRIC_ORDER:
        for run in run_order:
            html_parts.append(f"<th class='subhead'>{html.escape(run)}</th>")
    html_parts.extend(["</tr>", "</thead>", "<tbody>"])

    seen_groups = set()
    for _, row in wide.iterrows():
        group = row["group"]
        html_parts.append("<tr>")
        if group not in seen_groups:
            html_parts.append(
                f"<td class='group' rowspan='{rowspans[group]}'>{html.escape(str(group))}</td>"
            )
            seen_groups.add(group)
        html_parts.append(f"<td class='horizon'>{html.escape(str(row['horizon']))}</td>")
        html_parts.append(f"<td>{int(row['num_nodes'])}</td>")
        for metric in METRIC_ORDER:
            values = [row[(run, metric)] for run in run_order]
            highlighted = highlight_best(values, lower_better=True, markdown=False)
            for val in highlighted:
                html_parts.append(f"<td>{val}</td>")
        html_parts.append("</tr>")

    html_parts.extend(["</tbody>", "</table>", "</body></html>"])
    return "\n".join(html_parts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Format stratified evaluation CSV into a readable Markdown table.")
    parser.add_argument("--input-csv", required=True, help="Input stratified_eval CSV path")
    parser.add_argument("--run-order", nargs=2, required=True, help="Two run names in display order, e.g. physical adaptive")
    parser.add_argument("--include-all", action="store_true", help="Also include ALL group")
    parser.add_argument("--output-prefix", default="", help="Output prefix; default is based on input filename")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_csv = resolve_existing_path(args.input_csv, "input csv")
    df = pd.read_csv(input_csv)

    table_df = build_readable_table(df, args.run_order, args.include_all)
    html_table = build_html_table(df, args.run_order, args.include_all)

    if args.output_prefix:
        prefix = Path(args.output_prefix).expanduser().resolve()
    else:
        prefix = input_csv.with_suffix("")

    out_csv = prefix.with_name(prefix.name + "_readable.csv")
    out_md = prefix.with_name(prefix.name + "_readable.md")
    out_html = prefix.with_name(prefix.name + "_readable.html")

    table_df.to_csv(out_csv, index=False)
    out_md.write_text(table_df.to_markdown(index=False), encoding="utf-8")
    out_html.write_text(html_table, encoding="utf-8")

    log("")
    log(table_df.to_string(index=False))
    log("")
    log(f"CSV : {out_csv}")
    log(f"MD  : {out_md}")
    log(f"HTML: {out_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
