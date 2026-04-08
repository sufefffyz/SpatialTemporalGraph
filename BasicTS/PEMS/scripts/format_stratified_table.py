#!/usr/bin/env python3
"""
Format a stratified-eval CSV into a more readable comparison table.

Features:
- Keep ML / OR / FR / Ramp in one table block
- Preserve horizon rows: overall / h3 / h6 / h12
- Highlight the better value between two runs using Markdown bold
- Add delta columns (run_b - run_a) for quick comparison
"""

from __future__ import annotations

import argparse
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


def bold_better(a: float, b: float, lower_better: bool = True) -> tuple[str, str]:
    if pd.isna(a) or pd.isna(b):
        return fmt_val(a), fmt_val(b)
    if a == b:
        return fmt_val(a), fmt_val(b)
    a_better = a < b if lower_better else a > b
    if a_better:
        return f"**{fmt_val(a)}**", fmt_val(b)
    return fmt_val(a), f"**{fmt_val(b)}**"


def build_readable_table(df: pd.DataFrame, run_order: list[str], include_all: bool) -> pd.DataFrame:
    group_order = GROUP_ORDER if include_all else [g for g in GROUP_ORDER if g != "ALL"]
    df = df[df["group"].isin(group_order)].copy()
    df["group"] = pd.Categorical(df["group"], categories=group_order, ordered=True)
    df["horizon"] = pd.Categorical(df["horizon"], categories=HORIZON_ORDER, ordered=True)
    df = df.sort_values(["group", "horizon", "run"]).reset_index(drop=True)

    if len(run_order) != 2:
        raise ValueError(f"当前脚本只支持 2 个 run 的对比，收到: {run_order}")
    run_a, run_b = run_order

    left = df[df["run"] == run_a].copy()
    right = df[df["run"] == run_b].copy()
    merged = left.merge(
        right,
        on=["group", "horizon", "num_nodes"],
        suffixes=(f"_{run_a}", f"_{run_b}"),
        how="inner",
    )

    rows = []
    for row in merged.itertuples(index=False):
        row_dict = {
            "Group": row.group,
            "Horizon": row.horizon,
            "Nodes": int(row.num_nodes),
        }
        for metric in METRIC_ORDER:
            a = getattr(row, f"{metric}_{run_a}")
            b = getattr(row, f"{metric}_{run_b}")
            a_fmt, b_fmt = bold_better(a, b, lower_better=True)
            row_dict[f"{run_a}_{metric}"] = a_fmt
            row_dict[f"{run_b}_{metric}"] = b_fmt
            row_dict[f"Δ{metric}({run_b}-{run_a})"] = fmt_delta(b - a if pd.notna(a) and pd.notna(b) else float("nan"))
        rows.append(row_dict)

    return pd.DataFrame(rows)


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

    if args.output_prefix:
        prefix = Path(args.output_prefix).expanduser().resolve()
    else:
        prefix = input_csv.with_suffix("")

    out_csv = prefix.with_name(prefix.name + "_readable.csv")
    out_md = prefix.with_name(prefix.name + "_readable.md")
    out_html = prefix.with_name(prefix.name + "_readable.html")

    table_df.to_csv(out_csv, index=False)
    out_md.write_text(table_df.to_markdown(index=False), encoding="utf-8")
    out_html.write_text(table_df.to_html(index=False, escape=False), encoding="utf-8")

    log("")
    log(table_df.to_string(index=False))
    log("")
    log(f"CSV : {out_csv}")
    log(f"MD  : {out_md}")
    log(f"HTML: {out_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
