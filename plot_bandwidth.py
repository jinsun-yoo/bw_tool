#!/usr/bin/env python3
"""Plot bandwidth monitor CSV files (bw_watch.csv style).

Features:
- X axis: time since beginning (seconds)
- Y axis: rcv_gbps, xmit_gbps, or both (default both)
- Optional start/end filtering by absolute timestamp_ns or since-start seconds
- Output defaults to the input CSV directory unless overridden
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shlex
import sys
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot bw_watch CSV bandwidth over time")
    parser.add_argument("csv", type=Path, help="Input CSV path")
    parser.add_argument(
        "--series",
        choices=["both", "rcv", "xmit"],
        default="both",
        help="Which gbps series to plot (default: both)",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=None,
        help="Start bound value (interpreted by --range-mode)",
    )
    parser.add_argument(
        "--end",
        type=float,
        default=None,
        help="End bound value (interpreted by --range-mode)",
    )
    parser.add_argument(
        "--range-mode",
        choices=["since", "absolute", "auto"],
        default="auto",
        help=(
            "Interpret --start/--end as since-start seconds, absolute timestamp_ns, "
            "or auto-detect (default: auto)"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output image path or directory. If omitted, writes next to input CSV "
            "as <csv_stem>_bw_plot.png"
        ),
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional plot title",
    )
    parser.add_argument(
        "--ymin",
        type=float,
        default=None,
        help="Optional minimum y-axis value",
    )
    parser.add_argument(
        "--ymax",
        type=float,
        default=None,
        help="Optional maximum y-axis value",
    )
    parser.add_argument(
        "--avg",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Average every N rows before plotting (e.g. 100 for 100 ms windows). "
            "If omitted, each row is plotted as an individual dot."
        ),
    )
    parser.add_argument(
        "--plot-style",
        choices=["line", "scatter"],
        default="line",
        help="Draw series as a line plot or scatter plot (default: line)",
    )
    return parser.parse_args()


def resolve_output_path(csv_path: Path, output_arg: Optional[Path]) -> Path:
    default_name = f"{csv_path.stem}_bw_plot.png"
    if output_arg is None:
        return csv_path.parent / default_name

    if output_arg.exists() and output_arg.is_dir():
        return output_arg / default_name

    if output_arg.suffix:
        return output_arg

    # Treat non-existing path without suffix as directory.
    return output_arg / default_name


def output_pair(base_output: Path) -> Tuple[Path, Path]:
    # Always emit both PNG and PDF with the same basename.
    stem_path = base_output.with_suffix("") if base_output.suffix else base_output
    return stem_path.with_suffix(".png"), stem_path.with_suffix(".pdf")


def choose_bound_mode(start: Optional[float], end: Optional[float], mode: str) -> str:
    if mode != "auto":
        return mode

    # Heuristic: timestamp_ns values are around 1e18, while since-seconds are tiny.
    candidates = [v for v in (start, end) if v is not None]
    if not candidates:
        return "since"
    if any(v > 1e12 for v in candidates):
        return "absolute"
    return "since"


def compute_time_axes(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    if "timestamp_ns" not in df.columns:
        raise ValueError("CSV must include 'timestamp_ns' column")

    timestamp_ns = pd.to_numeric(df["timestamp_ns"], errors="raise")

    if "elapsed_ns" in df.columns:
        elapsed_ns = pd.to_numeric(df["elapsed_ns"], errors="coerce")
        if elapsed_ns.notna().all():
            since_s = elapsed_ns / 1e9
        else:
            since_s = (timestamp_ns - timestamp_ns.iloc[0]) / 1e9
    else:
        since_s = (timestamp_ns - timestamp_ns.iloc[0]) / 1e9

    return timestamp_ns, since_s


def apply_bounds(
    df: pd.DataFrame,
    timestamp_ns: pd.Series,
    since_s: pd.Series,
    start: Optional[float],
    end: Optional[float],
    mode: str,
) -> pd.DataFrame:
    filtered = df.copy()

    if start is None and end is None:
        return filtered

    if mode == "absolute":
        axis = timestamp_ns
    elif mode == "since":
        axis = since_s
    else:
        raise ValueError(f"Unsupported range mode: {mode}")

    mask = pd.Series(True, index=df.index)
    if start is not None:
        mask &= axis >= start
    if end is not None:
        mask &= axis <= end

    return filtered.loc[mask]


def compute_md5(file_path: Path) -> str:
    digest = hashlib.md5()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def embed_pdf_metadata(pdf_path: Path, metadata: dict[str, str]) -> None:
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError as exc:
        raise RuntimeError(
            "Embedding custom PDF metadata requires pypdf in the active Python environment"
        ) from exc

    reader = PdfReader(str(pdf_path))
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)

    existing_metadata = {}
    if reader.metadata is not None:
        existing_metadata = {str(key): str(value) for key, value in reader.metadata.items()}

    custom_metadata = {f"/{key}": value for key, value in metadata.items()}
    writer.add_metadata({**existing_metadata, **custom_metadata})

    with pdf_path.open("wb") as handle:
        writer.write(handle)


def main() -> None:
    args = parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)

    required_cols = {"rcv_gbps", "xmit_gbps", "timestamp_ns"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    mode = choose_bound_mode(args.start, args.end, args.range_mode)

    timestamp_ns, since_s = compute_time_axes(df)
    df = apply_bounds(df, timestamp_ns, since_s, args.start, args.end, mode)

    if df.empty:
        raise ValueError("No rows remain after applying time bounds")

    # Recompute since-start based on original timeline for stable x-axis meaning.
    since_s = (pd.to_numeric(df["timestamp_ns"], errors="raise") - timestamp_ns.iloc[0]) / 1e9

    # Optional block-averaging.
    if args.avg is not None and args.avg > 1:
        n = args.avg
        df = df.copy()
        df["_since_s"] = since_s.values
        # Group rows into blocks of n and average numeric columns.
        block_ids = df.index // n - df.index[0] // n
        df_avg = df.groupby(block_ids)[["_since_s", "rcv_gbps", "xmit_gbps"]].mean()
        since_s = df_avg["_since_s"]
        df = df_avg

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=140)

    plot_fn = ax.plot if args.plot_style == "line" else ax.scatter
    point_kwargs = {"s": 7, "alpha": 0.8}
    line_kwargs = {"alpha": 0.9, "linewidth": 1.5}

    if args.series in ("both", "xmit"):
        plot_fn(
            since_s,
            pd.to_numeric(df["xmit_gbps"], errors="coerce"),
            label="xmit_gbps",
            **(line_kwargs if args.plot_style == "line" else point_kwargs),
        )
    if args.series in ("both", "rcv"):
        plot_fn(
            since_s,
            pd.to_numeric(df["rcv_gbps"], errors="coerce"),
            label="rcv_gbps",
            **(line_kwargs if args.plot_style == "line" else point_kwargs),
        )

    ax.set_xlabel("Time Since Beginning (s)")
    ax.set_ylabel("Gbps")
    if args.ymin is not None or args.ymax is not None:
        ax.set_ylim(bottom=args.ymin, top=args.ymax)
    ax.grid(True, alpha=0.25)
    ax.legend()
    if args.title:
        ax.set_title(args.title)

    output_path = resolve_output_path(args.csv, args.output)
    png_path, pdf_path = output_pair(output_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    cmdline = " ".join(shlex.quote(arg) for arg in sys.argv)
    pdf_metadata = {
        "cmdline": cmdline,
        "csvfile hash": compute_md5(args.csv),
    }

    fig.tight_layout()
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    embed_pdf_metadata(pdf_path, pdf_metadata)

    print(f"Saved plot to: {png_path}")
    print(f"Saved plot to: {pdf_path}")
    print(f"Rows plotted: {len(df)}")
    print(f"Range mode: {mode}")


if __name__ == "__main__":
    main()
