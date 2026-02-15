"""Generate report figures from computed metrics."""

from __future__ import annotations

import argparse
import struct
import zlib
from pathlib import Path

from src.utils.io import ensure_dir, read_csv


def _to_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _simple_png_bars(path: str, values: list[float], colors: list[tuple[int, int, int]]) -> None:
    """Minimal PNG bar chart fallback without external plotting libs."""
    width, height = 640, 400
    image = [[(245, 245, 245) for _ in range(width)] for _ in range(height)]

    # axis
    for x in range(50, width - 30):
        image[height - 40][x] = (40, 40, 40)
    for y in range(20, height - 39):
        image[y][50] = (40, 40, 40)

    bar_width = 80
    gap = 40
    start_x = 90
    max_h = height - 80
    max_v = max(max(values), 1e-6)
    for idx, value in enumerate(values):
        h = int((value / max_v) * max_h)
        x0 = start_x + idx * (bar_width + gap)
        x1 = min(x0 + bar_width, width - 31)
        y0 = height - 41 - h
        y1 = height - 41
        color = colors[idx % len(colors)]
        for y in range(max(20, y0), y1):
            for x in range(x0, x1):
                image[y][x] = color

    _write_png(path, image)


def _write_png(path: str, image: list[list[tuple[int, int, int]]]) -> None:
    """Write RGB image matrix as PNG."""
    height = len(image)
    width = len(image[0]) if height else 0
    raw = b"".join(
        b"\x00" + b"".join(bytes((r, g, b)) for (r, g, b) in row) for row in image
    )
    compressed = zlib.compress(raw)

    def chunk(chunk_type: bytes, data: bytes) -> bytes:
        return (
            struct.pack("!I", len(data))
            + chunk_type
            + data
            + struct.pack("!I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
        )

    png = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png += chunk(b"IHDR", ihdr)
    png += chunk(b"IDAT", compressed)
    png += chunk(b"IEND", b"")
    Path(path).write_bytes(png)


def _plot_with_matplotlib(
    metrics_summary: list[dict[str, str]],
    error_rows: list[dict[str, str]],
    output_dir: Path,
) -> bool:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return False

    by_variant = {row["model_variant"]: row for row in metrics_summary}
    variants = ["baseline", "enhanced"]
    semantic = [_to_float(by_variant.get(v, {}).get("semantic_score_mean", "0")) for v in variants]
    safety = [1 - _to_float(by_variant.get(v, {}).get("unsafe_advice_rate", "0")) for v in variants]
    explain = [_to_float(by_variant.get(v, {}).get("citation_sufficiency_rate", "0")) for v in variants]

    x = range(len(variants))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([i - 0.22 for i in x], semantic, width=0.22, label="Semantic", color="#2c7fb8")
    ax.bar(x, safety, width=0.22, label="Safety", color="#41ab5d")
    ax.bar([i + 0.22 for i in x], explain, width=0.22, label="Citation", color="#fdae6b")
    ax.set_xticks(list(x), variants)
    ax.set_ylim(0, 1.05)
    ax.set_title("Baseline vs Enhanced (normalized metrics)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "baseline_vs_enhanced.png", dpi=140)
    plt.close(fig)

    if error_rows:
        labels = [r["error_type"] for r in error_rows]
        counts = [int(r["count"]) for r in error_rows]
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.bar(labels, counts, color="#6baed6")
        ax2.set_title("Error Distribution")
        ax2.tick_params(axis="x", rotation=35)
        fig2.tight_layout()
        fig2.savefig(output_dir / "error_distribution.png", dpi=140)
        plt.close(fig2)
    else:
        _simple_png_bars(
            str(output_dir / "error_distribution.png"),
            [0.0, 0.0, 0.0],
            [(107, 174, 214), (158, 202, 225), (198, 219, 239)],
        )
    return True


def make_figures(tables_dir: str, figures_dir: str) -> tuple[str, str]:
    """Generate required figure files from summary tables."""
    tdir = Path(tables_dir)
    fdir = ensure_dir(figures_dir)

    metrics_summary = read_csv(tdir / "metrics_summary.csv") if (tdir / "metrics_summary.csv").exists() else []
    error_rows: list[dict[str, str]] = []
    for file_name in ["error_counts_enhanced_test.csv", "error_counts_enhanced.csv"]:
        path = tdir / file_name
        if path.exists():
            error_rows = read_csv(path)
            break

    used_mpl = _plot_with_matplotlib(metrics_summary, error_rows, fdir)
    if not used_mpl:
        by_variant = {row["model_variant"]: row for row in metrics_summary}
        values = [
            _to_float(by_variant.get("baseline", {}).get("semantic_score_mean", "0")),
            _to_float(by_variant.get("enhanced", {}).get("semantic_score_mean", "0")),
        ]
        _simple_png_bars(
            str(fdir / "baseline_vs_enhanced.png"),
            values,
            [(44, 127, 184), (65, 171, 93)],
        )
        e_values = [float(r.get("count", "0")) for r in error_rows[:6]] if error_rows else [0.0, 0.0, 0.0]
        _simple_png_bars(
            str(fdir / "error_distribution.png"),
            e_values,
            [(107, 174, 214), (158, 202, 225), (198, 219, 239), (239, 138, 98)],
        )

    return str(fdir / "error_distribution.png"), str(fdir / "baseline_vs_enhanced.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate figures for report.")
    parser.add_argument("--tables_dir", default="reports/tables")
    parser.add_argument("--figures_dir", default="reports/figures")
    args = parser.parse_args()
    error_fig, compare_fig = make_figures(args.tables_dir, args.figures_dir)
    print(f"Wrote: {error_fig}")
    print(f"Wrote: {compare_fig}")


if __name__ == "__main__":
    main()
