"""IO utilities for CSV/JSON and lightweight YAML parsing."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """Create directory if missing and return its `Path`."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_csv(path: str | Path) -> list[dict[str, str]]:
    """Read CSV as list of dictionaries."""
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write list of dictionaries to CSV with fixed column order."""
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def read_json(path: str | Path) -> dict[str, Any]:
    """Read JSON object from file."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write JSON object to file with pretty formatting."""
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def parse_bool(value: Any) -> bool:
    """Parse loose boolean values used in CSV fields."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load YAML with PyYAML if available, otherwise via simple parser."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")

    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        pass

    return _simple_yaml_parse(text)


def _cast_scalar(raw: str) -> Any:
    value = raw.strip()
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _simple_yaml_parse(text: str) -> dict[str, Any]:
    """Very small YAML subset parser for local config files."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue

        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            continue

        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]

        if raw_value == "":
            current[key] = {}
            stack.append((indent, current[key]))
        else:
            current[key] = _cast_scalar(raw_value)

    return root

