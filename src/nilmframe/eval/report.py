"""Results tables.

A sweep produces a wide dataframe -- every metric of every family for every arm.
Printing it raw buries the comparison. These helpers select the columns that carry
the argument and mark which direction is better, because a table where some columns
want to be large and others small is a table people misread.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

__all__ = ["HEADLINE_COLUMNS", "compare", "format_table", "load_results"]

#: Column -> is-higher-better. One per family, so no family can be quietly dropped.
HEADLINE_COLUMNS: dict[str, bool] = {
    "f1_macro": True,
    "mcc": True,
    "mae": False,
    "nde": False,
    "sae": False,
    "teca": True,
    "modified_f1": True,
    "modified_jaccard": True,
}

_IDENTITY = ["name", "view", "align", "rate_hz", "model", "parameters", "seconds"]


def format_table(table: pd.DataFrame, *, precision: int = 4) -> str:
    """Render a sweep result as text, with arrows for metric direction.

    Args:
        table: a sweep result, from :func:`run_sweep` or :func:`load_results`.
        precision: decimal places for the metric columns.

    Returns:
        A text table. Metric names carry ``^`` when larger is better and ``v`` when
        smaller is, because a table mixing both directions is one people misread.

    Example:
        >>> import pandas as pd
        >>> from nilmframe.eval import format_table
        >>> table = pd.DataFrame([{'name': 'a', 'f1_macro': 0.8, 'mae': 10.0}])
        >>> print(format_table(table))
        name  f1_macro ^   mae v
           a      0.8000 10.0000
    """
    identity = [c for c in _IDENTITY if c in table.columns]
    metrics = [c for c in HEADLINE_COLUMNS if c in table.columns]
    view = table[identity + metrics].copy()

    renamed = {c: f"{c} {'^' if HEADLINE_COLUMNS[c] else 'v'}" for c in metrics}
    view = view.rename(columns=renamed)
    with pd.option_context("display.width", 200, "display.max_columns", 60):
        return view.to_string(index=False, float_format=lambda v: f"{v:.{precision}f}")


def compare(table: pd.DataFrame, baseline: str, *, on: str = "name") -> pd.DataFrame:
    """Deltas against one arm, signed so positive always means better.

    Args:
        table: a sweep result.
        baseline: the value of ``on`` to treat as the reference arm.
        on: the column identifying arms.

    Example:
        >>> import pandas as pd
        >>> from nilmframe.eval import compare
        >>> table = pd.DataFrame([{'name': 'a', 'f1_macro': 0.8, 'mae': 10.0},
        ...                       {'name': 'b', 'f1_macro': 0.6, 'mae': 20.0}])
        >>> compare(table, baseline='a')
          name  f1_macro   mae
        0    a       0.0  -0.0
        1    b      -0.2 -10.0
    """
    if baseline not in set(table[on]):
        raise ValueError(f"no arm {baseline!r} in column {on!r}; have {sorted(set(table[on]))}")

    reference = table[table[on] == baseline].iloc[0]
    rows = []
    for _, row in table.iterrows():
        delta = {on: row[on]}
        for column, higher_is_better in HEADLINE_COLUMNS.items():
            if column not in table.columns:
                continue
            change = float(row[column]) - float(reference[column])
            delta[column] = change if higher_is_better else -change
        rows.append(delta)
    return pd.DataFrame(rows)


def load_results(path: str | Path) -> pd.DataFrame:
    """Read a ``results.csv`` written by a sweep.

    Args:
        path: the file, or the run directory's ``results.csv``.

    Returns:
        One row per arm, with the identity columns (name, view, model,
        parameters, seconds) followed by every metric the evaluator produced.
        Pair it with :func:`format_table` to print, or :func:`compare` to diff
        arms against a baseline.

    Example:
        >>> import pathlib, tempfile, pandas as pd
        >>> from nilmframe.eval import load_results, format_table
        >>> run_dir = pathlib.Path(tempfile.mkdtemp())
        >>> pd.DataFrame([{"name": "lf", "f1_macro": 0.71, "nde": 0.42},
        ...               {"name": "hf", "f1_macro": 0.83, "nde": 0.29}]
        ...              ).to_csv(run_dir / "results.csv", index=False)
        >>> results = load_results(run_dir)
        >>> list(results.name)
        ['lf', 'hf']
        >>> print(format_table(results, precision=3))
        name  f1_macro ^  nde v
          lf       0.710  0.420
          hf       0.830  0.290
    """
    path = Path(path)
    if path.is_dir():
        path = path / "results.csv"
    return pd.read_csv(path)
