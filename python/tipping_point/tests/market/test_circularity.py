"""The labels cannot see the features (#29; research pass section 4, "Circularity, static"): no
module of the market package imports ews.indicators or ews.trend, directly or through any module
it imports. The labels manifest's sources are checked in test_cli."""

import ast
import importlib.util
from pathlib import Path

import tipping_point.market

FEATURES = {"tipping_point.ews.indicators", "tipping_point.ews.trend"}


def _imports(name: str) -> set[str]:
    """The tipping_point modules that module `name` imports, relative imports resolved."""
    spec = importlib.util.find_spec(name)
    assert spec is not None and spec.origin is not None
    tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))
    package = name if spec.submodule_search_locations is not None else name.rpartition(".")[0]
    named: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            named |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            relative = "." * node.level + (node.module or "")
            base = importlib.util.resolve_name(relative, package) if node.level else relative
            named |= {base} | {f"{base}.{alias.name}" for alias in node.names}
    return {m for m in named if m.split(".")[0] == "tipping_point" and _is_module(m)}


def _is_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:  # a name inside a module, such as tipping_point.__version__.x
        return False


def test_the_labeller_imports_nothing_from_the_ews_features() -> None:
    folder = Path(tipping_point.market.__file__).parent
    todo = ["tipping_point.market"] + [
        f"tipping_point.market.{p.stem}" for p in folder.glob("*.py") if p.stem != "__init__"
    ]
    seen: set[str] = set()
    while todo:
        name = todo.pop()
        if name not in seen:
            seen.add(name)
            todo += sorted(_imports(name))
    assert {"tipping_point.market.msvol", "tipping_point.ising.io"} <= seen  # it did walk
    assert not seen & FEATURES
