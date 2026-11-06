"""The labels cannot see the features (#29; research pass section 4, "Circularity, static"): no
module of the market package imports ews.indicators or ews.trend, directly or through any module
it imports. The labels manifest's sources are checked in test_cli.

The same holds for the order-book features (#30, research pass 4.3 item 5): the market package
imports nothing from tipping_point.book, and the book package imports nothing from the labeller
(market.msvol, market.cascade)."""

import ast
import importlib.util
from pathlib import Path

import tipping_point.book
import tipping_point.market

FEATURES = {"tipping_point.ews.indicators", "tipping_point.ews.trend"}
LABELLER = {"tipping_point.market.msvol", "tipping_point.market.cascade"}


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


def _closure(package: object) -> set[str]:
    """Every tipping_point module a package's modules import, directly or through each other."""
    folder = Path(package.__file__).parent  # type: ignore[attr-defined]
    name = package.__name__  # type: ignore[attr-defined]
    todo = [name] + [f"{name}.{p.stem}" for p in folder.glob("*.py") if p.stem != "__init__"]
    seen: set[str] = set()
    while todo:
        module = todo.pop()
        if module not in seen:
            seen.add(module)
            todo += sorted(_imports(module))
    return seen


def test_the_labeller_imports_nothing_from_the_ews_features() -> None:
    seen = _closure(tipping_point.market)
    assert {"tipping_point.market.msvol", "tipping_point.ising.io"} <= seen  # it did walk
    assert not seen & FEATURES


def test_the_labeller_and_the_book_features_import_nothing_from_each_other() -> None:
    market = _closure(tipping_point.market)
    assert not {m for m in market if m.split(".")[:2] == ["tipping_point", "book"]}
    book = _closure(tipping_point.book)
    assert {"tipping_point.book.features", "tipping_point.book.__main__"} <= book  # it did walk
    assert not book & LABELLER
