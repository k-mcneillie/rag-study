"""Tests that enforce the package's dependency rules.

The central claim of this architecture is that the two pipelines are
independent: either directory could be deleted and the other would still work.
That claim is easy to state and easy to break with a single convenient import,
so it is checked here rather than left to review.

The rules are read from the source itself, so they hold for code that has not
been written yet.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import rag

PACKAGE_ROOT = Path(rag.__file__).parent

#: What each package is permitted to import from within this project.
#: ``domain`` depends on nothing; ``storage`` on the contracts alone; each
#: pipeline on the shared foundations, and never on the other pipeline.
#: ``generation`` and ``feedback`` sit above the pipelines and see only the
#: contracts they are handed, which is why neither appears in any allowance
#: but its own.
ALLOWED_IMPORTS = {
    "config": set(),
    "domain": set(),
    "model_assets": set(),
    "feedback": {"domain"},
    "storage": {"domain", "config"},
    "ingestion": {"domain", "storage", "model_assets", "config"},
    "retrieval": {"domain", "storage", "model_assets", "config"},
    "generation": {"domain", "config"},
}


def _internal_imports(source: Path) -> set[str]:
    """Find which parts of this package a module imports.

    Args:
        source: The Python file to inspect.

    Returns:
        The top-level ``rag`` submodules the file imports.
    """
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    imported: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            module = node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("rag."):
                    imported.add(alias.name.split(".")[1])
            continue
        else:
            continue

        if module == "rag" or module.startswith("rag."):
            parts = module.split(".")
            if len(parts) > 1:
                imported.add(parts[1])

    return imported


def _modules_under(package: str) -> list[Path]:
    """List the Python files belonging to a top-level part of the package.

    Args:
        package: The submodule name, for example ``"ingestion"``.

    Returns:
        Every Python file in that submodule.
    """
    target = PACKAGE_ROOT / package
    if target.is_dir():
        return sorted(target.rglob("*.py"))
    return [PACKAGE_ROOT / f"{package}.py"]


@pytest.mark.parametrize("package", sorted(ALLOWED_IMPORTS))
def test_package_imports_stay_within_their_allowance(package: str) -> None:
    """Each part of the package imports only what its layer permits.

    Args:
        package: The submodule whose imports are checked.
    """
    allowed = ALLOWED_IMPORTS[package] | {package}

    for source in _modules_under(package):
        forbidden = _internal_imports(source) - allowed
        assert not forbidden, (
            f"{source.relative_to(PACKAGE_ROOT)} imports {sorted(forbidden)}, "
            f"which {package} may not depend on."
        )


def test_the_pipelines_never_import_each_other() -> None:
    """Deleting either pipeline leaves the other intact.

    This is the architecture's defining property, stated directly rather than
    inferred from the allowance table above.
    """
    for pipeline, forbidden in (("ingestion", "retrieval"), ("retrieval", "ingestion")):
        for source in _modules_under(pipeline):
            assert forbidden not in _internal_imports(source), (
                f"{source.relative_to(PACKAGE_ROOT)} imports {forbidden}; the "
                f"pipelines must remain independent."
            )


def test_generation_never_imports_a_pipeline() -> None:
    """Answering sees the contracts, not the machinery that produced them.

    The prompt's boundary between application instructions and untrusted
    document content is established once, by the prompt augmenter. A model
    client that could reach into retrieval could rebuild that prompt from
    parts, moving a security boundary it does not own. Keeping generation
    blind to both pipelines also means it can be tested against a
    hand-constructed context, with no database and no models.
    """
    for source in _modules_under("generation"):
        imported = _internal_imports(source)
        for forbidden in ("retrieval", "ingestion", "storage"):
            assert forbidden not in imported, (
                f"{source.relative_to(PACKAGE_ROOT)} imports {forbidden}; "
                f"generation consumes the domain contracts alone."
            )


def test_domain_contracts_are_free_of_frameworks() -> None:
    """The data contracts depend on no third-party library.

    A framework reaching into the domain would tie every component that speaks
    these contracts to that framework.
    """
    banned = {
        "sqlalchemy",
        "pymupdf",
        "pymupdf4llm",
        "sentence_transformers",
        "langchain_text_splitters",
        "ftfy",
        "numpy",
    }

    for source in _modules_under("domain"):
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            for name in names:
                assert name.split(".")[0] not in banned, (
                    f"{source.relative_to(PACKAGE_ROOT)} imports {name}; the "
                    f"domain contracts must stay framework-free."
                )


def test_only_storage_imports_sqlalchemy() -> None:
    """Database concerns stay inside the storage layer.

    A processing component that imported SQLAlchemy could reach past the
    repository, and the store would no longer be replaceable.
    """
    for package in (
        "domain",
        "config",
        "ingestion",
        "retrieval",
        "generation",
        "feedback",
        "model_assets",
    ):
        for source in _modules_under(package):
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    root = node.module.split(".")[0]
                elif isinstance(node, ast.Import):
                    root = node.names[0].name.split(".")[0]
                else:
                    continue
                assert root != "sqlalchemy", (
                    f"{source.relative_to(PACKAGE_ROOT)} imports SQLAlchemy; "
                    f"only the storage layer may."
                )
