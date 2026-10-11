"""Validate public API coverage, preserved guides and standalone examples."""

import argparse
import doctest
import inspect
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import conformity
from build_api_reference import CLASSES, public_methods

ROOT = Path(__file__).resolve().parents[1]


def check_coverage() -> None:
    """Ensure every exported component, parameter and method is documented."""
    api = (ROOT / "docs/api.md").read_text()
    for name in conformity.__all__:
        obj = getattr(conformity, name)
        if not re.search(r"\nExamples\n-{3,}\n", inspect.getdoc(obj) or ""):
            raise ValueError(f"Missing API example: {name}")
        if inspect.isclass(obj):
            if name not in CLASSES:
                raise ValueError(f"Undocumented class: {name}")
            content = (ROOT / f"docs/classes/{CLASSES[name]}.md").read_text()
            for method in public_methods(obj):
                if f"### {method} {{" not in content:
                    raise ValueError(f"Undocumented method: {name}.{method}")
            targets = [obj, *public_methods(obj).values()]
        else:
            content = api
            if f"## {name} {{" not in api:
                raise ValueError(f"Undocumented export: {name}")
            targets = [obj]
        for target in targets:
            doc = (
                inspect.getdoc(target.__init__ if inspect.isclass(target) else target)
                or ""
            )
            for parameter in inspect.signature(target).parameters:
                if parameter != "self" and not re.search(
                    rf"^\s*\*{{0,2}}{re.escape(parameter)}(?:\s*:.*)?\s*$", doc, re.M
                ):
                    raise ValueError(
                        f"Missing parameter description: {name}.{parameter}"
                    )
        if inspect.isclass(obj):
            for parent in obj.__mro__:
                if parent.__module__.startswith("conformity"):
                    for attr in getattr(parent, "__annotations__", {}):
                        if attr.endswith("_") and attr not in content:
                            raise ValueError(f"Missing fitted attribute: {name}.{attr}")
    for filename in ("DOCUMENTATION.md", "THEORY.md"):
        if (ROOT / filename).read_bytes() != (
            ROOT / "docs/legacy" / filename
        ).read_bytes():
            raise ValueError(
                f"Preserved guide is stale: {filename}; refresh its docs/legacy copy"
            )


def main() -> None:
    """Lint, optionally format, and execute Python fences as isolated examples."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", action="store_true")
    args = parser.parse_args()
    check_coverage()
    count = 0
    legacy_prelude = """
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline as pipeline
from sklearn.preprocessing import StandardScaler
from conformity import ConformalRegressor
X_all, y_all = make_regression(n_samples=1200, n_features=4, noise=20, random_state=42)
X, y = X_all[:1000], y_all[:1000]
X_new = X_all[1000:1005]
X_calib, y_calib = X_all[1005:], y_all[1005:]
pipe = pipeline(StandardScaler(), ConformalRegressor(Ridge(), random_state=42))
"""
    paths = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
    with tempfile.TemporaryDirectory(prefix="conformity-docs-") as directory:
        for path in paths:
            content = path.read_text()
            # Archived snippets use variables and imports from preceding blocks.
            # Execute the complete page with an explicit synthetic-data fixture.
            if "legacy" in path.parts:
                blocks = re.findall(r"```python\n(.*?)```", content, flags=re.S)
                if blocks:
                    code = legacy_prelude + "\n" + "\n".join(blocks)
                    example = Path(directory) / "legacy.py"
                    example.write_text(code)
                    result = subprocess.run(
                        [sys.executable, str(example)],
                        cwd=directory,
                        env=dict(os.environ, MPLBACKEND="Agg", MPLCONFIGDIR=directory),
                        text=True,
                        capture_output=True,
                        timeout=120,
                    )
                    if result.returncode:
                        raise ValueError(
                            f"Archived examples failed in {path}:\n{result.stdout}{result.stderr}"
                        )
                    count += len(blocks)
                continue

            for index, transcript in enumerate(
                re.findall(r"```pycon\n(.*?)```", content, flags=re.S)
            ):
                code = doctest.script_from_examples(transcript)
                lint = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "ruff",
                        "check",
                        "--select",
                        "E4,E7,E9,F",
                        "--stdin-filename",
                        "example.py",
                        "-",
                    ],
                    input=code,
                    text=True,
                    capture_output=True,
                    cwd=ROOT,
                )
                if lint.returncode:
                    raise ValueError(
                        f"Invalid API example in {path}:\n{lint.stdout}{lint.stderr}"
                    )
                example = Path(directory) / f"api-example-{index}.txt"
                example.write_text(transcript)
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "doctest",
                        "-o",
                        "NORMALIZE_WHITESPACE",
                        "-o",
                        "ELLIPSIS",
                        str(example),
                    ],
                    cwd=directory,
                    text=True,
                    capture_output=True,
                    timeout=120,
                )
                if result.returncode:
                    raise ValueError(
                        f"API output mismatch in {path}:\n{result.stdout}{result.stderr}"
                    )
                count += 1

            def check(match: re.Match[str]) -> str:
                nonlocal count
                code = match[1]
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "ruff",
                        "check",
                        "--select",
                        "E4,E7,E9,F",
                        "--stdin-filename",
                        "example.py",
                        "-",
                    ],
                    input=code,
                    text=True,
                    capture_output=True,
                    cwd=ROOT,
                )
                if result.returncode:
                    raise ValueError(
                        f"Invalid example in {path}:\n{result.stdout}{result.stderr}"
                    )
                # Preserve user-written legacy snippets; format only new examples.
                is_legacy = "legacy" in path.parts
                formatted = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "ruff",
                        "format",
                        "--stdin-filename",
                        "example.py",
                        "-",
                    ],
                    input=code,
                    text=True,
                    capture_output=True,
                    check=True,
                    cwd=ROOT,
                ).stdout
                if not is_legacy and not args.format and code != formatted:
                    raise ValueError(f"Unformatted example: {path}; run with --format")
                example = Path(directory) / "example.py"
                example.write_text(code)
                env = dict(os.environ, MPLBACKEND="Agg", MPLCONFIGDIR=directory)
                result = subprocess.run(
                    [sys.executable, str(example)],
                    cwd=directory,
                    env=env,
                    text=True,
                    capture_output=True,
                    timeout=120,
                )
                if result.returncode:
                    raise ValueError(
                        f"Example failed in {path}:\n{result.stdout}{result.stderr}"
                    )
                count += 1
                return (
                    match[0]
                    if is_legacy
                    else f"```python\n{formatted if args.format else code}```"
                )

            updated = re.sub(r"```python\n(.*?)```", check, content, flags=re.S)
            if args.format and updated != content:
                path.write_text(updated)
    print(f"Verified API coverage and {count} runnable Python examples.")


if __name__ == "__main__":
    main()
