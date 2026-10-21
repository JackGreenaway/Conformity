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
    api = "\n".join(
        path.read_text()
        for path in [
            ROOT / "docs/api/metrics.md",
            *sorted((ROOT / "docs/api/metrics").glob("*.md")),
        ]
    )
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
            heading = rf"^#{{1,6}} {re.escape(name)} \{{#{re.escape(name.replace('_', '-'))}\}}"
            if not re.search(heading, api, re.M):
                raise ValueError(f"Undocumented export: {name}")
            targets = [obj]
        for target in targets:
            if not inspect.isclass(target) and target.__module__.startswith(
                "conformity"
            ):
                for section in ("Returns", "Raises"):
                    if not re.search(
                        rf"\n{section}\n-{{3,}}\n", inspect.getdoc(target) or ""
                    ):
                        raise ValueError(
                            f"Missing {section} contract: {name}.{target.__name__}"
                        )
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


def main() -> None:
    """Lint, optionally format, and execute Python fences as isolated examples."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", action="store_true")
    args = parser.parse_args()
    check_coverage()
    count = 0
    outputs = {}
    paths = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
    with tempfile.TemporaryDirectory(prefix="conformity-docs-") as directory:
        for path in paths:
            content = path.read_text()
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
                if not args.format and code != formatted:
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
                if path.parent == ROOT / "docs/examples":
                    download = ROOT / "docs/examples/downloads" / f"{path.stem}.py"
                    if args.format:
                        download.parent.mkdir(parents=True, exist_ok=True)
                        download.write_text(formatted)
                    elif not download.exists() or download.read_text() != formatted:
                        raise ValueError(f"Stale downloadable example: {download}")
                    output_match = re.search(
                        r"<!-- BEGIN EXAMPLE OUTPUT -->.*?```text\n(.*?)```.*?<!-- END EXAMPLE OUTPUT -->",
                        content,
                        re.S,
                    )
                    if not args.format and (
                        output_match is None or output_match[1] != result.stdout
                    ):
                        raise ValueError(
                            f"Displayed output mismatch: {path}; run with --format"
                        )
                    if args.format:
                        outputs[path] = result.stdout
                count += 1
                return f"```python\n{formatted if args.format else code}```"

            updated = re.sub(r"```python\n(.*?)```", check, content, flags=re.S)
            if args.format and path in outputs:
                rendered = (
                    "<!-- BEGIN EXAMPLE OUTPUT -->\n\n"
                    f"```text\n{outputs[path]}```\n\n<!-- END EXAMPLE OUTPUT -->"
                )
                if "<!-- BEGIN EXAMPLE OUTPUT -->" in updated:
                    updated = re.sub(
                        r"<!-- BEGIN EXAMPLE OUTPUT -->.*?<!-- END EXAMPLE OUTPUT -->",
                        lambda _: rendered,
                        updated,
                        flags=re.S,
                    )
                else:
                    updated = (
                        updated.replace(
                            "## Results and interpretation",
                            rendered + "\n\n## Results and interpretation",
                        )
                        if "## Results and interpretation" in updated
                        else updated.replace(
                            "## Configuration and interpretation",
                            rendered + "\n\n## Configuration and interpretation",
                        )
                    )
            if args.format and updated != content:
                path.write_text(updated)
    print(f"Verified API coverage and {count} runnable Python examples.")


if __name__ == "__main__":
    main()
