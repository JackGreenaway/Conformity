"""Generate scikit-learn-style class references from live signatures/docstrings."""

from __future__ import annotations

import argparse
import html
import inspect
import re
from pathlib import Path
from typing import Any, Callable

import conformity

ROOT = Path(__file__).resolve().parents[1]
CLASSES = {
    "BaseConformalPredictor": "base-conformal-predictor",
    "ConformalRegressor": "conformal-regressor",
    "ConformalClassifier": "conformal-classifier",
}
START = "<!-- BEGIN GENERATED REFERENCE -->"
END = "<!-- END GENERATED REFERENCE -->"


def signature(obj: Callable[..., Any]) -> str:
    """Render callable parameters without annotations or the self argument."""
    sig = inspect.signature(obj)
    parameters = [
        p.replace(annotation=inspect.Parameter.empty)
        for p in sig.parameters.values()
        if p.name != "self"
    ]
    return str(
        sig.replace(parameters=parameters, return_annotation=inspect.Signature.empty)
    )


def signature_block(
    obj: Callable[..., Any], name: str, *, is_class: bool = False
) -> str:
    """Render an HTML signature with a link to its source implementation."""
    parameters = []
    keyword_separator = False
    for param in inspect.signature(obj).parameters.values():
        if param.name == "self":
            continue
        if param.kind == inspect.Parameter.KEYWORD_ONLY and not keyword_separator:
            parameters.append("*")
            keyword_separator = True
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            keyword_separator = True
        parameters.append(str(param.replace(annotation=inspect.Parameter.empty)))
    label = "class " if is_class else ""
    prefix = (
        "conformity."
        if is_class
        or inspect.isfunction(obj)
        and obj.__module__ == "conformity.metrics"
        else ""
    )
    rendered_name = (
        f'<span class="api-keyword">{label}</span>'
        f'<span class="api-module">{prefix}</span>'
        f'<span class="api-name">{html.escape(name)}</span>'
    )
    if is_class or len(parameters) > 2:
        arguments = "\n".join("    " + html.escape(p) + "," for p in parameters)
        call = f"{rendered_name}(\n{arguments}\n)"
    else:
        call = f"{rendered_name}({html.escape(', '.join(parameters))})"
    source_path = Path(inspect.getsourcefile(obj)).resolve()
    package_root = ROOT / "src"
    if source_path.is_relative_to(package_root):
        line = inspect.getsourcelines(obj)[1]
        source_url = (
            "https://github.com/JackGreenaway/Conformity/blob/main/"
            f"{source_path.relative_to(ROOT).as_posix()}#L{line}"
        )
        source_label = "[source]"
    else:
        source_url = (
            "https://scikit-learn.org/stable/modules/generated/"
            f"sklearn.base.BaseEstimator.html#sklearn.base.BaseEstimator.{name}"
        )
        source_label = "[inherited API]"
    return (
        '<div class="api-signature">'
        f'<a class="api-source" href="{source_url}">{source_label}</a>'
        f"<pre><code>{call}</code></pre></div>\n"
    )


def inline(text: str) -> str:
    """Escape text and render inline code from NumPy-style docstrings."""
    text = html.escape(text)
    return re.sub(r"``(.*?)``", r"<code>\1</code>", text)


def render_doc(
    doc: str,
    level: int,
    *,
    include_summary: bool = True,
    include_examples: bool = False,
    obj: Callable[..., Any] | None = None,
) -> str:
    """Render a short summary and structured reference fields from a docstring."""
    sections = re.split(r"\n([A-Za-z ]+)\n-{3,}\n", inspect.cleandoc(doc))
    # Reference pages contain the short code contract and structured fields.
    # Keep API examples concise; broader explanations belong in the user guide.
    summary = sections[0].split("\n\n", 1)[0].replace("``", "`").strip()
    output = [summary + "\n"] if include_summary else []
    for heading, body in zip(sections[1::2], sections[2::2]):
        if heading == "Examples" and include_examples:
            output.append(f"{'#' * level} Examples\n\n```pycon\n{body.strip()}\n```\n")
            continue
        if heading not in {"Parameters", "Attributes", "Returns", "Raises"}:
            continue
        output.append(f"{'#' * level} {heading}\n")
        if heading in {"Parameters", "Attributes", "Returns", "Raises"}:
            output.append('<dl class="api-fields">')
            entries = re.split(r"\n(?=\S)", body.strip())
            for entry in entries:
                lines = entry.splitlines()
                term = lines[0]
                description = " ".join(line.strip() for line in lines[1:])
                field_name, separator, field_type = term.partition(" : ")
                if heading == "Parameters" and obj is not None:
                    parameter = inspect.signature(obj).parameters.get(
                        field_name.lstrip("*")
                    )
                    if (
                        parameter is not None
                        and parameter.default is not inspect.Parameter.empty
                    ):
                        field_type += (
                            ", " if field_type else ""
                        ) + f"default={parameter.default!r}"
                        separator = True
                label = f'<code class="api-field-name">{inline(field_name)}</code>'
                if separator:
                    label += (
                        f' <span class="api-field-type">{inline(field_type)}</span>'
                    )
                output.append(f"<dt>{label}</dt><dd><p>{inline(description)}</p></dd>")
            output.append("</dl>\n")
        else:
            output.append(body.replace("``", "`").strip() + "\n")
    return "\n".join(output)


def public_methods(cls: type) -> dict[str, Callable[..., Any]]:
    """Collect public callable members, including inherited estimator methods."""
    return {
        name: obj
        for name, obj in inspect.getmembers(cls, callable)
        if not name.startswith("_")
    }


def reference(cls: type) -> str:
    """Generate the constructor, fitted fields and public method reference."""
    name = cls.__name__
    output = [
        "## Reference\n",
        signature_block(cls, name, is_class=True),
        '<div class="api-class-body" markdown="1">\n',
        (inspect.getdoc(cls) or "").split("\n\n", 1)[0] + "\n",
        render_doc(
            inspect.getdoc(cls.__init__) or "", 3, include_summary=False, obj=cls
        ),
        render_doc(inspect.getdoc(cls) or "", 3, include_summary=False),
    ]
    if cls is not conformity.BaseConformalPredictor:
        output.append(
            render_doc(
                inspect.getdoc(conformity.BaseConformalPredictor) or "",
                3,
                include_summary=False,
            )
        )
    class_doc = inspect.getdoc(cls) or ""
    examples = re.search(
        r"\nExamples\n-{3,}\n(.*?)(?=\n[A-Za-z ]+\n-{3,}\n|\Z)", class_doc, re.S
    )
    if examples:
        output.append(f"### Examples\n\n```pycon\n{examples[1].strip()}\n```\n")
    output.append("### Methods\n\n| Method | Purpose |\n| --- | --- |")
    for method, obj in public_methods(cls).items():
        summary = (inspect.getdoc(obj) or "").splitlines()[0]
        output.append(f"| [{method}](#{method.replace('_', '-')}) | {summary} |")
    for method, obj in public_methods(cls).items():
        output.extend(
            [
                f"### {method} {{#{method.replace('_', '-')} .api-method-title}}\n",
                '<div class="api-method-body" markdown="1">\n',
                signature_block(obj, method),
                render_doc(inspect.getdoc(obj) or "", 4, obj=obj),
                "</div>\n",
            ]
        )
    for name, obj in inspect.getmembers(cls):
        if isinstance(obj, property) and not name.startswith("_"):
            output.extend(
                [f"### {name}\n", render_doc(inspect.getdoc(obj) or "", 4, obj=obj)]
            )
    output.append("</div>\n")
    return "\n".join(output).strip()


def main() -> None:
    """Regenerate marked reference blocks or verify they match the source."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for name, slug in CLASSES.items():
        path = ROOT / "docs" / "classes" / f"{slug}.md"
        content = path.read_text()
        generated = START + "\n" + reference(getattr(conformity, name)) + "\n" + END
        updated = re.sub(
            re.escape(START) + r".*?" + re.escape(END),
            lambda _, replacement=generated: replacement,
            content,
            flags=re.S,
        )
        updated = updated.rstrip() + "\n"
        if START not in content:
            raise ValueError(f"Missing reference marker in {path}")
        if args.check and updated != content:
            raise ValueError(
                f"Stale reference in {path}; run scripts/build_api_reference.py"
            )
        if not args.check:
            path.write_text(updated)
    api_path = ROOT / "docs" / "api.md"
    api = api_path.read_text()
    blocks = []
    for name in conformity.__all__:
        obj = getattr(conformity, name)
        if inspect.isclass(obj):
            continue
        blocks.extend(
            [
                f"## {name} {{#{name.replace('_', '-')}}}\n",
                signature_block(obj, name),
                render_doc(
                    inspect.getdoc(obj) or "", 3, include_examples=True, obj=obj
                ),
            ]
        )
    generated = START + "\n" + "\n".join(blocks) + "\n" + END
    if START not in api or END not in api:
        raise ValueError("Missing API reference markers")
    api = re.sub(
        re.escape(START) + r".*?" + re.escape(END), lambda _: generated, api, flags=re.S
    )
    if args.check and api != api_path.read_text():
        raise ValueError("Stale API signatures; run scripts/build_api_reference.py")
    if not args.check:
        api_path.write_text(api)
    print("All public references match source signatures and docstrings.")


if __name__ == "__main__":
    main()
