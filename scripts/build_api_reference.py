"""Generate scikit-learn-style class references from live signatures/docstrings."""

import argparse
import html
import inspect
import re
from pathlib import Path

import conformity

ROOT = Path(__file__).resolve().parents[1]
CLASSES = {
    "BaseConformalPredictor": "base-conformal-predictor",
    "ConformalRegressor": "conformal-regressor",
    "ConformalClassifier": "conformal-classifier",
}
START = "<!-- BEGIN GENERATED REFERENCE -->"
END = "<!-- END GENERATED REFERENCE -->"


def signature(obj):
    sig = inspect.signature(obj)
    parameters = [
        p.replace(annotation=inspect.Parameter.empty)
        for p in sig.parameters.values()
        if p.name != "self"
    ]
    return str(
        sig.replace(parameters=parameters, return_annotation=inspect.Signature.empty)
    )


def signature_block(obj, name, *, is_class=False):
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
    prefix = "conformity." if is_class else ""
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


def inline(text):
    text = html.escape(text)
    return re.sub(r"``(.*?)``", r"<code>\1</code>", text)


def render_doc(doc, level):
    sections = re.split(r"\n([A-Za-z ]+)\n-{3,}\n", inspect.cleandoc(doc))
    output = [sections[0].replace("``", "`").strip() + "\n"]
    for heading, body in zip(sections[1::2], sections[2::2]):
        if heading in {"Examples", "See Also"}:
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


def public_methods(cls):
    return {
        name: obj
        for name, obj in inspect.getmembers(cls, callable)
        if not name.startswith("_")
    }


def reference(cls):
    name = cls.__name__
    output = [
        "## Reference\n",
        signature_block(cls, name, is_class=True),
        '<div class="api-class-body" markdown="1">\n',
        render_doc(inspect.getdoc(cls) or "", 3),
        render_doc(inspect.getdoc(cls.__init__) or "", 3),
    ]
    if cls is not conformity.BaseConformalPredictor:
        output.append(
            render_doc(inspect.getdoc(conformity.BaseConformalPredictor) or "", 3)
        )
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
                render_doc(inspect.getdoc(obj) or "", 4),
                "</div>\n",
            ]
        )
    for name, obj in inspect.getmembers(cls):
        if isinstance(obj, property) and not name.startswith("_"):
            output.extend([f"### {name}\n", render_doc(inspect.getdoc(obj) or "", 4)])
    output.append("</div>\n")
    return "\n".join(output).strip()


def main():
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
                render_doc(inspect.getdoc(obj) or "", 3),
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
