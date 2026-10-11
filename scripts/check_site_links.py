"""Check rendered local links, anchors and assets under the Pages repository path."""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
import re
from urllib.parse import unquote, urljoin, urlsplit

ROOT = Path(__file__).resolve().parents[1] / "site"
BASE = "https://jackgreenaway.github.io/Conformity/"


class Page(HTMLParser):
    """Collect rendered targets and references without network requests."""

    def __init__(self, path: Path) -> None:
        """Parse an HTML file and collect its anchors and local references."""
        super().__init__()
        self.ids = set()
        self.links = []
        self.feed(path.read_text())

    def handle_starttag(
        self, tag: str, attributes: list[tuple[str, str | None]]
    ) -> None:
        """Record anchor targets and references from an opening HTML tag."""
        attrs = dict(attributes)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag == "a" and "name" in attrs:
            self.ids.add(attrs["name"])
        for key in ("href", "src", "poster"):
            if attrs.get(key):
                self.links.append(attrs[key])


def main() -> None:
    """Resolve links as a browser would on GitHub Pages, including fragments."""
    pages = {path.resolve(): Page(path) for path in ROOT.rglob("*.html")}
    if not pages:
        raise ValueError("Build the site first")
    checked = 0
    for path, page in pages.items():
        relative = path.relative_to(ROOT.resolve()).as_posix()
        url = urljoin(BASE, relative.removesuffix("index.html"))
        for link in page.links:
            target = urlsplit(urljoin(url, link))
            if (
                target.scheme not in {"http", "https"}
                or target.netloc != urlsplit(BASE).netloc
            ):
                continue
            if not target.path.startswith("/Conformity/"):
                raise ValueError(f"Link escapes repository path: {relative}: {link}")
            destination = (ROOT / unquote(target.path[len("/Conformity/") :])).resolve()
            if destination.is_dir():
                destination /= "index.html"
            if (
                not destination.is_relative_to(ROOT.resolve())
                or not destination.is_file()
            ):
                raise ValueError(f"Missing rendered target: {relative}: {link}")
            if (
                target.fragment
                and destination in pages
                and unquote(target.fragment) not in pages[destination].ids
            ):
                raise ValueError(f"Missing rendered anchor: {relative}: {link}")
            checked += 1
    # CSS can reference fonts or images absent from the HTML link inventory.
    for path in ROOT.rglob("*.css"):
        relative = path.relative_to(ROOT).as_posix()
        for link in re.findall(r"url\(\s*['\"]?([^)'\"\s]+)", path.read_text()):
            target = urlsplit(urljoin(urljoin(BASE, relative), link))
            if (
                target.scheme not in {"http", "https"}
                or target.netloc != urlsplit(BASE).netloc
            ):
                continue
            if not target.path.startswith("/Conformity/"):
                raise ValueError(
                    f"CSS asset escapes repository path: {relative}: {link}"
                )
            destination = (ROOT / unquote(target.path[len("/Conformity/") :])).resolve()
            if (
                not destination.is_relative_to(ROOT.resolve())
                or not destination.is_file()
            ):
                raise ValueError(f"Missing CSS asset: {relative}: {link}")
            checked += 1
    # Theme and MathJax contract: present on every documentation page.
    for path, page in pages.items():
        if path.name in {"404.html"}:
            continue
        if "main-content" not in page.ids:
            raise ValueError(f"Missing custom theme: {path}")
        if not any("javascripts/mathjax.js" in link for link in page.links):
            raise ValueError(f"Missing math configuration: {path}")
    print(
        f"Verified {checked} rendered links and assets across {len(pages)} HTML pages."
    )


if __name__ == "__main__":
    main()
