"""Validate source versions, release tags and built archive metadata."""

import argparse
import ast
import email
import os
from pathlib import Path
import re
import tarfile
import tomllib
import zipfile

import conformity

ROOT = Path(__file__).resolve().parents[1]


def literal_version(source):
    """Read the explicit package version without executing archive contents."""
    for statement in ast.parse(source).body:
        if isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            if statement.target.id == "__version__":
                return ast.literal_eval(statement.value)
        if isinstance(statement, ast.Assign):
            if any(
                isinstance(target, ast.Name) and target.id == "__version__"
                for target in statement.targets
            ):
                return ast.literal_eval(statement.value)
    raise ValueError("Archive has no explicit __version__")


def check_distributions(directory, version):
    """Check exactly one wheel and sdist and their embedded code and metadata."""
    paths = list(directory.iterdir())
    wheels = list(directory.glob("*.whl"))
    sources = list(directory.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1 or len(paths) != 2:
        raise ValueError(
            "Expected exactly one wheel and one source archive in a clean directory"
        )
    with zipfile.ZipFile(wheels[0]) as archive:
        names = archive.namelist()
        metadata_names = [
            name for name in names if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_names) != 1:
            raise ValueError("Wheel must have one metadata record")
        wheel_metadata = email.message_from_bytes(archive.read(metadata_names[0]))
        wheel_version = literal_version(archive.read("conformity/__init__.py"))
    with tarfile.open(sources[0]) as archive:

        def read(suffix):
            matches = [
                member
                for member in archive.getmembers()
                if member.name.endswith(suffix) and member.isfile()
            ]
            if len(matches) != 1:
                raise ValueError(f"Expected one source member: {suffix}")
            return archive.extractfile(matches[0]).read()

        source_metadata = email.message_from_bytes(read("/PKG-INFO"))
        source_version = literal_version(read("/src/conformity/__init__.py"))
        project_version = tomllib.loads(read("/pyproject.toml").decode())["project"][
            "version"
        ]
    for metadata in (wheel_metadata, source_metadata):
        if metadata["Name"] != "conformity-calib" or metadata["Version"] != version:
            raise ValueError("Distribution name/version does not match project")
        if metadata["Requires-Python"] != ">=3.9":
            raise ValueError("Distribution changed supported Python versions")
    if {wheel_version, source_version, project_version} != {version}:
        raise ValueError("Distribution versions disagree with source")


def main():
    """Verify a release before creation or trusted publishing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=os.environ.get("RELEASE_TAG"))
    parser.add_argument("--dist", type=Path)
    parser.add_argument("--github-output", action="store_true")
    args = parser.parse_args()
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version):
        raise ValueError("Release version must have the form X.Y.Z")
    source_version = literal_version((ROOT / "src/conformity/__init__.py").read_text())
    if conformity.__version__ != version or source_version != version:
        raise ValueError("Package __version__ must match project.version")
    tag = f"v{version}"
    if args.tag is not None and args.tag != tag:
        raise ValueError("Release tag, project.version and __version__ must agree")
    if args.dist:
        check_distributions(args.dist, version)
    if args.github_output:
        with open(os.environ["GITHUB_OUTPUT"], "a") as output:
            output.write(f"tag={tag}\n")
    print(f"Validated {tag}" + (" and distributions" if args.dist else ""))


if __name__ == "__main__":
    main()
