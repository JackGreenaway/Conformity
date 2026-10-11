# Development and deployment

## Build from a checkout

The package supports Python 3.9 and newer; CI explicitly tests 3.9, 3.10, 3.11 and 3.12. Documentation and release tooling use Python 3.12. Use the committed lockfile so checks and builds use the same dependencies:

```bash
uv python install 3.12
uv sync --locked --group docs --group release --python 3.12
uv run --locked --group docs --group release --python 3.12 pytest --doctest-modules src tests
uv run --locked --group docs --group release --python 3.12 ruff check src/conformity tests
uv run --locked --group docs --group release --python 3.12 python scripts/build_api_reference.py
uv run --locked --group docs --group release --python 3.12 python scripts/check_docs.py
uv run --locked --group docs --group release --python 3.12 mkdocs build --strict
uv run --locked --group docs --group release --python 3.12 python scripts/check_site_links.py
uv run --locked --group docs --group release --python 3.12 mkdocs serve
```

`bash scripts/validate.sh` runs all release gates without publishing. It explicitly selects Python 3.12 in every uv run; set `TASK_PYTHON_VERSION` only when intentionally validating another supported tooling interpreter.

Pandas is a development dependency so DataFrame compatibility checks run rather than being skipped. Matplotlib executes the plotting example with a noninteractive backend. These are not required runtime dependencies. After changing dependency declarations or package version, run `uv lock`, then sync with `--locked`.

## Maintain the documentation

### How to write documentation

Start with the reader's question and explain the resulting behavior in plain language. Explain why and when to use a workflow before presenting its code. Introduce configuration choices that affect interpretation, such as `alpha`, calibration size and split strategy. After the example, explain output shapes, units and practical meaning, then link to the next relevant guide. Define technical terms where readers first need them.

Put task walkthroughs in getting started and the examples gallery, reusable explanations in the user guide, mathematical assumptions and proofs in theory, and contributor instructions on this page. Keep the root README concise and link to complete workflows. API contracts belong in source docstrings; generated reference sections must be regenerated rather than edited directly.

### Example conventions

- Use `Pipeline` with explicit, meaningful step names rather than `make_pipeline`. Prefer a wrapper around the final model and manual calibration held out before fitting learned preprocessing.
- Explain any whole-pipeline wrapper at the example itself: use it when automatic calibration must split before learned preprocessing or tuning. A final-step wrapper with automatic calibration is safe for preceding stateless or independently fixed transformations under the same sampling assumptions. Link to [pipeline placement](user-guide/estimators.md#why-the-split-matters) for the details.
- Show where training, calibration and test rows come from. Keep final calibration independent of preprocessing fitting, model fitting and model selection. State any departure from the usual exchangeability assumptions; computational support alone is not a coverage guarantee.
- For final-step wrappers, transform held-out features with the fitted pipeline prefix before calling custom wrapper methods. Use the full pipeline for `predict` so preprocessing is applied consistently.
- Make current Python blocks independently runnable: include imports, deterministic synthetic data or a bundled dataset, and fixed seeds where random choices affect results. Avoid network downloads and dependence on earlier blocks. Archived guides retain their documented page-level execution fixture.
- Include meaningful assertions for shapes or public behavior. Explain tuple ordering, class-column mapping and target units where relevant. Captured results must come from execution; describe finite-sample coverage as a diagnostic rather than an exact nominal target.
- Keep point-model scoring distinct from uncertainty evaluation. Explain that calibration used during model selection must be replaced by fresh held-out calibration for the final guarantee. Describe coverage as marginal under the stated assumptions, without implying individual or subgroup guarantees.

### API docstrings

Use the existing NumPy-style sections and document parameters, return values, relevant errors, fitted attributes and a short executable `Examples` transcript. State defaults, accepted shapes, fitted/calibrated prerequisites, and any state invalidation or replacement behavior. Keep signatures and behavior consistent with the implementation. Update the source docstring first, then regenerate the API reference so the site and source doctests share one contract.

### Review and validation

Check that the prose explains the actual code, including wrapper placement, calibration timing and any stated sample counts. Review generated outputs and update their interpretation if results change. When changing an example, refresh its downloadable script and captured output with the formatter below; when changing API docstrings, regenerate references. Run `bash scripts/validate.sh` before submitting documentation changes, and inspect the rendered page for readability, tables, mathematics and navigation.

### Sources and tooling

The existing MkDocs theme is adapted to scikit-learn’s section navigation, central reading column, page contents, signature panels and structured API fields. Both dark and light modes use the same navigation and keyboard focus indicators. All local URLs use MkDocs URL filters and relative Markdown paths. The canonical repository prefix is `/Conformity/`, including its capital C.

Edit source docstrings for signatures, parameters, fitted attributes and method behavior; then regenerate references. The generator replaces only marked regions, preserving authored guidance around them. `--check` rejects stale references. `check_docs.py` checks public export, parameter, fitted attribute and public method coverage, lints Python examples, and executes each block in an isolated subprocess and temporary working directory. Scikit-learn-style API examples are generated from source `Examples` docstring sections as `pycon` transcripts; their displayed outputs are checked with doctest both in the source suite and in the rendered Markdown example checks.

The three gallery workflows include downloadable Python scripts and captured output. The example checker verifies both against execution in the locked environment; `--format` refreshes them when an intentional change alters results. Review refreshed output before committing.

Format new examples with `python scripts/check_docs.py --format` through the locked uv command above. Existing guides are preserved in `docs/legacy/`: their contextual examples execute as a complete page with a seeded synthetic-data fixture, without reformatting. Keep `DOCUMENTATION.md` and `THEORY.md` copies synchronized if you edit their originals. The original README is preserved in `docs/legacy/README.md`; the root README is now the concise entry point.

`mkdocs build --strict` checks Markdown navigation and link anchors. `check_site_links.py` checks generated HTML links, anchors and local assets using the repository Pages URL, including the custom template. It checks local rendered targets; it does not contact external websites. Mathematical notation uses Arithmatex and MathJax. The MathJax runtime is pinned to 3.2.2 on jsDelivr; rendering mathematics needs access to that CDN.

## Build distributions locally

Choose a fresh empty output directory to avoid mixing versions. The release group locks build, Hatchling and Twine. Hatchling is constrained to >=1.26,<1.28 because later releases can emit metadata 2.5 that the locked Twine cannot validate; disabling build isolation makes the builder use that locked environment. Both wheel and source archive are produced, and the wheel is built from the source archive:

```bash
uv sync --locked --group docs --group release --python 3.12
uv run --locked --group docs --group release --python 3.12 python -m build --no-isolation --outdir /tmp/conformity-release-dist
uv run --locked --group docs --group release --python 3.12 python -m twine check --strict /tmp/conformity-release-dist/*
uv run --locked --group docs --group release --python 3.12 python scripts/check_release.py --tag v0.1.8 --dist /tmp/conformity-release-dist
```

The final command checks project.version, explicit `conformity.__version__`, the supplied tag, archive metadata, embedded versions, distribution name and Python support. The workflow creates distributions in a fresh runner temporary directory; local `dist/` files are never published.

## GitHub Pages setup

In this repository's **Settings → Pages**, select **GitHub Actions** as the build source. Allow Actions to run and the `github-pages` environment to deploy from `main`; configure any desired environment protection rules. The Documentation workflow checks pull requests and deploys only successful non-PR builds on `main`, using `pages: write` and `id-token: write` in the isolated deployment job.

The site will be available at `https://jackgreenaway.github.io/Conformity/` after the first successful deployment. If the repository name or default branch changes, update `site_url`, the rendered-link checker and workflow branch guards together. No deployment has been performed as part of these local changes.

## PyPI trusted publishing setup

Create the GitHub environment **`pypi`** and set its protection rules. On the **`conformity-calib`** PyPI project, configure a trusted publisher for:

| Setting | Value |
| --- | --- |
| Owner | `JackGreenaway` |
| Repository | `Conformity` |
| Workflow filename | `python-publish.yml` |
| Environment | `pypi` |

The publisher uses OIDC in a separate job with `id-token: write`, downloads only validated artifacts and does not use an API-token secret. Configure these settings before invoking a release. Install both release and publishing workflows on the default branch. The publisher listens for `workflow_run`, which requires its workflow file on that branch.

## Release procedure

1. Set `project.version` and `src/conformity/__init__.py`'s `__version__` to the same new `X.Y.Z`; update `uv.lock` and regenerate references if implementation changed.
2. Run the local validation and distribution checks above. Commit and push the intended revision.
3. Run **Create GitHub Release** from Actions on the intended revision. It runs package tests and documentation checks on Python 3.12, rejects existing version tags and builds/checks fresh distributions before creating `vX.Y.Z` and attaching artifacts.
4. Once **Create GitHub Release** finishes successfully, a separate **Upload Python Package** run starts through `workflow_run`. It accepts only successful manual release-workflow runs from this repository, downloads `created-release-dists` from that exact run, and publishes those same artifacts through the `pypi` environment. This matches polars-learn's handoff and works when `GITHUB_TOKEN` suppresses release-triggered workflows. Releases published manually through GitHub also validate the tag and source and build/check fresh distributions before publishing.

Publishing runs directly in `python-publish.yml`, rather than through a reusable workflow. PyPI rejected the previous reusable handoff because its attestation named the parent `create-release.yml` instead of the configured trusted publisher; see [PyPI's reusable-workflow limitation](https://docs.pypi.org/trusted-publishers/troubleshooting/#reusable-workflows-on-github). Keep the trusted publisher set to `python-publish.yml` and environment `pypi`.

If publishing fails after release creation, fix the setup and rerun **Upload Python Package** while the original release-run artifacts remain available. Do not try to create the same tag again. PyPI versions cannot be overwritten; use a new version for changed artifacts. Prerelease versions are outside these `X.Y.Z` workflows.

### Recovering the existing v0.1.7 release

The failed run created the `v0.1.7` GitHub release successfully; its PyPI upload was rejected. Do not run **Create GitHub Release** again for that version: its existing-tag check will correctly fail.

After committing these workflow fixes to `main`, start a new publisher run for the existing release by converting it to a draft and publishing it again through GitHub Releases. The `release: published` path checks out the existing tag and validates/builds that source before uploading it. This publishes the tagged version, not any later uncommitted documentation edits. Confirm that the PyPI publisher settings above are in place. Rerunning the old failed reusable-workflow jobs alone uses the old workflow revision and does not apply this fix.

## Documentation design references

Inspected directly on 11 October 2026: scikit-learn’s
[user guide](https://scikit-learn.org/stable/user_guide.html),
[examples gallery](https://scikit-learn.org/stable/auto_examples/index.html),
[StandardScaler class reference](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html),
and [train_test_split function reference](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html).
The guide separates conceptual explanations from callable contracts; the gallery
organizes complete workflows by question; API pages pair generated signatures
with structured fields, methods, source links and short executable examples.
Conformity retains MkDocs rather than adding a second documentation framework.
