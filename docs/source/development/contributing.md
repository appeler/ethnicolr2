# Contributing to ethnicolr2

`ethnicolr2` accepts maintenance changes for correctness, security,
compatibility, packaging, tests, and documentation. Propose new models and
features in [ethnicolr](https://github.com/appeler/ethnicolr), the canonical
package.

## Report a bug

Open a GitHub issue with:

1. The `ethnicolr2`, Python, and operating system versions.
2. The smallest example that reproduces the problem.
3. The expected and actual results.
4. The complete traceback, if one exists.

Model-output reports should include the input rows, selected name columns, and
the function called. Remove personal data before posting an example.

## Develop locally

Clone the repository and install every development group with
[uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/appeler/ethnicolr2.git
cd ethnicolr2
uv sync --all-groups
```

Create a focused branch for the fix. Include a regression test that fails
without the fix and passes with it. Do not add new model families or change the
pinned checkpoints unless the pull request fixes a documented model defect and
includes reproducible validation.

Run the complete local checks before opening a pull request:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pydoclint src/ethnicolr2
uv run pytest
uv run sphinx-build -W -b html docs/source docs/build/html
```

Pull requests should explain the defect, the fix, and the evidence that the fix
works. Keep unrelated cleanup out of the same change.
