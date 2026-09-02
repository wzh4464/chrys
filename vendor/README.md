# Vendored PACT runtime

`wheels/pact_core-0.2.0.dev0-py3-none-any.whl` is built from
`SELab-Leibniz/pact` commit `aa9073bed4970481a035755990e1682e9de486d8`.
`pact-core.json` records its provenance and SHA-256 digest.

`uv sync` uses this wheel through `[tool.uv.sources]`. For a pip installation
from this checkout, make the wheel available to the resolver:

```bash
python -m pip install --find-links vendor/wheels .
```

Do not replace the wheel without updating the provenance file, lockfile, and
Chrys-PACT compatibility tests together.
