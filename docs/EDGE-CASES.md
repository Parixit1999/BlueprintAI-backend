# Edge-case catalog

Every case from the requirement plus the ones we discovered building and
testing against real archive data. Each entry says how the system behaves -
"handled" means implemented AND exercised against the live stack.

## From the requirement

| Edge case | Behavior |
|---|---|
| Inconsistent project names | Matcher normalizes tokens; partial-name overlap (>= 50% of name words) still matches with the overlap stated in the reason |
| Partial project names | Same token-overlap scoring ("engineering building" -> Engineering Building Additions) |
| Project abbreviations | Initials matching ("AG" ~ Project Alpha Gamma) at reduced confidence, never auto-applied |
| Similar project numbers | Numbers match by exact equality only - 991 never matches 9911 |
| Files associated with the wrong project | Human-confirmed assignment, one-click Undo on auto-assign, unassign/reassign endpoints, visible AUTO tag for auditability |
| Missing metadata | Every registry field nullable; citations and cards degrade gracefully (set/version shown only "when applicable"); files with no signals simply get no suggestions |
| Duplicate files | Embedding-based duplicate detection across formats (PNG of a DXF still flags), compare-side-by-side modal, delete-from-compare |
| Duplicate drawing numbers | Version groups model this explicitly; auto-assign REFUSES when a DWG number maps to more than one drawing - picking a version is a human decision |
| Multiple versions of one drawing | Version-aware retrieval: answers name the version used, disclose siblings, and never blend versions' content (single-file evidence rule) |
| Drawings in a set | Set membership joined into evidence/citations |
| Drawings with no set number | Nullable; UI shows an em dash, citations omit the set line |
| Unclear/incorrect filenames | Content-based matching reads DWG/project signals from the extracted title block ("IMG_0001.dxf" identified from its content, verified) |
| Conflicting dates/years/versions | Raw date string preserved alongside best-effort parsed year; version notes carried; answers disclose the version instead of silently resolving conflicts |
| Low-quality PDFs/images | Enhancement pipeline before vision: EXIF orientation fix, autocontrast + sharpen for faint scans, upscale for tiny images; per-region confidence + HITL review as the safety net |
| Unsupported/unreadable files | Extension allowlist with actionable guidance for DWG (ODA converter route) and RVT (export guidance); corrupt files -> clean 422; failed extractions keep the row with the error and a Retry |
| Multi-drawing questions | Multi-drawing responses: per-drawing grouped retrieval, attributed answer, all sources cited |

## Discovered beyond the requirement

| Edge case | Behavior |
|---|---|
| Wedged vision service (connection accepted, never serviced) | Absolute deadline on top of socket timeouts; row -> failed + Retry (observed in the wild: 7h hang before the fix) |
| One slow upload starving the whole API | All blocking work moved off the event loop (verified: 5-17 ms responses during a 28 s extraction) |
| Vision models returning coordinates in arbitrary scales | Scale detection (fraction/percent/absolute) + reversed-corner normalization + known-size downscale |
| Sideways phone photos | EXIF transpose applied identically in extraction and preview so bboxes align (verified with a rotated JPEG) |
| Registry vocabulary hijacking retrieval | Registry answers require a +0.12 dominance margin over content (a materials question was being answered by the "Set 12A" card before this) |
| Terse follow-up questions ("and its part number?") | Conversation carried into generation; retrieval retries with previous-question context when the follow-up alone scores below the relevance floor |
| Feedback double-counting | Re-rating applies only the delta; weights clamped [0.3, 2.0] |
| Folder moved into its own subtree | Cycle check -> 422; move dialogs exclude the subtree |
| Deleting a folder with contents | Recursive delete removes subtree files properly (storage objects + chunks), with a warning dialog |
| Orphaned storage objects from failed/deleted uploads | Delete paths clean storage; failure paths keep the row (status failed) so nothing dangles silently |
| Path-traversal-style filenames (`../../evil/../weird.dxf`) | Sanitized to basename with control/path characters stripped (verified) |
| Unbounded query input (100k-char questions) | 2000-char cap + bounded top_k (verified 422) |
| Duplicate uploads mid-queue | Sequential upload queue (protects the local vision model); queue survives navigation |
| Chat evidence outliving deleted files | Evidence is persisted per message; the viewer degrades when the file is gone rather than breaking the transcript |
| Off-topic questions | Relevance floor -> honest "couldn't find" with zero fabricated citations |
