# Stats

`GET /stats` → `StatsRepository.snapshot()`. Powers the dashboard.

Returns: `documents_total`, `documents_by_status`, `documents_by_type`,
`chunks_total`, `chunks_by_confidence`, `chunks_corrected`, `chat_sessions`,
`questions_asked` (count of user chat messages).

Pure aggregate queries over `files`, `chunks`, `chat_sessions`, `chat_messages`.
The frontend renames "chunks" to "extracted regions" for users.
