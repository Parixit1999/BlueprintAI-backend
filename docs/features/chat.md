# Chat persistence & sessions

`app/services/chat_service.py`, `routers/chats.py`, `ChatRepository`.

- **Sessions** are DB-backed and scoped to `user_id` (defaults to
  `settings.default_user_id = "global"` — single-user now; real auth just supplies
  real ids). Title auto-set from the first question.
- Every question runs the RAG pipeline (`QueryService.ask`); both the user turn
  and the assistant turn are stored, with the assistant's **evidence references**
  (`chat_messages.evidence` jsonb) so past answers stay verifiable.

## Endpoints
- `POST /chats` create · `GET /chats` list (with message counts) ·
  `GET /chats/{id}` messages · `POST /chats/{id}/messages` ask
- `PATCH /chats/{id}` rename (ownership-checked, trimmed, capped) ·
  `DELETE /chats/{id}` delete (messages cascade via FK)

All mutating ops verify the session belongs to the current user first (404 if not).
