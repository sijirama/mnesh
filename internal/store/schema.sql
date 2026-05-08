CREATE TABLE IF NOT EXISTS command_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  cmd TEXT NOT NULL,
  cwd TEXT NOT NULL,
  shell TEXT NOT NULL,
  hostname TEXT NOT NULL,
  exit_code INTEGER NOT NULL DEFAULT 0,
  source TEXT NOT NULL DEFAULT 'shell',
  created_at TEXT NOT NULL,
  git_branch TEXT NOT NULL DEFAULT '',
  model_version TEXT NOT NULL DEFAULT '',
  accepted_suggestion INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_command_events_created_at
ON command_events(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_command_events_session_id
ON command_events(session_id, created_at DESC);
