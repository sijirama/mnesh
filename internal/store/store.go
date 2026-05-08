package store

import (
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"os/exec"
	"slices"
	"time"
)

//go:embed schema.sql
var schemaSQL string

type CommandEvent struct {
	SessionID          string    `json:"session_id"`
	Command            string    `json:"cmd"`
	Cwd                string    `json:"cwd"`
	Shell              string    `json:"shell"`
	Hostname           string    `json:"hostname"`
	ExitCode           int       `json:"exit_code"`
	Source             string    `json:"source"`
	CreatedAt          time.Time `json:"created_at"`
	GitBranch          string    `json:"git_branch"`
	ModelVersion       string    `json:"model_version"`
	AcceptedSuggestion bool      `json:"accepted_suggestion"`
}

func EnsureSchema(ctx context.Context, dbPath string) error {
	payload := map[string]string{
		"db_path": dbPath,
		"schema":  schemaSQL,
	}
	return runPythonSQLite(ctx, ensureSchemaScript, payload)
}

func InsertCommandEvent(ctx context.Context, dbPath string, event CommandEvent) error {
	if event.CreatedAt.IsZero() {
		event.CreatedAt = time.Now().UTC()
	}
	payload := map[string]any{
		"db_path": dbPath,
		"event":   event,
	}
	return runPythonSQLite(ctx, insertEventScript, payload)
}

func ListRecentCommandEvents(ctx context.Context, dbPath string, limit int) ([]CommandEvent, error) {
	payload := map[string]any{
		"db_path": dbPath,
		"limit":   limit,
	}

	out, err := runPythonSQLiteOutput(ctx, listRecentEventsScript, payload)
	if err != nil {
		return nil, err
	}

	var events []CommandEvent
	if err := json.Unmarshal(out, &events); err != nil {
		return nil, fmt.Errorf("decode recent events: %w", err)
	}
	return events, nil
}

func LatestSessionID(ctx context.Context, dbPath string) (string, error) {
	payload := map[string]any{
		"db_path": dbPath,
	}

	out, err := runPythonSQLiteOutput(ctx, latestSessionIDScript, payload)
	if err != nil {
		return "", err
	}

	var resp struct {
		SessionID string `json:"session_id"`
	}
	if err := json.Unmarshal(out, &resp); err != nil {
		return "", fmt.Errorf("decode latest session id: %w", err)
	}
	return resp.SessionID, nil
}

func ListSessionWindow(ctx context.Context, dbPath, sessionID string, limit int) ([]CommandEvent, error) {
	payload := map[string]any{
		"db_path":    dbPath,
		"session_id": sessionID,
		"limit":      limit,
	}

	out, err := runPythonSQLiteOutput(ctx, listSessionWindowScript, payload)
	if err != nil {
		return nil, err
	}

	var events []CommandEvent
	if err := json.Unmarshal(out, &events); err != nil {
		return nil, fmt.Errorf("decode session window: %w", err)
	}
	slices.Reverse(events)
	return events, nil
}

func runPythonSQLite(ctx context.Context, script string, payload any) error {
	_, err := runPythonSQLiteOutput(ctx, script, payload)
	return err
}

func runPythonSQLiteOutput(ctx context.Context, script string, payload any) ([]byte, error) {
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshal sqlite payload: %w", err)
	}

	cmd := exec.CommandContext(ctx, "python3", "-c", script, string(body))
	out, err := cmd.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("python sqlite helper failed: %w: %s", err, string(out))
	}
	return out, nil
}

const ensureSchemaScript = `
import json, sqlite3, sys
payload = json.loads(sys.argv[1])
conn = sqlite3.connect(payload["db_path"])
try:
    conn.executescript(payload["schema"])
    conn.commit()
finally:
    conn.close()
print("ok")
`

const insertEventScript = `
import json, sqlite3, sys
payload = json.loads(sys.argv[1])
event = payload["event"]
conn = sqlite3.connect(payload["db_path"])
try:
    conn.execute(
        """
        INSERT INTO command_events (
            session_id, cmd, cwd, shell, hostname, exit_code, source,
            created_at, git_branch, model_version, accepted_suggestion
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event["session_id"],
            event["cmd"],
            event["cwd"],
            event["shell"],
            event["hostname"],
            event["exit_code"],
            event["source"],
            event["created_at"],
            event["git_branch"],
            event["model_version"],
            1 if event["accepted_suggestion"] else 0,
        ),
    )
    conn.commit()
finally:
    conn.close()
print("ok")
`

const listRecentEventsScript = `
import json, sqlite3, sys
payload = json.loads(sys.argv[1])
conn = sqlite3.connect(payload["db_path"])
conn.row_factory = sqlite3.Row
try:
    rows = conn.execute(
        """
        SELECT session_id, cmd, cwd, shell, hostname, exit_code, source,
               created_at, git_branch, model_version, accepted_suggestion
        FROM command_events
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (payload["limit"],),
    ).fetchall()
    result = []
    for row in rows:
        result.append({
            "session_id": row["session_id"],
            "cmd": row["cmd"],
            "cwd": row["cwd"],
            "shell": row["shell"],
            "hostname": row["hostname"],
            "exit_code": row["exit_code"],
            "source": row["source"],
            "created_at": row["created_at"],
            "git_branch": row["git_branch"],
            "model_version": row["model_version"],
            "accepted_suggestion": bool(row["accepted_suggestion"]),
        })
finally:
    conn.close()
print(json.dumps(result))
`

const latestSessionIDScript = `
import json, sqlite3, sys
payload = json.loads(sys.argv[1])
conn = sqlite3.connect(payload["db_path"])
conn.row_factory = sqlite3.Row
try:
    row = conn.execute(
        """
        SELECT session_id
        FROM command_events
        ORDER BY created_at DESC
        LIMIT 1
        """
    ).fetchone()
    print(json.dumps({"session_id": row["session_id"] if row else ""}))
finally:
    conn.close()
`

const listSessionWindowScript = `
import json, sqlite3, sys
payload = json.loads(sys.argv[1])
conn = sqlite3.connect(payload["db_path"])
conn.row_factory = sqlite3.Row
try:
    rows = conn.execute(
        """
        SELECT session_id, cmd, cwd, shell, hostname, exit_code, source,
               created_at, git_branch, model_version, accepted_suggestion
        FROM command_events
        WHERE session_id = ?
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (payload["session_id"], payload["limit"]),
    ).fetchall()
    result = []
    for row in rows:
        result.append({
            "session_id": row["session_id"],
            "cmd": row["cmd"],
            "cwd": row["cwd"],
            "shell": row["shell"],
            "hostname": row["hostname"],
            "exit_code": row["exit_code"],
            "source": row["source"],
            "created_at": row["created_at"],
            "git_branch": row["git_branch"],
            "model_version": row["model_version"],
            "accepted_suggestion": bool(row["accepted_suggestion"]),
        })
finally:
    conn.close()
print(json.dumps(result))
`
