package hooks

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

func SupportedShells() []string {
	return []string{"zsh", "bash"}
}

func Render(shell, binPath string) (string, error) {
	switch shell {
	case "zsh":
		return zshHook(binPath), nil
	case "bash":
		return bashHook(binPath), nil
	default:
		return "", fmt.Errorf("unsupported shell %q", shell)
	}
}

func FileName(shell string) (string, error) {
	switch shell {
	case "zsh":
		return "mnesh.zsh", nil
	case "bash":
		return "mnesh.bash", nil
	default:
		return "", fmt.Errorf("unsupported shell %q", shell)
	}
}

func Write(dir, shell, binPath string) (string, error) {
	body, err := Render(shell, binPath)
	if err != nil {
		return "", err
	}
	name, err := FileName(shell)
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	target := filepath.Join(dir, name)
	if err := os.WriteFile(target, []byte(body), 0o644); err != nil {
		return "", err
	}
	return target, nil
}

func zshHook(binPath string) string {
	const tmpl = `# mnesh zsh hook
if [[ -z "${MNESH_SESSION_ID:-}" ]]; then
  export MNESH_SESSION_ID="sess-${EPOCHREALTIME//./}-${$}-${RANDOM}"
fi

if [[ -z "${MNESH_BIN:-}" ]]; then
  export MNESH_BIN=__MNESH_BIN__
fi

_mnesh_git_branch() {
  command git rev-parse --abbrev-ref HEAD 2>/dev/null || true
}

_mnesh_preexec() {
  MNESH_LAST_CMD="$1"
  MNESH_LAST_CWD="$PWD"
}

_mnesh_extract_suggestion() {
  local payload="$1"
  if [[ -z "$payload" ]]; then
    return 0
  fi
  python3 - "$payload" <<'PY'
import json
import sys

try:
    payload = json.loads(sys.argv[1])
except Exception:
    print("")
    raise SystemExit(0)

print(payload.get("suggestion", ""))
PY
}

_mnesh_predict_widget() {
  zle -M "mnesh predicting..."
  zle -R

  local payload suggestion
  payload="$("$MNESH_BIN" predict --limit 10 2>/dev/null)"
  suggestion="$(_mnesh_extract_suggestion "$payload")"

  if [[ -z "$suggestion" ]]; then
    zle -M "mnesh: no suggestion"
    return 0
  fi

  BUFFER="$suggestion"
  CURSOR=${#BUFFER}
  zle -M ""
  zle redisplay
}

_mnesh_precmd() {
  local exit_code=$?
  local cmd="${MNESH_LAST_CMD:-}"
  local cwd="${MNESH_LAST_CWD:-$PWD}"

  if [[ -z "$cmd" ]]; then
    return
  fi
  if [[ "$cmd" == "$MNESH_BIN"* ]]; then
    MNESH_LAST_CMD=""
    return
  fi

  local git_branch
  git_branch="$(_mnesh_git_branch)"

  "$MNESH_BIN" record \
    --cmd "$cmd" \
    --cwd "$cwd" \
    --shell zsh \
    --session-id "$MNESH_SESSION_ID" \
    --hostname "${HOST:-localhost}" \
    --exit-code "$exit_code" \
    --git-branch "$git_branch" \
    >/dev/null 2>&1

  MNESH_LAST_CMD=""
}

autoload -Uz add-zsh-hook
add-zsh-hook preexec _mnesh_preexec
add-zsh-hook precmd _mnesh_precmd
zle -N _mnesh_predict_widget
bindkey '^[p' _mnesh_predict_widget
`
	return strings.ReplaceAll(tmpl, "__MNESH_BIN__", strconv.Quote(binPath))
}

func bashHook(binPath string) string {
	const tmpl = `# mnesh bash hook
if [[ -z "${MNESH_SESSION_ID:-}" ]]; then
  export MNESH_SESSION_ID="sess-$(date +%s)-$$-$RANDOM"
fi

if [[ -z "${MNESH_BIN:-}" ]]; then
  export MNESH_BIN=__MNESH_BIN__
fi

_mnesh_git_branch() {
  command git rev-parse --abbrev-ref HEAD 2>/dev/null || true
}

_mnesh_capture_command() {
  local cmd
  cmd="$(history 1 | sed 's/^[ ]*[0-9]\+[ ]*//')"
  if [[ -n "$cmd" ]]; then
    MNESH_LAST_CMD="$cmd"
    MNESH_LAST_CWD="$PWD"
  fi
}

_mnesh_precmd() {
  local exit_code=$?
  local cmd="${MNESH_LAST_CMD:-}"
  local cwd="${MNESH_LAST_CWD:-$PWD}"

  if [[ -z "$cmd" ]]; then
    return
  fi
  if [[ "$cmd" == "$MNESH_BIN"* ]]; then
    MNESH_LAST_CMD=""
    return
  fi

  local git_branch
  git_branch="$(_mnesh_git_branch)"

  "$MNESH_BIN" record \
    --cmd "$cmd" \
    --cwd "$cwd" \
    --shell bash \
    --session-id "$MNESH_SESSION_ID" \
    --hostname "${HOSTNAME:-localhost}" \
    --exit-code "$exit_code" \
    --git-branch "$git_branch" \
    >/dev/null 2>&1

  MNESH_LAST_CMD=""
}

trap '_mnesh_capture_command' DEBUG
PROMPT_COMMAND="_mnesh_precmd${PROMPT_COMMAND:+;$PROMPT_COMMAND}"
`
	return strings.ReplaceAll(tmpl, "__MNESH_BIN__", strconv.Quote(binPath))
}
