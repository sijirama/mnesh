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

_mnesh_expand_alias() {
  local cmd="$1"
  local -a parts
  parts=(${(z)cmd})
  if (( ${#parts[@]} == 0 )); then
    printf '%s\n' "$cmd"
    return 0
  fi

  local head="${parts[1]}"
  local expanded="$head"
  local seen=0
  while [[ -n "${aliases[$expanded]:-}" && $seen -lt 5 ]]; do
    expanded="${aliases[$expanded]}"
    seen=$((seen + 1))
    local -a nested
    nested=(${(z)expanded})
    if (( ${#nested[@]} == 0 )); then
      break
    fi
    expanded="${nested[1]}"
  done

  if [[ -n "${aliases[$head]:-}" ]]; then
    parts[1]="${aliases[$head]}"
    printf '%s\n' "${(j: :)parts}"
    return 0
  fi

  printf '%s\n' "$cmd"
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
  cmd="$(_mnesh_expand_alias "$cmd")"
  local mnesh_name="${MNESH_BIN:t}"
  if [[ "$cmd" == "$MNESH_BIN"* || "$cmd" == "${mnesh_name}"* ]]; then
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
    MNESH_LAST_CMD="$(_mnesh_expand_alias "$cmd")"
    MNESH_LAST_CWD="$PWD"
  fi
}

_mnesh_expand_alias() {
  local cmd="$1"
  local head rest expanded line seen
  head="${cmd%% *}"
  if [[ -z "$head" ]]; then
    printf '%s\n' "$cmd"
    return 0
  fi
  rest="${cmd#"$head"}"
  expanded="$head"
  seen=0
  while [[ $seen -lt 5 ]]; do
    line="$(alias "$expanded" 2>/dev/null || true)"
    if [[ -z "$line" ]]; then
      break
    fi
    expanded="${line#*=}"
    expanded="${expanded#\'}"
    expanded="${expanded%\'}"
    expanded="${expanded#\"}"
    expanded="${expanded%\"}"
    seen=$((seen + 1))
    if [[ "$expanded" != *" "* ]]; then
      continue
    fi
    break
  done
  if [[ "$expanded" != "$head" ]]; then
    printf '%s%s\n' "$expanded" "$rest"
    return 0
  fi
  printf '%s\n' "$cmd"
}

_mnesh_precmd() {
  local exit_code=$?
  local cmd="${MNESH_LAST_CMD:-}"
  local cwd="${MNESH_LAST_CWD:-$PWD}"

  if [[ -z "$cmd" ]]; then
    return
  fi
  cmd="$(_mnesh_expand_alias "$cmd")"
  local mnesh_name
  mnesh_name="$(basename "$MNESH_BIN")"
  if [[ "$cmd" == "$MNESH_BIN"* || "$cmd" == "${mnesh_name}"* ]]; then
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
