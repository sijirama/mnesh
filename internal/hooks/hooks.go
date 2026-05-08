package hooks

import "fmt"

func Render(shell string) (string, error) {
	switch shell {
	case "zsh":
		return zshHook(), nil
	case "bash":
		return bashHook(), nil
	default:
		return "", fmt.Errorf("unsupported shell %q", shell)
	}
}

func zshHook() string {
	return `# mnesh zsh hook
if [[ -z "${MNESH_SESSION_ID:-}" ]]; then
  export MNESH_SESSION_ID="sess-${EPOCHREALTIME//./}-${$}-${RANDOM}"
fi

if [[ -z "${MNESH_BIN:-}" ]]; then
  export MNESH_BIN="mnesh"
fi

_mnesh_git_branch() {
  command git rev-parse --abbrev-ref HEAD 2>/dev/null || true
}

_mnesh_preexec() {
  MNESH_LAST_CMD="$1"
  MNESH_LAST_CWD="$PWD"
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
`
}

func bashHook() string {
	return `# mnesh bash hook
if [[ -z "${MNESH_SESSION_ID:-}" ]]; then
  export MNESH_SESSION_ID="sess-$(date +%s)-$$-$RANDOM"
fi

if [[ -z "${MNESH_BIN:-}" ]]; then
  export MNESH_BIN="mnesh"
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
}
