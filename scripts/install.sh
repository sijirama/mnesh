#!/usr/bin/env bash
# mnesh installer
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/sijirama/mnesh/main/scripts/install.sh | bash
#   curl -fsSL https://raw.githubusercontent.com/sijirama/mnesh/main/scripts/install.sh | MNESH_VERSION=v0.2.0 bash
#
# Honors:
#   MNESH_VERSION  pin to a specific tag (default: latest GitHub release)
#   MNESH_HOME     install dir (default: $HOME/.mnesh)
#   MNESH_REPO     owner/repo (default: sijirama/mnesh)
set -euo pipefail

REPO="${MNESH_REPO:-sijirama/mnesh}"
HOME_DIR="${MNESH_HOME:-${HOME}/.mnesh}"
BIN_DIR="${HOME_DIR}/bin"
BIN_PATH="${BIN_DIR}/mnesh"

print_banner() {
cat <<'EOF'
 __  __ _   _ _____ ____  _   _
|  \/  | \ | | ____/ ___|| | | |
| |\/| |  \| |  _| \___ \| |_| |
| |  | | |\  | |___ ___) |  _  |
|_|  |_|_| \_|_____|____/|_| |_|

           installer
EOF
}

die() { echo "error: $*" >&2; exit 1; }

require() {
  command -v "$1" >/dev/null 2>&1 || die "missing dependency: $1"
}

detect_os() {
  case "$(uname -s)" in
    Linux*)  echo "linux" ;;
    Darwin*) echo "darwin" ;;
    *)       die "unsupported OS: $(uname -s)" ;;
  esac
}

detect_arch() {
  case "$(uname -m)" in
    x86_64|amd64)  echo "amd64" ;;
    arm64|aarch64) echo "arm64" ;;
    *)             die "unsupported arch: $(uname -m)" ;;
  esac
}

resolve_version() {
  if [[ -n "${MNESH_VERSION:-}" ]]; then
    echo "${MNESH_VERSION}"
    return
  fi
  local api="https://api.github.com/repos/${REPO}/releases/latest"
  local tag
  tag="$(curl -fsSL "$api" | grep -oE '"tag_name":[[:space:]]*"[^"]+"' | head -n1 | sed -E 's/.*"([^"]+)"$/\1/')"
  [[ -n "$tag" ]] || die "could not resolve latest release from $api"
  echo "$tag"
}

main() {
  print_banner
  echo

  require curl
  require tar
  require uname

  local os arch version archive url tmp
  os="$(detect_os)"
  arch="$(detect_arch)"
  version="$(resolve_version)"
  archive="mnesh_${version}_${os}_${arch}.tar.gz"
  url="https://github.com/${REPO}/releases/download/${version}/${archive}"

  echo "installing mnesh ${version} (${os}/${arch})"
  echo "  source: ${url}"
  echo "  target: ${BIN_PATH}"
  echo

  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT

  curl -fL --progress-bar -o "${tmp}/${archive}" "$url" \
    || die "download failed for $url"

  tar -xzf "${tmp}/${archive}" -C "$tmp"

  local extracted="${tmp}/mnesh_${version}_${os}_${arch}/mnesh"
  [[ -f "$extracted" ]] || die "binary not found in archive at $extracted"

  mkdir -p "$BIN_DIR"
  install -m 0755 "$extracted" "$BIN_PATH"

  echo "ok: installed ${BIN_PATH}"
  echo
  "$BIN_PATH" version || true
  echo
  echo "next steps:"
  echo "  ${BIN_PATH} init"
  echo "  ${BIN_PATH} install-hook zsh   # or bash"
  echo
  echo "to use \`mnesh\` directly from your shell, add to PATH:"
  echo "  export PATH=\"${BIN_DIR}:\$PATH\""
}

main "$@"
