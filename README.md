# mnesh

mnesh is a local shell next-command predictor.

it learns from real and synthetic terminal sessions (commands, cwd, git state, environment context, etc.) and predicts what you’re likely to run next.

the goal is simple:
make the terminal feel like it understands your workflow.

it takes in structured session history — things like:

current directory
git branch
previous commands
exit codes
environment context

and outputs a suggested next command.

built initially with an rnn trained on synthetic shell telemetry, and now wired to a local qwen + `llama-server` runtime for the actual product path.

## install

quick install (downloads the latest published release into `~/.mnesh/bin/mnesh`):

```bash
curl -fsSL https://raw.githubusercontent.com/sijirama/mnesh/main/scripts/install.sh | bash
```

pin a version with `MNESH_VERSION`:

```bash
curl -fsSL https://raw.githubusercontent.com/sijirama/mnesh/main/scripts/install.sh | MNESH_VERSION=v0.2.0 bash
```

after the binary is installed, finish setup:

```bash
~/.mnesh/bin/mnesh init
~/.mnesh/bin/mnesh install-hook zsh   # or bash
exec zsh
~/.mnesh/bin/mnesh llm install
~/.mnesh/bin/mnesh llm start
~/.mnesh/bin/mnesh llm status
~/.mnesh/bin/mnesh doctor
```

to fully reset a local install (interactive — prompts for confirmation):

```bash
curl -fsSL https://raw.githubusercontent.com/sijirama/mnesh/main/scripts/uninstall.sh | bash
```

## runtime flow

the repo now includes a go bootstrap cli for the local runtime:

```bash
go run ./cmd/mnesh init
go run ./cmd/mnesh doctor
go run ./cmd/mnesh daemon
go run ./cmd/mnesh record --cmd "git status" --cwd "$PWD"
go run ./cmd/mnesh recent --limit 5
go run ./cmd/mnesh window --limit 10
go run ./cmd/mnesh predict --model v5 --limit 10
go run ./cmd/mnesh hook zsh
go run ./cmd/mnesh hook --write zsh
go run ./cmd/mnesh install-hook zsh
go run ./cmd/mnesh llm install
go run ./cmd/mnesh llm start
go run ./cmd/mnesh llm status
bash ./mnesh_uninstall
```

`mnesh init` creates `~/.mnesh/`, writes a default config, initializes sqlite, writes the zsh/bash hook files into `~/.mnesh/hooks/`, installs the current `mnesh` binary into `~/.mnesh/bin/`, writes a user-level `llama-server` systemd unit, downloads the published `v5` and `v6` model bundles from hugging face, and downloads the default qwen gguf into `~/.mnesh/models/qwen/`.

default local layout:

```text
~/.mnesh/
  bin/mnesh
  data/commands.db
  hooks/
  logs/
  cache/
  models/
    active
    qwen/qwen2.5-coder-0.5b-q4_k_m.gguf
    v5/
    v6/
  config.json
```

to enable shell capture, print a hook and source it in your shell config:

```bash
go run ./cmd/mnesh install-hook zsh
go run ./cmd/mnesh install-hook bash
```

that appends a safe PATH line plus one `source ~/.mnesh/hooks/...` line into your shell rc file.

the hook records commands into `~/.mnesh/data/commands.db` with a per-shell session id, cwd, exit code, host, and best-effort git branch.

for zsh, the installed hook also binds `Alt-p` to fetch a prediction and insert it into the current command line.

aliases are expanded before commands are stored, and `mnesh ...` commands are skipped from recording.

for the local qwen runtime, `mnesh` now also exposes:

```bash
go run ./cmd/mnesh llm install
go run ./cmd/mnesh llm start
go run ./cmd/mnesh llm stop
go run ./cmd/mnesh llm restart
go run ./cmd/mnesh llm status
```

the `llm install` command writes/enables a user systemd service for `llama-server`, and `doctor` checks the unit, model file, binary, and local health endpoint.

current default inference path:
- active model defaults to `qwen`
- `mnesh predict` calls the local `llama-server`
- `v5` and `v6` are still available explicitly with `--model v5` or `--model v6`

recommended local startup:

```bash
GOCACHE=/tmp/mnesh_gocache go build -o ./mnesh ./cmd/mnesh
./mnesh init
./mnesh install-hook zsh
exec zsh
mnesh llm install
mnesh llm start
mnesh llm status
mnesh doctor
```

to reset a local checkout, run:

```bash
bash ./mnesh_uninstall
```

## useful resources

- https://www.sciencedirect.com/science/article/pii/S2352340921006806?via%3Dihub
- https://is.muni.cz/publication/1783801/2021-FIE-toolset-collecting-shell-commands-its-application-hands-on-cybersecurity-training-paper.pdf
