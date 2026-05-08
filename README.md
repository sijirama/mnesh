# mnesh

mnesh is a next-command prediction engine for shell environments.

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

built initially with an rnn trained on large-scale synthetic shell telemetry, with plans for fine-tuning on personal command history.

## local service bootstrap

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
```

`mnesh init` creates `~/.mnesh/`, writes a default config, initializes sqlite, writes the zsh/bash hook files into `~/.mnesh/hooks/`, and downloads the published `v5` and `v6` model bundles from hugging face.

to enable shell capture, print a hook and source it in your shell config:

```bash
go run ./cmd/mnesh install-hook zsh
go run ./cmd/mnesh install-hook bash
```

that appends one safe `source ~/.mnesh/hooks/...` line into your shell rc file.

the hook records commands into `~/.mnesh/data/commands.db` with a per-shell session id, cwd, exit code, host, and best-effort git branch.

## useful resources

- https://www.sciencedirect.com/science/article/pii/S2352340921006806?via%3Dihub
- https://is.muni.cz/publication/1783801/2021-FIE-toolset-collecting-shell-commands-its-application-hands-on-cybersecurity-training-paper.pdf
