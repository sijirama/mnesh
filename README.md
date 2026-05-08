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
```

`mnesh init` creates `~/.mnesh/`, writes a default config, touches the sqlite path, and downloads the published `v5` and `v6` model bundles from hugging face.

## useful resources

- https://www.sciencedirect.com/science/article/pii/S2352340921006806?via%3Dihub
- https://is.muni.cz/publication/1783801/2021-FIE-toolset-collecting-shell-commands-its-application-hands-on-cybersecurity-training-paper.pdf
