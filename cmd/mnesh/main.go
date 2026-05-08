package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/sijirama/mnesh/internal/bootstrap"
	"github.com/sijirama/mnesh/internal/hooks"
	"github.com/sijirama/mnesh/internal/mneshfs"
	"github.com/sijirama/mnesh/internal/store"
)

// version is overridden at build time via -ldflags "-X main.version=...".
var version = "dev"

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(1)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	switch os.Args[1] {
	case "init":
		skipDownloads := hasFlag(os.Args[2:], "--skip-downloads")
		if err := bootstrap.Init(ctx, bootstrap.Options{SkipDownloads: skipDownloads}); err != nil {
			fatal(err)
		}
	case "doctor":
		if err := bootstrap.Doctor(); err != nil {
			fatal(err)
		}
	case "daemon":
		if err := bootstrap.Daemon(); err != nil {
			fatal(err)
		}
	case "record":
		if err := runRecord(ctx, os.Args[2:]); err != nil {
			fatal(err)
		}
	case "recent":
		if err := runRecent(ctx, os.Args[2:]); err != nil {
			fatal(err)
		}
	case "window":
		if err := runWindow(ctx, os.Args[2:]); err != nil {
			fatal(err)
		}
	case "predict":
		if err := runPredict(ctx, os.Args[2:]); err != nil {
			fatal(err)
		}
	case "hook":
		if err := runHook(os.Args[2:]); err != nil {
			fatal(err)
		}
	case "install-hook":
		if err := runInstallHook(os.Args[2:]); err != nil {
			fatal(err)
		}
	case "version":
		fmt.Printf("mnesh %s\n", version)
	default:
		usage()
		os.Exit(1)
	}
}

func usage() {
	paths, _ := mneshfs.Resolve()
	fmt.Println("mnesh CLI")
	fmt.Println()
	fmt.Println("usage:")
	fmt.Println("  mnesh init [--skip-downloads]")
	fmt.Println("  mnesh doctor")
	fmt.Println("  mnesh daemon")
	fmt.Println("  mnesh record --cmd <command> [--cwd <dir>] [--shell <name>] [--session-id <id>]")
	fmt.Println("  mnesh recent [--limit N] [--json]")
	fmt.Println("  mnesh window [--session-id <id>] [--limit N]")
	fmt.Println("  mnesh predict [--model <v5|v6>] [--session-id <id>] [--limit N]")
	fmt.Println("  mnesh hook <zsh|bash>")
	fmt.Println("  mnesh install-hook <zsh|bash>")
	fmt.Println("  mnesh version")
	fmt.Println()
	fmt.Println("default home:")
	fmt.Printf("  %s\n", paths.Root)
}

func hasFlag(args []string, want string) bool {
	for _, arg := range args {
		if arg == want {
			return true
		}
	}
	return false
}

func fatal(err error) {
	fmt.Fprintf(os.Stderr, "error: %v\n", err)
	os.Exit(1)
}

func runRecord(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("record", flag.ContinueOnError)
	cmd := fs.String("cmd", "", "command text")
	cwd := fs.String("cwd", ".", "working directory")
	shell := fs.String("shell", "zsh", "shell name")
	sessionID := fs.String("session-id", fmt.Sprintf("sess-%d", time.Now().Unix()), "session identifier")
	hostname := fs.String("hostname", hostOrLocal(), "host name")
	exitCode := fs.Int("exit-code", 0, "command exit code")
	source := fs.String("source", "shell", "event source")
	gitBranch := fs.String("git-branch", "", "git branch")
	modelVersion := fs.String("model-version", "", "associated model version")
	acceptedSuggestion := fs.Bool("accepted-suggestion", false, "whether user accepted a model suggestion")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *cmd == "" {
		return fmt.Errorf("--cmd is required")
	}

	paths, err := mneshfs.Resolve()
	if err != nil {
		return err
	}
	event := store.CommandEvent{
		SessionID:          *sessionID,
		Command:            *cmd,
		Cwd:                *cwd,
		Shell:              *shell,
		Hostname:           *hostname,
		ExitCode:           *exitCode,
		Source:             *source,
		CreatedAt:          time.Now().UTC(),
		GitBranch:          *gitBranch,
		ModelVersion:       *modelVersion,
		AcceptedSuggestion: *acceptedSuggestion,
	}
	if err := store.InsertCommandEvent(ctx, paths.DBPath, event); err != nil {
		return err
	}
	fmt.Println("recorded command event")
	return nil
}

func runRecent(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("recent", flag.ContinueOnError)
	limit := fs.Int("limit", 10, "number of recent command events")
	asJSON := fs.Bool("json", false, "emit raw JSON instead of a formatted table")
	if err := fs.Parse(args); err != nil {
		return err
	}

	paths, err := mneshfs.Resolve()
	if err != nil {
		return err
	}
	events, err := store.ListRecentCommandEvents(ctx, paths.DBPath, *limit)
	if err != nil {
		return err
	}

	if *asJSON {
		body, err := json.MarshalIndent(events, "", "  ")
		if err != nil {
			return err
		}
		fmt.Println(string(body))
		return nil
	}

	if len(events) == 0 {
		fmt.Println("no recent commands")
		return nil
	}

	tw := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "TIME\tWHERE\tCMD")
	for _, e := range events {
		ts := e.CreatedAt.Local().Format("15:04:05")
		where := prettyCwd(e.Cwd)
		if e.GitBranch != "" {
			where = fmt.Sprintf("%s (%s)", where, e.GitBranch)
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\n", ts, where, e.Command)
	}
	return tw.Flush()
}

func prettyCwd(cwd string) string {
	if cwd == "" {
		return "-"
	}
	if home, err := os.UserHomeDir(); err == nil && home != "" {
		if cwd == home {
			return "~"
		}
		if strings.HasPrefix(cwd, home+string(os.PathSeparator)) {
			return "~" + cwd[len(home):]
		}
	}
	return cwd
}

func runWindow(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("window", flag.ContinueOnError)
	sessionID := fs.String("session-id", "", "session identifier")
	limit := fs.Int("limit", 10, "number of session commands")
	if err := fs.Parse(args); err != nil {
		return err
	}

	paths, err := mneshfs.Resolve()
	if err != nil {
		return err
	}

	id := *sessionID
	if id == "" {
		id, err = store.LatestSessionID(ctx, paths.DBPath)
		if err != nil {
			return err
		}
		if id == "" {
			fmt.Println("[]")
			return nil
		}
	}

	events, err := store.ListSessionWindow(ctx, paths.DBPath, id, *limit)
	if err != nil {
		return err
	}
	body, err := json.MarshalIndent(events, "", "  ")
	if err != nil {
		return err
	}
	fmt.Println(string(body))
	return nil
}

func runPredict(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("predict", flag.ContinueOnError)
	modelName := fs.String("model", "", "model name, e.g. v5 or v6")
	sessionID := fs.String("session-id", "", "session identifier")
	limit := fs.Int("limit", 10, "number of session commands")
	maxTokens := fs.Int("max-tokens", 32, "maximum generated tokens")
	if err := fs.Parse(args); err != nil {
		return err
	}

	paths, err := mneshfs.Resolve()
	if err != nil {
		return err
	}

	id := *sessionID
	if id == "" {
		id, err = store.LatestSessionID(ctx, paths.DBPath)
		if err != nil {
			return err
		}
		if id == "" {
			return fmt.Errorf("no recorded sessions available")
		}
	}

	events, err := store.ListSessionWindow(ctx, paths.DBPath, id, *limit)
	if err != nil {
		return err
	}
	if len(events) == 0 {
		return fmt.Errorf("no events found for session %s", id)
	}

	selectedModel, err := resolveModel(paths, *modelName)
	if err != nil {
		return err
	}

	workerPath := filepath.Join("scripts", "predict_worker.py")
	payload := map[string]any{
		"model_dir":  filepath.Join(paths.ModelsDir, selectedModel),
		"events":     events,
		"max_tokens": *maxTokens,
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}

	cmd := exec.CommandContext(ctx, resolvePython(), workerPath, string(body))
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("predict worker failed: %w: %s", err, string(out))
	}
	fmt.Println(string(out))
	return nil
}

func runHook(args []string) error {
	fs := flag.NewFlagSet("hook", flag.ContinueOnError)
	write := fs.Bool("write", false, "write hook file into ~/.mnesh/hooks")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 1 {
		return fmt.Errorf("usage: mnesh hook [--write] <zsh|bash>")
	}
	shell := fs.Arg(0)
	paths, err := mneshfs.Resolve()
	if err != nil {
		return err
	}
	body, err := hooks.Render(shell, paths.BinPath)
	if err != nil {
		return err
	}

	if *write {
		target, err := hooks.Write(paths.HooksDir, shell, paths.BinPath)
		if err != nil {
			return err
		}
		fmt.Println(target)
		return nil
	}

	fmt.Print(body)
	return nil
}

func runInstallHook(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: mnesh install-hook <zsh|bash>")
	}
	shell := args[0]
	paths, err := mneshfs.Resolve()
	if err != nil {
		return err
	}
	fmt.Printf("1/3 writing %s hook file...\n", shell)
	hookPath, err := hooks.Write(paths.HooksDir, shell, paths.BinPath)
	if err != nil {
		return err
	}
	fmt.Printf("   ok: %s\n", hookPath)
	pathLine := pathLineForBin(paths.BinDir)
	rcPath, err := shellRCPath(paths.Root, shell)
	if err != nil {
		return err
	}
	fmt.Printf("2/3 updating %s...\n", rcPath)
	sourceLine := sourceLineForHook(hookPath)
	existing, _ := os.ReadFile(rcPath)
	if strings.Contains(string(existing), sourceLine) && strings.Contains(string(existing), pathLine) {
		fmt.Printf("   ok: hook already installed in %s\n", rcPath)
		fmt.Println("3/3 shell restart required")
		fmt.Println("   run: exec zsh")
		return nil
	}
	file, err := os.OpenFile(rcPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer file.Close()
	if len(existing) > 0 && !strings.HasSuffix(string(existing), "\n") {
		if _, err := file.WriteString("\n"); err != nil {
			return err
		}
	}
	block := "\n# mnesh shell hook\n"
	if !strings.Contains(string(existing), pathLine) {
		block += pathLine + "\n"
	}
	if !strings.Contains(string(existing), sourceLine) {
		block += sourceLine + "\n"
	}
	if _, err := file.WriteString(block); err != nil {
		return err
	}
	fmt.Printf("   ok: installed hook into %s\n", rcPath)
	fmt.Println("3/3 shell restart required")
	fmt.Println("   run: exec zsh")
	return nil
}

func hostOrLocal() string {
	host, err := os.Hostname()
	if err != nil || host == "" {
		return "localhost"
	}
	return host
}

func resolveModel(paths mneshfs.Paths, requested string) (string, error) {
	if requested != "" {
		return requested, nil
	}

	raw, err := os.ReadFile(paths.ActiveModelPath)
	if err != nil {
		return "", fmt.Errorf("read active model: %w", err)
	}
	modelName := strings.TrimSpace(string(raw))
	if modelName == "" {
		return "", fmt.Errorf("active model marker is empty")
	}
	return modelName, nil
}

func resolvePython() string {
	if custom := strings.TrimSpace(os.Getenv("MNESH_PYTHON")); custom != "" {
		return custom
	}
	if _, err := os.Stat(filepath.Join(".venv", "bin", "python3")); err == nil {
		return filepath.Join(".venv", "bin", "python3")
	}
	return "python3"
}

func shellRCPath(root, shell string) (string, error) {
	home := filepath.Dir(root)
	switch shell {
	case "zsh":
		return filepath.Join(home, ".zshrc"), nil
	case "bash":
		return filepath.Join(home, ".bashrc"), nil
	default:
		return "", fmt.Errorf("unsupported shell %q", shell)
	}
}

func sourceLineForHook(hookPath string) string {
	return fmt.Sprintf("[[ -f %q ]] && source %q", hookPath, hookPath)
}

func pathLineForBin(binDir string) string {
	return fmt.Sprintf("export PATH=%q:$PATH", binDir)
}
