package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/sijirama/mnesh/internal/bootstrap"
	"github.com/sijirama/mnesh/internal/mneshfs"
	"github.com/sijirama/mnesh/internal/store"
)

const version = "0.1.0"

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
	fmt.Println("  mnesh recent [--limit N]")
	fmt.Println("  mnesh window [--session-id <id>] [--limit N]")
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
	body, err := json.MarshalIndent(events, "", "  ")
	if err != nil {
		return err
	}
	fmt.Println(string(body))
	return nil
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

func hostOrLocal() string {
	host, err := os.Hostname()
	if err != nil || host == "" {
		return "localhost"
	}
	return host
}
