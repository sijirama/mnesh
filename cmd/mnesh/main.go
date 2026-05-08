package main

import (
	"context"
	"fmt"
	"os"
	"time"

	"github.com/sijirama/mnesh/internal/bootstrap"
	"github.com/sijirama/mnesh/internal/mneshfs"
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
