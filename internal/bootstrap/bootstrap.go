package bootstrap

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/sijirama/mnesh/internal/hooks"
	"github.com/sijirama/mnesh/internal/llm"
	"github.com/sijirama/mnesh/internal/mneshfs"
	"github.com/sijirama/mnesh/internal/python"
	"github.com/sijirama/mnesh/internal/store"
)

type Options struct {
	SkipDownloads bool
}

type Config struct {
	ActiveModel      string     `json:"active_model"`
	DefaultModel     string     `json:"default_model"`
	InferenceBackend string     `json:"inference_backend"`
	DBPath           string     `json:"db_path"`
	ModelsDir        string     `json:"models_dir"`
	LogsDir          string     `json:"logs_dir"`
	CacheDir         string     `json:"cache_dir"`
	InstalledModels  []string   `json:"installed_models"`
	LLM              llm.Config `json:"llm"`
}

type modelSpec struct {
	Name  string
	Files []string
}

var modelCatalog = []modelSpec{
	{
		Name: "v5",
		Files: []string{
			"mnesh_best.pt",
			"mnesh_unigram.model",
			"mnesh_unigram.vocab",
			"VERSION",
			"metadata.json",
		},
	},
	{
		Name: "v6",
		Files: []string{
			"mnesh_best.pt",
			"mnesh_unigram.model",
			"mnesh_unigram.vocab",
			"VERSION",
			"metadata.json",
		},
	},
}

func Init(ctx context.Context, opts Options) error {
	paths, err := mneshfs.Resolve()
	if err != nil {
		return err
	}

	fmt.Println("1/10 creating local directories...")
	if err := ensureDirs(paths); err != nil {
		return err
	}
	fmt.Printf("   ok: %s\n", paths.Root)

	fmt.Println("2/10 preparing sqlite database...")
	if err := touch(paths.DBPath); err != nil {
		return fmt.Errorf("create commands db placeholder: %w", err)
	}
	if err := store.EnsureSchema(ctx, paths.DBPath); err != nil {
		return fmt.Errorf("initialize sqlite schema: %w", err)
	}
	fmt.Printf("   ok: %s\n", paths.DBPath)

	fmt.Println("3/10 writing config...")
	if err := writeDefaultConfig(paths); err != nil {
		return err
	}
	fmt.Printf("   ok: %s\n", paths.ConfigPath)

	fmt.Println("4/10 setting active model...")
	if err := os.WriteFile(paths.ActiveModelPath, []byte("qwen\n"), 0o644); err != nil {
		return fmt.Errorf("write active model marker: %w", err)
	}
	fmt.Printf("   ok: %s -> qwen\n", paths.ActiveModelPath)

	fmt.Println("5/10 writing shell hook files...")
	for _, shell := range hooks.SupportedShells() {
		if _, err := hooks.Write(paths.HooksDir, shell, paths.BinPath); err != nil {
			return fmt.Errorf("write %s hook: %w", shell, err)
		}
		fmt.Printf("   ok: %s/%s\n", paths.HooksDir, hookFileName(shell))
	}

	fmt.Println("6/10 installing local binary...")
	if err := installBinary(paths); err != nil {
		return err
	}
	fmt.Printf("   ok: %s\n", paths.BinPath)

	fmt.Println("7/10 materializing python runtime...")
	if err := python.Materialize(paths.PythonDir); err != nil {
		return fmt.Errorf("materialize python runtime: %w", err)
	}
	fmt.Printf("   ok: %s\n", paths.PredictWorkerPath)

	fmt.Println("8/10 writing llama service unit...")
	llmCfg := llm.DefaultConfig(paths)
	if err := llm.WriteUnit(paths, llmCfg); err != nil {
		return fmt.Errorf("write llama service unit: %w", err)
	}
	fmt.Printf("   ok: %s\n", paths.LLMServicePath)
	if llmCfg.ServerBin == "" {
		fmt.Println("   note: llama-server not found on PATH; doctor will report this until installed")
	} else if err := llm.DaemonReload(ctx); err != nil {
		fmt.Printf("   note: daemon-reload skipped: %v\n", err)
	} else {
		fmt.Println("   ok: systemd --user daemon-reload")
	}

	fmt.Printf("9/10 installing model bundles%s...\n", skipNote(opts.SkipDownloads))
	if !opts.SkipDownloads {
		for _, spec := range modelCatalog {
			fmt.Printf("   downloading %s...\n", spec.Name)
			if err := downloadModelBundle(ctx, paths.ModelsDir, spec); err != nil {
				return err
			}
			fmt.Printf("   ok: %s\n", filepath.Join(paths.ModelsDir, spec.Name))
		}
	} else {
		fmt.Println("   skipped downloads")
	}

	fmt.Printf("10/10 installing qwen gguf%s...\n", skipNote(opts.SkipDownloads))
	if !opts.SkipDownloads {
		if err := downloadQwenModel(ctx, paths); err != nil {
			return err
		}
		fmt.Printf("   ok: %s\n", paths.QwenModelPath)
	} else {
		fmt.Println("   skipped downloads")
	}

	fmt.Println("mnesh home initialized")
	fmt.Printf("root: %s\n", paths.Root)
	fmt.Printf("db:   %s\n", paths.DBPath)
	fmt.Printf("models: %s\n", paths.ModelsDir)
	return nil
}

func Doctor() error {
	paths, err := mneshfs.Resolve()
	if err != nil {
		return err
	}

	checks := []struct {
		label string
		path  string
	}{
		{"root", paths.Root},
		{"config", paths.ConfigPath},
		{"db", paths.DBPath},
		{"models", paths.ModelsDir},
		{"bin", paths.BinDir},
		{"binary", paths.BinPath},
		{"logs", paths.LogsDir},
		{"cache", paths.CacheDir},
		{"hooks", paths.HooksDir},
		{"active_model", paths.ActiveModelPath},
		{"python", paths.PythonDir},
		{"worker", paths.PredictWorkerPath},
		{"llm_service", paths.LLMServicePath},
	}

	for _, check := range checks {
		if _, err := os.Stat(check.path); err != nil {
			fmt.Printf("%-12s missing  %s\n", check.label, check.path)
			continue
		}
		fmt.Printf("%-12s ok      %s\n", check.label, check.path)
	}

	activeModel := "unknown"
	if raw, err := os.ReadFile(paths.ActiveModelPath); err == nil {
		activeModel = strings.TrimSpace(string(raw))
	}
	fmt.Printf("%-12s %s\n", "active", activeModel)

	for _, modelName := range []string{"v5", "v6"} {
		modelPath := filepath.Join(paths.ModelsDir, modelName, "mnesh_best.pt")
		if _, err := os.Stat(modelPath); err == nil {
			fmt.Printf("%-12s ok      %s\n", "model:"+modelName, modelPath)
		} else {
			fmt.Printf("%-12s missing  %s\n", "model:"+modelName, modelPath)
		}
	}

	if _, err := os.Stat(paths.QwenModelPath); err == nil {
		fmt.Printf("%-12s ok      %s\n", "model:qwen", paths.QwenModelPath)
	} else {
		fmt.Printf("%-12s missing  %s\n", "model:qwen", paths.QwenModelPath)
	}

	cfg, _ := ReadConfig(paths)
	llmCfg := llm.MergeConfig(paths, cfg.LLM)
	if llmCfg.ServerBin == "" {
		fmt.Printf("%-12s missing  llama-server\n", "llama_bin")
	} else if _, err := os.Stat(llmCfg.ServerBin); err == nil {
		fmt.Printf("%-12s ok      %s\n", "llama_bin", llmCfg.ServerBin)
	} else if resolved, err := exec.LookPath(llmCfg.ServerBin); err == nil {
		fmt.Printf("%-12s ok      %s\n", "llama_bin", resolved)
	} else {
		fmt.Printf("%-12s missing  %s\n", "llama_bin", llmCfg.ServerBin)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	llmStatus := llm.StatusCheck(ctx, paths, llmCfg)
	fmt.Printf("%-12s %v\n", "llm_active", llmStatus.IsActive)
	fmt.Printf("%-12s %v\n", "llm_enabled", llmStatus.IsEnabled)
	if llmStatus.HealthOK {
		fmt.Printf("%-12s ok      %s\n", "llm_health", llmStatus.HealthStatus)
	} else {
		fmt.Printf("%-12s down    %s\n", "llm_health", llmStatus.HealthStatus)
	}

	for _, shell := range hooks.SupportedShells() {
		rcPath, err := shellRCPath(paths.Root, shell)
		if err != nil {
			continue
		}
		hookPath := filepath.Join(paths.HooksDir, hookFileName(shell))
		sourceLine := sourceLineForHook(hookPath)
		pathLine := pathLineForBin(paths.BinDir)
		content, readErr := os.ReadFile(rcPath)
		if readErr != nil {
			fmt.Printf("%-12s missing  %s\n", shell+":rc", rcPath)
			continue
		}
		text := string(content)
		if strings.Contains(text, sourceLine) && strings.Contains(text, pathLine) {
			fmt.Printf("%-12s ok      %s\n", shell+":rc", rcPath)
		} else {
			fmt.Printf("%-12s partial %s\n", shell+":rc", rcPath)
		}
	}
	return nil
}

func Daemon() error {
	paths, err := mneshfs.Resolve()
	if err != nil {
		return err
	}

	active := "unknown"
	if raw, err := os.ReadFile(paths.ActiveModelPath); err == nil {
		active = string(raw)
	}

	fmt.Println("mnesh daemon bootstrap")
	fmt.Printf("home:         %s\n", paths.Root)
	fmt.Printf("database:     %s\n", paths.DBPath)
	fmt.Printf("active model: %s", active)
	fmt.Println("status:       shell capture and dashboard bootstrap pending")
	return nil
}

func ensureDirs(paths mneshfs.Paths) error {
	dirs := []string{
		paths.Root,
		paths.DataDir,
		paths.ModelsDir,
		paths.BinDir,
		paths.LogsDir,
		paths.CacheDir,
		paths.HooksDir,
	}
	for _, dir := range dirs {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return fmt.Errorf("mkdir %s: %w", dir, err)
		}
	}
	return nil
}

func writeDefaultConfig(paths mneshfs.Paths) error {
	cfg := Config{
		ActiveModel:      "qwen",
		DefaultModel:     "qwen",
		InferenceBackend: "llama-server",
		DBPath:           paths.DBPath,
		ModelsDir:        paths.ModelsDir,
		LogsDir:          paths.LogsDir,
		CacheDir:         paths.CacheDir,
		InstalledModels:  []string{"qwen", "v5", "v6"},
		LLM:              llm.DefaultConfig(paths),
	}

	body, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal config: %w", err)
	}

	if _, err := os.Stat(paths.ConfigPath); err == nil {
		return nil
	}
	if err := os.WriteFile(paths.ConfigPath, append(body, '\n'), 0o644); err != nil {
		return fmt.Errorf("write config: %w", err)
	}
	return nil
}

func ReadConfig(paths mneshfs.Paths) (Config, error) {
	var cfg Config
	body, err := os.ReadFile(paths.ConfigPath)
	if err != nil {
		return cfg, err
	}
	if err := json.Unmarshal(body, &cfg); err != nil {
		return cfg, err
	}
	return cfg, nil
}

func touch(path string) error {
	if _, err := os.Stat(path); err == nil {
		return nil
	}
	file, err := os.OpenFile(path, os.O_CREATE|os.O_RDONLY, 0o644)
	if err != nil {
		return err
	}
	return file.Close()
}

func downloadModelBundle(ctx context.Context, modelsDir string, spec modelSpec) error {
	targetDir := filepath.Join(modelsDir, spec.Name)
	if err := os.MkdirAll(targetDir, 0o755); err != nil {
		return fmt.Errorf("mkdir model dir %s: %w", targetDir, err)
	}

	client := &http.Client{Timeout: 30 * time.Minute}
	for _, fileName := range spec.Files {
		targetPath := filepath.Join(targetDir, fileName)
		if _, err := os.Stat(targetPath); err == nil {
			continue
		}
		url := fmt.Sprintf("https://huggingface.co/sijirama/mnesh-%s/resolve/main/%s", spec.Name, fileName)
		if err := downloadFile(ctx, client, url, targetPath); err != nil {
			return fmt.Errorf("download %s for %s: %w", fileName, spec.Name, err)
		}
	}
	return nil
}

func downloadQwenModel(ctx context.Context, paths mneshfs.Paths) error {
	if err := os.MkdirAll(paths.QwenDir, 0o755); err != nil {
		return fmt.Errorf("mkdir qwen model dir %s: %w", paths.QwenDir, err)
	}
	if _, err := os.Stat(paths.QwenModelPath); err == nil {
		return nil
	}
	client := &http.Client{Timeout: 90 * time.Minute}
	url := fmt.Sprintf("https://huggingface.co/%s/resolve/main/%s", llm.DefaultRepoID, llm.DefaultFileName)
	if err := downloadFile(ctx, client, url, paths.QwenModelPath); err != nil {
		return fmt.Errorf("download qwen gguf: %w", err)
	}
	return nil
}

func downloadFile(ctx context.Context, client *http.Client, url, targetPath string) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}

	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("unexpected status %s", resp.Status)
	}

	tmpPath := targetPath + ".part"
	file, err := os.Create(tmpPath)
	if err != nil {
		return err
	}
	if _, err := io.Copy(file, resp.Body); err != nil {
		file.Close()
		return err
	}
	if err := file.Close(); err != nil {
		return err
	}
	return os.Rename(tmpPath, targetPath)
}

func installBinary(paths mneshfs.Paths) error {
	exePath, err := os.Executable()
	if err != nil {
		return fmt.Errorf("resolve current executable: %w", err)
	}

	src, err := os.Open(exePath)
	if err != nil {
		return fmt.Errorf("open current executable: %w", err)
	}
	defer src.Close()

	if err := os.MkdirAll(paths.BinDir, 0o755); err != nil {
		return fmt.Errorf("mkdir bin dir: %w", err)
	}

	tmpPath := paths.BinPath + ".tmp"
	dst, err := os.Create(tmpPath)
	if err != nil {
		return fmt.Errorf("create temp binary: %w", err)
	}
	if _, err := io.Copy(dst, src); err != nil {
		dst.Close()
		return fmt.Errorf("copy executable: %w", err)
	}
	if err := dst.Close(); err != nil {
		return fmt.Errorf("close temp binary: %w", err)
	}
	if err := os.Chmod(tmpPath, 0o755); err != nil {
		return fmt.Errorf("chmod temp binary: %w", err)
	}
	if err := os.Rename(tmpPath, paths.BinPath); err != nil {
		if errors.Is(err, os.ErrPermission) {
			return fmt.Errorf("install binary into %s: %w", paths.BinPath, err)
		}
		return fmt.Errorf("move binary into place: %w", err)
	}
	return nil
}

func hookFileName(shell string) string {
	switch shell {
	case "zsh":
		return "mnesh.zsh"
	case "bash":
		return "mnesh.bash"
	default:
		return shell
	}
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

func skipNote(skip bool) string {
	if skip {
		return " (skip-downloads)"
	}
	return ""
}
