package bootstrap

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/sijirama/mnesh/internal/mneshfs"
	"github.com/sijirama/mnesh/internal/store"
)

type Options struct {
	SkipDownloads bool
}

type Config struct {
	ActiveModel      string   `json:"active_model"`
	DefaultModel     string   `json:"default_model"`
	InferenceBackend string   `json:"inference_backend"`
	DBPath           string   `json:"db_path"`
	ModelsDir        string   `json:"models_dir"`
	LogsDir          string   `json:"logs_dir"`
	CacheDir         string   `json:"cache_dir"`
	InstalledModels  []string `json:"installed_models"`
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

	if err := ensureDirs(paths); err != nil {
		return err
	}
	if err := touch(paths.DBPath); err != nil {
		return fmt.Errorf("create commands db placeholder: %w", err)
	}
	if err := store.EnsureSchema(ctx, paths.DBPath); err != nil {
		return fmt.Errorf("initialize sqlite schema: %w", err)
	}
	if err := writeDefaultConfig(paths); err != nil {
		return err
	}
	if err := os.WriteFile(paths.ActiveModelPath, []byte("v5\n"), 0o644); err != nil {
		return fmt.Errorf("write active model marker: %w", err)
	}

	if !opts.SkipDownloads {
		for _, spec := range modelCatalog {
			if err := downloadModelBundle(ctx, paths.ModelsDir, spec); err != nil {
				return err
			}
		}
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
		{"logs", paths.LogsDir},
		{"cache", paths.CacheDir},
		{"hooks", paths.HooksDir},
		{"active_model", paths.ActiveModelPath},
	}

	for _, check := range checks {
		if _, err := os.Stat(check.path); err != nil {
			fmt.Printf("%-12s missing  %s\n", check.label, check.path)
			continue
		}
		fmt.Printf("%-12s ok      %s\n", check.label, check.path)
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
		ActiveModel:      "v5",
		DefaultModel:     "v5",
		InferenceBackend: "python-worker",
		DBPath:           paths.DBPath,
		ModelsDir:        paths.ModelsDir,
		LogsDir:          paths.LogsDir,
		CacheDir:         paths.CacheDir,
		InstalledModels:  []string{"v5", "v6"},
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
