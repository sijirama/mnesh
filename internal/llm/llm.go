package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/sijirama/mnesh/internal/mneshfs"
	"github.com/sijirama/mnesh/internal/store"
)

const (
	DefaultRepoID      = "itlwas/Qwen2.5-Coder-0.5B-Q4_K_M-GGUF"
	DefaultFileName    = "qwen2.5-coder-0.5b-q4_k_m.gguf"
	DefaultHost        = "127.0.0.1"
	DefaultPort        = 8012
	DefaultContextSize = 4096
	ServiceName        = "mnesh-llama.service"
)

type Config struct {
	RepoID      string `json:"repo_id"`
	FileName    string `json:"file_name"`
	ModelPath   string `json:"model_path"`
	ServerBin   string `json:"server_bin"`
	Host        string `json:"host"`
	Port        int    `json:"port"`
	ContextSize int    `json:"context_size"`
}

type Status struct {
	ServiceFileExists bool
	ServerBinExists   bool
	ModelExists       bool
	SystemctlExists   bool
	IsEnabled         bool
	IsActive          bool
	HealthOK          bool
	HealthStatus      string
}

type Prediction struct {
	ModelVersion     string         `json:"model_version"`
	PredictedCmdType string         `json:"predicted_cmd_type"`
	TopCmdTypes      []string       `json:"top_cmd_types"`
	SessionContext   string         `json:"session_context"`
	Ecosystem        string         `json:"ecosystem"`
	Context          map[string]any `json:"context"`
	WindowCommands   []string       `json:"window_commands"`
	Suggestion       string         `json:"suggestion"`
}

func DefaultConfig(paths mneshfs.Paths) Config {
	return Config{
		RepoID:      DefaultRepoID,
		FileName:    DefaultFileName,
		ModelPath:   paths.QwenModelPath,
		ServerBin:   resolveServerBin(),
		Host:        DefaultHost,
		Port:        DefaultPort,
		ContextSize: DefaultContextSize,
	}
}

func MergeConfig(paths mneshfs.Paths, cfg Config) Config {
	def := DefaultConfig(paths)
	if strings.TrimSpace(cfg.RepoID) == "" {
		cfg.RepoID = def.RepoID
	}
	if strings.TrimSpace(cfg.FileName) == "" {
		cfg.FileName = def.FileName
	}
	if strings.TrimSpace(cfg.ModelPath) == "" {
		cfg.ModelPath = def.ModelPath
	}
	if strings.TrimSpace(cfg.ServerBin) == "" {
		cfg.ServerBin = def.ServerBin
	}
	if strings.TrimSpace(cfg.Host) == "" {
		cfg.Host = def.Host
	}
	if cfg.Port == 0 {
		cfg.Port = def.Port
	}
	if cfg.ContextSize == 0 {
		cfg.ContextSize = def.ContextSize
	}
	return cfg
}

func UnitContents(paths mneshfs.Paths, cfg Config) string {
	return fmt.Sprintf(`[Unit]
Description=mnesh local llama.cpp server
After=network.target

[Service]
Type=simple
ExecStart=%s -m %s --host %s --port %d -c %d
WorkingDirectory=%s
Restart=on-failure
RestartSec=3
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
`,
		cfg.ServerBin,
		cfg.ModelPath,
		cfg.Host,
		cfg.Port,
		cfg.ContextSize,
		paths.Root,
	)
}

func WriteUnit(paths mneshfs.Paths, cfg Config) error {
	if err := os.MkdirAll(paths.SystemdUserDir, 0o755); err != nil {
		return fmt.Errorf("mkdir systemd user dir: %w", err)
	}
	return os.WriteFile(paths.LLMServicePath, []byte(UnitContents(paths, cfg)), 0o644)
}

func DaemonReload(ctx context.Context) error {
	cmd := exec.CommandContext(ctx, "systemctl", "--user", "daemon-reload")
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("systemctl --user daemon-reload: %w: %s", err, strings.TrimSpace(string(out)))
	}
	return nil
}

func Enable(ctx context.Context) error {
	cmd := exec.CommandContext(ctx, "systemctl", "--user", "enable", ServiceName)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("enable %s: %w: %s", ServiceName, err, strings.TrimSpace(string(out)))
	}
	return nil
}

func Start(ctx context.Context) error {
	cmd := exec.CommandContext(ctx, "systemctl", "--user", "start", ServiceName)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("start %s: %w: %s", ServiceName, err, strings.TrimSpace(string(out)))
	}
	return nil
}

func Stop(ctx context.Context) error {
	cmd := exec.CommandContext(ctx, "systemctl", "--user", "stop", ServiceName)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("stop %s: %w: %s", ServiceName, err, strings.TrimSpace(string(out)))
	}
	return nil
}

func Restart(ctx context.Context) error {
	cmd := exec.CommandContext(ctx, "systemctl", "--user", "restart", ServiceName)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("restart %s: %w: %s", ServiceName, err, strings.TrimSpace(string(out)))
	}
	return nil
}

func StatusCheck(ctx context.Context, paths mneshfs.Paths, cfg Config) Status {
	status := Status{}
	if _, err := os.Stat(paths.LLMServicePath); err == nil {
		status.ServiceFileExists = true
	}
	if cfg.ServerBin != "" {
		if _, err := os.Stat(cfg.ServerBin); err == nil {
			status.ServerBinExists = true
		} else if resolved, err := exec.LookPath(cfg.ServerBin); err == nil && resolved != "" {
			status.ServerBinExists = true
		}
	}
	if _, err := os.Stat(cfg.ModelPath); err == nil {
		status.ModelExists = true
	}
	if _, err := exec.LookPath("systemctl"); err == nil {
		status.SystemctlExists = true
		status.IsEnabled = systemctlOK(ctx, "--user", "is-enabled", ServiceName)
		status.IsActive = systemctlOK(ctx, "--user", "is-active", ServiceName)
	}
	if ok, desc := healthCheck(ctx, cfg.Host, cfg.Port); ok {
		status.HealthOK = true
		status.HealthStatus = desc
	} else {
		status.HealthStatus = desc
	}
	return status
}

func healthCheck(ctx context.Context, host string, port int) (bool, string) {
	url := fmt.Sprintf("http://%s:%d/health", host, port)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return false, err.Error()
	}
	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return false, err.Error()
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return false, resp.Status
	}
	return true, resp.Status
}

func systemctlOK(ctx context.Context, args ...string) bool {
	cmd := exec.CommandContext(ctx, "systemctl", args...)
	return cmd.Run() == nil
}

func DownloadPayload() []byte {
	body, _ := json.Marshal(map[string]string{
		"repo_id":  DefaultRepoID,
		"filename": DefaultFileName,
	})
	return body
}

func DownloadCommand(paths mneshfs.Paths) *exec.Cmd {
	cmd := exec.Command(
		"llama-cli",
		"--hf-repo", DefaultRepoID,
		"--hf-file", DefaultFileName,
		"-p", "ok",
	)
	cmd.Dir = paths.QwenDir
	cmd.Env = append(os.Environ(),
		"LLAMA_CACHE="+paths.CacheDir,
	)
	return cmd
}

func DownloadViaLLamaCLI(ctx context.Context, paths mneshfs.Paths) error {
	if err := os.MkdirAll(paths.QwenDir, 0o755); err != nil {
		return fmt.Errorf("mkdir qwen dir: %w", err)
	}
	cmd := DownloadCommand(paths)
	cmd = exec.CommandContext(ctx, cmd.Path, cmd.Args[1:]...)
	cmd.Dir = paths.QwenDir
	cmd.Env = append(os.Environ(), "LLAMA_CACHE="+paths.CacheDir)
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("llama-cli hf download probe failed: %w: %s", err, strings.TrimSpace(stderr.String()))
	}
	downloaded := filepath.Join(paths.QwenDir, DefaultFileName)
	if _, err := os.Stat(downloaded); err == nil && downloaded != paths.QwenModelPath {
		if err := os.Rename(downloaded, paths.QwenModelPath); err == nil {
			return nil
		}
	}
	if _, err := os.Stat(paths.QwenModelPath); err == nil {
		return nil
	}
	return fmt.Errorf("expected gguf not found at %s after llama-cli run", paths.QwenModelPath)
}

func resolveServerBin() string {
	if custom := strings.TrimSpace(os.Getenv("MNESH_LLAMA_SERVER")); custom != "" {
		return custom
	}
	if resolved, err := exec.LookPath("llama-server"); err == nil {
		return resolved
	}
	if wd, err := os.Getwd(); err == nil {
		candidates := []string{
			filepath.Join(wd, "..", "llama.cpp", "build", "bin", "llama-server"),
			filepath.Join(wd, "llama.cpp", "build", "bin", "llama-server"),
		}
		for _, candidate := range candidates {
			if abs, err := filepath.Abs(candidate); err == nil {
				if _, err := os.Stat(abs); err == nil {
					return abs
				}
			}
		}
	}
	if home, err := os.UserHomeDir(); err == nil {
		candidates := []string{
			filepath.Join(home, "Desktop", "me", "coder", "llama.cpp", "build", "bin", "llama-server"),
			filepath.Join(home, "code", "llama.cpp", "build", "bin", "llama-server"),
		}
		for _, candidate := range candidates {
			if _, err := os.Stat(candidate); err == nil {
				return candidate
			}
		}
	}
	return ""
}

func Predict(ctx context.Context, cfg Config, events []store.CommandEvent, maxTokens int) (Prediction, error) {
	prompts := []string{
		buildPrompt(events),
		buildFallbackPrompt(events),
		buildMinimalPrompt(events),
	}
	suggestion := ""
	for _, prompt := range prompts {
		out, err := complete(ctx, cfg, prompt, maxTokens)
		if err != nil {
			return Prediction{}, err
		}
		if isValidSuggestion(out) {
			suggestion = out
			break
		}
	}
	last := latestEvent(events)
	ctxMap := map[string]any{
		"cwd":        last.Cwd,
		"git_branch": last.GitBranch,
		"shell":      last.Shell,
		"hostname":   last.Hostname,
	}
	return Prediction{
		ModelVersion:     "qwen2.5-coder-0.5b-q4_k_m-llama.cpp",
		PredictedCmdType: inferCmdType(suggestion),
		TopCmdTypes:      []string{},
		SessionContext:   inferSessionContext(events),
		Ecosystem:        inferEcosystem(events),
		Context:          ctxMap,
		WindowCommands:   commandsOnly(events),
		Suggestion:       suggestion,
	}, nil
}

func complete(ctx context.Context, cfg Config, prompt string, maxTokens int) (string, error) {
	reqBody := map[string]any{
		"model":       cfg.FileName,
		"prompt":      prompt,
		"temperature": 0.2,
		"top_p":       0.95,
		"max_tokens":  maxTokens,
		"stream":      false,
		"stop":        []string{"<|im_end|>", "<|endoftext|>"},
	}
	body, err := json.Marshal(reqBody)
	if err != nil {
		return "", err
	}

	url := fmt.Sprintf("http://%s:%d/v1/completions", cfg.Host, cfg.Port)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("llama-server returned %s", resp.Status)
	}

	var payload struct {
		Choices []struct {
			Text string `json:"text"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return "", err
	}
	if len(payload.Choices) == 0 {
		return "", fmt.Errorf("llama-server returned no choices")
	}
	return sanitizeSuggestion(payload.Choices[0].Text), nil
}

func buildPrompt(events []store.CommandEvent) string {
	last := latestEvent(events)
	promptEvents := promptEvents(events)
	ecosystem := inferEcosystem(events)
	sessionContext := inferSessionContext(events)
	workflowHint := inferWorkflowHint(events)
	var b strings.Builder
	b.WriteString("You are mnesh.\n")
	b.WriteString("Task: predict one likely next shell command.\n")
	b.WriteString("Output rules: one command only, no numbering, no bullets, no explanation.\n")
	b.WriteString("Prefer the next useful action, not a repeated inspection command unless the history strongly suggests it.\n")
	b.WriteString("Ignore low-signal commands like clear, pwd, and simple cd navigation unless they are central to the workflow.\n")
	b.WriteString("Respect the dominant ecosystem and workflow hinted below more than a single misleading tail command.\n")
	fmt.Fprintf(&b, "shell: %s\n", emptyFallback(last.Shell, "zsh"))
	fmt.Fprintf(&b, "cwd: %s\n", emptyFallback(last.Cwd, "."))
	fmt.Fprintf(&b, "git_branch: %s\n", emptyFallback(last.GitBranch, "-"))
	fmt.Fprintf(&b, "hostname: %s\n", emptyFallback(last.Hostname, "localhost"))
	fmt.Fprintf(&b, "ecosystem_hint: %s\n", ecosystem)
	fmt.Fprintf(&b, "session_context_hint: %s\n", sessionContext)
	fmt.Fprintf(&b, "workflow_hint: %s\n", workflowHint)
	b.WriteString("recent_commands:\n")
	for _, event := range promptEvents {
		fmt.Fprintf(&b, "- %s\n", event.Command)
	}
	b.WriteString("command> ")
	return b.String()
}

func buildFallbackPrompt(events []store.CommandEvent) string {
	promptEvents := promptEvents(events)
	ecosystem := inferEcosystem(events)
	workflowHint := inferWorkflowHint(events)
	var b strings.Builder
	b.WriteString("Predict one likely next shell command.\n")
	b.WriteString("Ignore clear and trivial navigation. Output one useful command only.\n")
	fmt.Fprintf(&b, "ecosystem_hint: %s\n", ecosystem)
	fmt.Fprintf(&b, "workflow_hint: %s\n", workflowHint)
	b.WriteString("history:\n")
	for _, event := range promptEvents {
		fmt.Fprintf(&b, "- %s\n", event.Command)
	}
	b.WriteString("command> ")
	return b.String()
}

func buildMinimalPrompt(events []store.CommandEvent) string {
	promptEvents := promptEvents(events)
	var b strings.Builder
	b.WriteString("One likely next shell command only.\n")
	for _, event := range promptEvents {
		fmt.Fprintf(&b, "%s\n", event.Command)
	}
	b.WriteString("command> ")
	return b.String()
}

func sanitizeSuggestion(text string) string {
	text = strings.TrimSpace(text)
	if idx := strings.IndexByte(text, '\n'); idx >= 0 {
		text = text[:idx]
	}
	text = strings.Trim(text, "`")
	text = strings.TrimSpace(text)
	text = strings.TrimPrefix(text, "- ")
	text = strings.TrimPrefix(text, "* ")
	text = strings.TrimSpace(text)
	if strings.HasPrefix(strings.ToLower(text), "mnesh ") {
		return ""
	}
	return text
}

func isValidSuggestion(text string) bool {
	text = strings.TrimSpace(text)
	if text == "" {
		return false
	}
	lower := strings.ToLower(text)
	if strings.Contains(lower, "next_command") || strings.Contains(lower, "recent_commands") || strings.Contains(lower, "history:") || strings.Contains(lower, "command>") {
		return false
	}
	if strings.HasPrefix(lower, "you are ") {
		return false
	}
	allDigits := true
	hasAlpha := false
	for _, r := range text {
		if r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' {
			hasAlpha = true
		}
		if r < '0' || r > '9' {
			if r != ' ' && r != '\t' {
				allDigits = false
			}
		}
	}
	if allDigits {
		return false
	}
	if !hasAlpha && !strings.ContainsAny(text, "/.-_") {
		return false
	}
	return true
}

func commandsOnly(events []store.CommandEvent) []string {
	out := make([]string, 0, len(events))
	for _, event := range events {
		out = append(out, event.Command)
	}
	return out
}

func promptEvents(events []store.CommandEvent) []store.CommandEvent {
	filtered := make([]store.CommandEvent, 0, len(events))
	for _, event := range events {
		if isLowSignal(event.Command) {
			continue
		}
		filtered = append(filtered, event)
	}
	if len(filtered) == 0 {
		return events
	}
	return filtered
}

func isLowSignal(cmd string) bool {
	text := strings.ToLower(strings.TrimSpace(cmd))
	switch {
	case text == "":
		return true
	case text == "clear", text == "pwd", text == "ls", text == "ls -l", text == "ls -la", text == "ls -lah":
		return true
	case strings.HasPrefix(text, "cd "):
		return true
	default:
		return false
	}
}

func latestEvent(events []store.CommandEvent) store.CommandEvent {
	if len(events) == 0 {
		return store.CommandEvent{}
	}
	return events[len(events)-1]
}

func emptyFallback(v, fallback string) string {
	if strings.TrimSpace(v) == "" {
		return fallback
	}
	return v
}

func inferSessionContext(events []store.CommandEvent) string {
	eco := inferEcosystem(events)
	switch eco {
	case "node":
		return "frontend_dev"
	case "python", "go", "rust":
		return "backend_dev"
	case "infra":
		return "devops"
	case "db", "system":
		return "system_admin"
	default:
		return "exploration"
	}
}

func inferEcosystem(events []store.CommandEvent) string {
	blob := strings.ToLower(strings.Join(commandsOnly(events), "\n"))
	switch {
	case strings.Contains(blob, "npm ") || strings.Contains(blob, "pnpm ") || strings.Contains(blob, "npx ") || strings.Contains(blob, "vite"):
		return "node"
	case strings.Contains(blob, "python") || strings.Contains(blob, "pytest") || strings.Contains(blob, "pip ") || strings.Contains(blob, "manage.py"):
		return "python"
	case strings.Contains(blob, "docker") || strings.Contains(blob, "kubectl") || strings.Contains(blob, "helm ") || strings.Contains(blob, "terraform "):
		return "infra"
	case strings.Contains(blob, "go "):
		return "go"
	case strings.Contains(blob, "cargo "):
		return "rust"
	case strings.Contains(blob, "psql") || strings.Contains(blob, "mysql") || strings.Contains(blob, "mongosh") || strings.Contains(blob, "redis"):
		return "db"
	default:
		return "misc"
	}
}

func inferWorkflowHint(events []store.CommandEvent) string {
	counts := map[string]int{}
	for _, event := range events {
		cmdType := inferCmdType(event.Command)
		if cmdType == "filesystem" && isLowSignal(event.Command) {
			continue
		}
		counts[cmdType]++
	}

	bestType := "misc"
	bestCount := 0
	for typ, count := range counts {
		if count > bestCount {
			bestType = typ
			bestCount = count
		}
	}

	switch bestType {
	case "git":
		return "git workflow, likely next action is a meaningful git command or nearby development step"
	case "container":
		return "container or infra workflow, likely next action is docker or kubectl related"
	case "python":
		return "python workflow, likely next action is run, test, package, or edit-related"
	case "node":
		return "node or frontend workflow, likely next action is build, dev, test, or package-related"
	case "system":
		return "system administration workflow, likely next action is service, ssh, inspection, or disk/process related"
	case "text_processing":
		return "text inspection workflow, likely next action is grep, rg, sed, awk, or log analysis"
	default:
		return "general shell workflow; prefer a useful continuation over repeated inspection"
	}
}

func inferCmdType(cmd string) string {
	text := strings.ToLower(strings.TrimSpace(cmd))
	switch {
	case strings.HasPrefix(text, "git "):
		return "git"
	case strings.HasPrefix(text, "docker ") || strings.HasPrefix(text, "docker compose") || strings.HasPrefix(text, "kubectl "):
		return "container"
	case strings.HasPrefix(text, "python") || strings.HasPrefix(text, "pytest") || strings.HasPrefix(text, "pip "):
		return "python"
	case strings.HasPrefix(text, "npm ") || strings.HasPrefix(text, "pnpm ") || strings.HasPrefix(text, "npx "):
		return "node"
	case strings.HasPrefix(text, "ssh ") || strings.HasPrefix(text, "systemctl ") || strings.HasPrefix(text, "sudo systemctl"):
		return "system"
	case strings.HasPrefix(text, "grep ") || strings.HasPrefix(text, "rg ") || strings.HasPrefix(text, "sed ") || strings.HasPrefix(text, "awk "):
		return "text_processing"
	case strings.HasPrefix(text, "curl ") || strings.HasPrefix(text, "wget "):
		return "network"
	case strings.HasPrefix(text, "apt ") || strings.HasPrefix(text, "brew "):
		return "package"
	case strings.HasPrefix(text, "ls") || strings.HasPrefix(text, "cd ") || strings.HasPrefix(text, "pwd") || strings.HasPrefix(text, "find "):
		return "filesystem"
	default:
		return "misc"
	}
}
