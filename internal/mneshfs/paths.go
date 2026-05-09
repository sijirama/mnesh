package mneshfs

import (
	"fmt"
	"os"
	"path/filepath"
)

type Paths struct {
	Root              string
	DataDir           string
	ModelsDir         string
	BinDir            string
	BinPath           string
	LogsDir           string
	CacheDir          string
	HooksDir          string
	ConfigPath        string
	DBPath            string
	ActiveModelPath   string
	PythonDir         string
	PredictWorkerPath string
	VenvPython        string
}

func Resolve() (Paths, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return Paths{}, fmt.Errorf("resolve user home: %w", err)
	}

	root := filepath.Join(home, ".mnesh")
	pythonDir := filepath.Join(root, "python")
	return Paths{
		Root:              root,
		DataDir:           filepath.Join(root, "data"),
		ModelsDir:         filepath.Join(root, "models"),
		BinDir:            filepath.Join(root, "bin"),
		BinPath:           filepath.Join(root, "bin", "mnesh"),
		LogsDir:           filepath.Join(root, "logs"),
		CacheDir:          filepath.Join(root, "cache"),
		HooksDir:          filepath.Join(root, "hooks"),
		ConfigPath:        filepath.Join(root, "config.json"),
		DBPath:            filepath.Join(root, "data", "commands.db"),
		ActiveModelPath:   filepath.Join(root, "models", "active"),
		PythonDir:         pythonDir,
		PredictWorkerPath: filepath.Join(pythonDir, "scripts", "predict_worker.py"),
		VenvPython:        filepath.Join(root, ".venv", "bin", "python3"),
	}, nil
}
