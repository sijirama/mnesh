// Package python embeds the runtime Python files (predict worker + minimal
// model package) into the mnesh binary. They are materialized to a stable
// absolute path under MNESH_HOME on `mnesh init` so prediction works from
// any cwd (including curl-installed setups with no repo checkout).
//
// To refresh embedded copies after editing model/*.py or
// scripts/predict_worker.py, run:
//
//	go generate ./internal/python
package python

//go:generate sh -c "cp ../../scripts/predict_worker.py py/scripts/predict_worker.py && cp ../../model/main.py py/model/main.py && cp ../../model/v5_main.py py/model/v5_main.py"

import (
	"embed"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
)

//go:embed all:py
var embedded embed.FS

// PredictWorkerRel is the path of the predict worker entrypoint, relative
// to the directory passed to Materialize.
const PredictWorkerRel = "scripts/predict_worker.py"

// Materialize writes the embedded python tree into targetDir, creating
// parent directories as needed and overwriting existing files. Existing
// content under targetDir is left untouched unless an embedded file shares
// its path.
func Materialize(targetDir string) error {
	return fs.WalkDir(embedded, "py", func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel := path
		if rel == "py" {
			rel = ""
		} else if len(rel) > 3 && rel[:3] == "py/" {
			rel = rel[3:]
		}

		dst := filepath.Join(targetDir, rel)

		if d.IsDir() {
			return os.MkdirAll(dst, 0o755)
		}

		if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
			return err
		}

		data, err := embedded.ReadFile(path)
		if err != nil {
			return fmt.Errorf("read embedded %s: %w", path, err)
		}
		if err := os.WriteFile(dst, data, 0o644); err != nil {
			return fmt.Errorf("write %s: %w", dst, err)
		}
		return nil
	})
}
