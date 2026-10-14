package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"
)

// ModelInfo describes a local GGUF model file.
type ModelInfo struct {
	Name          string `json:"name"`
	Path          string `json:"path"`
	SizeBytes     int64  `json:"sizeBytes"`
	SizeFormatted string `json:"sizeFormatted"`
	Quant         string `json:"quant"`
	ModifiedAt    string `json:"modifiedAt"`
	IsExternal    bool   `json:"isExternal"`
}

// HuggingFaceFile represents a GGUF model file in a Hugging Face repo.
type HuggingFaceFile struct {
	Filename      string `json:"filename"`
	SizeBytes     int64  `json:"sizeBytes"`
	SizeFormatted string `json:"sizeFormatted"`
	DownloadURL   string `json:"downloadUrl"`
	Quant         string `json:"quant"`
}

type ModelManager struct {
	modelsDir       string
	appDataDir      string
	registeredPaths []string
	mu              sync.Mutex
	isBusy          bool
	cancelChan      chan struct{}
}

func NewModelManager(modelsDir, appDataDir string) *ModelManager {
	_ = os.MkdirAll(modelsDir, 0755)
	mm := &ModelManager{
		modelsDir:  modelsDir,
		appDataDir: appDataDir,
	}
	mm.loadRegisteredModels()
	return mm
}

func (mm *ModelManager) getRegistryPath() string {
	if mm.appDataDir != "" {
		return filepath.Join(mm.appDataDir, "registered_models.json")
	}
	return filepath.Join(mm.modelsDir, "registered_models.json")
}

func (mm *ModelManager) loadRegisteredModels() {
	p := mm.getRegistryPath()
	data, err := os.ReadFile(p)
	if err != nil {
		return
	}
	var paths []string
	if err := json.Unmarshal(data, &paths); err == nil {
		mm.registeredPaths = paths
	}
}

func (mm *ModelManager) saveRegisteredModels() {
	p := mm.getRegistryPath()
	data, err := json.MarshalIndent(mm.registeredPaths, "", "  ")
	if err == nil {
		_ = os.WriteFile(p, data, 0644)
	}
}

func (mm *ModelManager) GetModelsDir() string {
	mm.mu.Lock()
	defer mm.mu.Unlock()
	return mm.modelsDir
}

func (mm *ModelManager) SetModelsDir(dir string) error {
	mm.mu.Lock()
	defer mm.mu.Unlock()
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}
	mm.modelsDir = dir
	return nil
}

func (mm *ModelManager) ListModels() ([]ModelInfo, error) {
	mm.mu.Lock()
	dir := mm.modelsDir
	registered := make([]string, len(mm.registeredPaths))
	copy(registered, mm.registeredPaths)
	mm.mu.Unlock()

	var models []ModelInfo
	seenPaths := make(map[string]bool)

	// 1. Scan default models directory (including symlinks)
	entries, err := os.ReadDir(dir)
	if err == nil {
		for _, entry := range entries {
			if entry.IsDir() || !strings.HasSuffix(strings.ToLower(entry.Name()), ".gguf") {
				continue
			}

			p := filepath.Join(dir, entry.Name())
			targetInfo, err := os.Stat(p) // Resolves symlink to inspect target file
			if err != nil || targetInfo.Size() < 1024*1024 {
				continue // Skip corrupt or 0-byte files or broken symlinks
			}

			lstatInfo, _ := os.Lstat(p)
			isSymlink := lstatInfo != nil && (lstatInfo.Mode()&os.ModeSymlink != 0)

			cleanP := filepath.Clean(p)
			seenPaths[cleanP] = true
			if isSymlink {
				if realTarget, err := filepath.EvalSymlinks(p); err == nil {
					seenPaths[filepath.Clean(realTarget)] = true
				}
			}

			models = append(models, ModelInfo{
				Name:          entry.Name(),
				Path:          p,
				SizeBytes:     targetInfo.Size(),
				SizeFormatted: formatBytes(targetInfo.Size()),
				Quant:         detectQuant(entry.Name()),
				ModifiedAt:    targetInfo.ModTime().Format("2006-01-02 15:04"),
				IsExternal:    isSymlink,
			})
		}
	}

	// 2. Scan registered external model files
	var validRegistered []string
	changed := false

	for _, p := range registered {
		cleanP := filepath.Clean(p)
		if seenPaths[cleanP] {
			validRegistered = append(validRegistered, p)
			continue
		}

		info, err := os.Stat(p)
		if err != nil || info.Size() < 1024*1024 {
			changed = true
			continue
		}

		validRegistered = append(validRegistered, p)
		seenPaths[cleanP] = true

		models = append(models, ModelInfo{
			Name:          filepath.Base(p),
			Path:          p,
			SizeBytes:     info.Size(),
			SizeFormatted: formatBytes(info.Size()),
			Quant:         detectQuant(filepath.Base(p)),
			ModifiedAt:    info.ModTime().Format("2006-01-02 15:04"),
			IsExternal:    true,
		})
	}

	if changed {
		mm.mu.Lock()
		mm.registeredPaths = validRegistered
		mm.saveRegisteredModels()
		mm.mu.Unlock()
	}

	return models, nil
}

func (mm *ModelManager) DeleteModel(identifier string) error {
	mm.mu.Lock()
	defer mm.mu.Unlock()

	cleanIdent := filepath.Clean(identifier)
	baseIdent := filepath.Base(identifier)

	// Check if this model is in registeredPaths
	var newRegistered []string
	removedFromRegistry := false
	for _, p := range mm.registeredPaths {
		if filepath.Clean(p) == cleanIdent || filepath.Base(p) == baseIdent {
			removedFromRegistry = true
		} else {
			newRegistered = append(newRegistered, p)
		}
	}

	if removedFromRegistry {
		mm.registeredPaths = newRegistered
		mm.saveRegisteredModels()
		return nil
	}

	// If the file in modelsDir is a symlink or file, os.Remove safely deletes only the link/file
	target := filepath.Join(mm.modelsDir, baseIdent)
	if _, err := os.Lstat(target); err == nil {
		return os.Remove(target)
	}

	// Try removing direct path first if it exists outside modelsDir
	if _, err := os.Stat(cleanIdent); err == nil && !strings.EqualFold(filepath.Dir(cleanIdent), mm.modelsDir) {
		return os.Remove(cleanIdent)
	}

	return os.Remove(target)
}

func createSymlink(target, link string) error {
	_ = os.Remove(link)

	// Try standard os.Symlink first
	if err := os.Symlink(target, link); err == nil {
		return nil
	}

	// On Windows, fall back to mklink via cmd
	if runtime.GOOS == "windows" {
		cmd := prepareCmd(exec.Command("cmd", "/c", "mklink", link, target))
		if out, err := cmd.CombinedOutput(); err == nil && !strings.Contains(strings.ToLower(string(out)), "error") {
			return nil
		}
	}

	return fmt.Errorf("symlink not supported")
}

func (mm *ModelManager) ImportModel(sourcePath string) (ModelInfo, error) {
	sourcePath = strings.TrimSpace(sourcePath)
	if !strings.HasSuffix(strings.ToLower(sourcePath), ".gguf") {
		return ModelInfo{}, fmt.Errorf("selected file is not a .gguf model: %s", sourcePath)
	}

	srcInfo, err := os.Stat(sourcePath)
	if err != nil {
		return ModelInfo{}, fmt.Errorf("cannot read source file: %w", err)
	}
	if srcInfo.Size() < 1024*1024 {
		return ModelInfo{}, fmt.Errorf("selected model file is empty or invalid (%d bytes): %s", srcInfo.Size(), sourcePath)
	}

	mm.mu.Lock()
	dir := mm.modelsDir
	mm.mu.Unlock()

	cleanSource := filepath.Clean(sourcePath)
	destName := filepath.Base(sourcePath)
	destPath := filepath.Join(dir, destName)
	cleanDest := filepath.Clean(destPath)

	// Case 1: Already inside modelsDir
	if cleanSource == cleanDest {
		return ModelInfo{
			Name:          destName,
			Path:          cleanDest,
			SizeBytes:     srcInfo.Size(),
			SizeFormatted: formatBytes(srcInfo.Size()),
			Quant:         detectQuant(destName),
			ModifiedAt:    srcInfo.ModTime().Format("2006-01-02 15:04"),
			IsExternal:    false,
		}, nil
	}

	// Case 2: Create symbolic link in modelsDir (0 extra disk space, instant)
	if err := createSymlink(cleanSource, destPath); err == nil {
		return ModelInfo{
			Name:          destName,
			Path:          destPath,
			SizeBytes:     srcInfo.Size(),
			SizeFormatted: formatBytes(srcInfo.Size()),
			Quant:         detectQuant(destName),
			ModifiedAt:    srcInfo.ModTime().Format("2006-01-02 15:04"),
			IsExternal:    true,
		}, nil
	}

	// Case 3: Register in-place (instant, 0 disk space across volumes if OS restricts symlinks)
	mm.mu.Lock()
	alreadyRegistered := false
	for _, p := range mm.registeredPaths {
		if filepath.Clean(p) == cleanSource {
			alreadyRegistered = true
			break
		}
	}
	if !alreadyRegistered {
		mm.registeredPaths = append(mm.registeredPaths, cleanSource)
		mm.saveRegisteredModels()
	}
	mm.mu.Unlock()

	return ModelInfo{
		Name:          destName,
		Path:          cleanSource,
		SizeBytes:     srcInfo.Size(),
		SizeFormatted: formatBytes(srcInfo.Size()),
		Quant:         detectQuant(destName),
		ModifiedAt:    srcInfo.ModTime().Format("2006-01-02 15:04"),
		IsExternal:    true,
	}, nil
}


func (mm *ModelManager) FetchHFRepoGGUFs(repoID string) ([]HuggingFaceFile, error) {
	repoID = strings.TrimSpace(repoID)
	if repoID == "" {
		return nil, fmt.Errorf("repository ID cannot be empty")
	}

	url := fmt.Sprintf("https://huggingface.co/api/models/%s", repoID)
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", "llama-manager")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Hugging Face API returned HTTP %d for '%s'", resp.StatusCode, repoID)
	}

	var data struct {
		Siblings []struct {
			RFilename string `json:"rfilename"`
			Size      int64  `json:"size"`
		} `json:"siblings"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		return nil, err
	}

	var files []HuggingFaceFile
	for _, s := range data.Siblings {
		if strings.HasSuffix(strings.ToLower(s.RFilename), ".gguf") {
			downloadURL := fmt.Sprintf("https://huggingface.co/%s/resolve/main/%s", repoID, s.RFilename)
			files = append(files, HuggingFaceFile{
				Filename:      filepath.Base(s.RFilename),
				SizeBytes:     s.Size,
				SizeFormatted: formatBytes(s.Size),
				DownloadURL:   downloadURL,
				Quant:         detectQuant(s.RFilename),
			})
		}
	}

	return files, nil
}

func (mm *ModelManager) DownloadModel(url, filename string, onProgress func(ProgressReport)) error {
	mm.mu.Lock()
	if mm.isBusy {
		mm.mu.Unlock()
		return fmt.Errorf("a model download is already running")
	}
	mm.isBusy = true
	mm.cancelChan = make(chan struct{})
	dir := mm.modelsDir
	mm.mu.Unlock()

	defer func() {
		mm.mu.Lock()
		mm.isBusy = false
		mm.cancelChan = nil
		mm.mu.Unlock()
	}()

	cleanFilename := filepath.Base(filename)
	destPath := filepath.Join(dir, cleanFilename)
	partPath := destPath + ".part"

	var existingSize int64
	if stat, err := os.Stat(partPath); err == nil {
		existingSize = stat.Size()
	}

	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return err
	}
	req.Header.Set("User-Agent", "llama-manager")
	if existingSize > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", existingSize))
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusPartialContent {
		return fmt.Errorf("server returned HTTP %d", resp.StatusCode)
	}

	var totalBytes int64 = resp.ContentLength
	if resp.StatusCode == http.StatusPartialContent {
		totalBytes += existingSize
	}

	var out *os.File
	if existingSize > 0 && resp.StatusCode == http.StatusPartialContent {
		out, err = os.OpenFile(partPath, os.O_WRONLY|os.O_APPEND, 0644)
	} else {
		existingSize = 0
		out, err = os.Create(partPath)
	}
	if err != nil {
		return err
	}
	defer out.Close()

	buf := make([]byte, 128*1024)
	downloaded := existingSize
	startTime := time.Now()
	lastReport := time.Now()

	for {
		select {
		case <-mm.cancelChan:
			return fmt.Errorf("download cancelled")
		default:
		}

		n, err := resp.Body.Read(buf)
		if n > 0 {
			if _, wErr := out.Write(buf[:n]); wErr != nil {
				return wErr
			}
			downloaded += int64(n)

			if time.Since(lastReport) > 250*time.Millisecond {
				lastReport = time.Now()
				elapsedSec := time.Since(startTime).Seconds()
				var speed float64
				if elapsedSec > 0 {
					speed = (float64(downloaded-existingSize) / (1024 * 1024)) / elapsedSec
				}
				var pct float64
				if totalBytes > 0 {
					pct = float64(downloaded) / float64(totalBytes) * 100
				}
				onProgress(ProgressReport{
					Task:       "download_model",
					ItemName:   cleanFilename,
					Status:     "downloading",
					BytesRead:  downloaded,
					TotalBytes: totalBytes,
					Percent:    pct,
					SpeedMBps:  speed,
				})
			}
		}

		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
	}

	out.Close()
	_ = os.Remove(destPath)
	if err := os.Rename(partPath, destPath); err != nil {
		return err
	}

	onProgress(ProgressReport{
		Task:     "download_model",
		ItemName: cleanFilename,
		Status:   "completed",
		Percent:  100,
	})

	return nil
}

func detectQuant(name string) string {
	upper := strings.ToUpper(name)
	quants := []string{
		"Q4_K_M", "Q4_K_S", "Q4_0", "Q4_1",
		"Q5_K_M", "Q5_K_S", "Q5_0", "Q5_1",
		"Q8_0", "Q6_K", "Q3_K_M", "Q3_K_S", "Q2_K",
		"BF16", "F16", "F32",
	}
	for _, q := range quants {
		if strings.Contains(upper, q) {
			return q
		}
	}
	return "GGUF"
}

func formatBytes(b int64) string {
	const unit = 1024
	if b < unit {
		return fmt.Sprintf("%d B", b)
	}
	div, exp := int64(unit), 0
	for n := b / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(b)/float64(div), "KMGTPE"[exp])
}
