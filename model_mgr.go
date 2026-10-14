package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
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
	modelsDir  string
	mu         sync.Mutex
	isBusy     bool
	cancelChan chan struct{}
}

func NewModelManager(modelsDir string) *ModelManager {
	_ = os.MkdirAll(modelsDir, 0755)
	return &ModelManager{
		modelsDir: modelsDir,
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
	mm.mu.Unlock()

	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}

	var models []ModelInfo
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(strings.ToLower(entry.Name()), ".gguf") {
			continue
		}

		info, err := entry.Info()
		if err != nil {
			continue
		}

		models = append(models, ModelInfo{
			Name:          entry.Name(),
			Path:          filepath.Join(dir, entry.Name()),
			SizeBytes:     info.Size(),
			SizeFormatted: formatBytes(info.Size()),
			Quant:         detectQuant(entry.Name()),
			ModifiedAt:    info.ModTime().Format("2006-01-02 15:04"),
		})
	}

	return models, nil
}

func (mm *ModelManager) DeleteModel(filename string) error {
	mm.mu.Lock()
	dir := mm.modelsDir
	mm.mu.Unlock()

	cleanName := filepath.Base(filename)
	target := filepath.Join(dir, cleanName)
	return os.Remove(target)
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

	mm.mu.Lock()
	dir := mm.modelsDir
	mm.mu.Unlock()

	destName := filepath.Base(sourcePath)
	destPath := filepath.Join(dir, destName)

	if filepath.Clean(sourcePath) != filepath.Clean(destPath) {
		_ = os.Remove(destPath)
		// Try hard link first (instant, 0 disk space on same volume)
		if err := os.Link(sourcePath, destPath); err != nil {
			// Fallback to copy
			in, err := os.Open(sourcePath)
			if err != nil {
				return ModelInfo{}, err
			}
			defer in.Close()

			out, err := os.Create(destPath)
			if err != nil {
				return ModelInfo{}, err
			}
			defer out.Close()

			if _, err := io.Copy(out, in); err != nil {
				return ModelInfo{}, err
			}
		}
	}

	info, _ := os.Stat(destPath)
	if info == nil {
		info = srcInfo
	}

	return ModelInfo{
		Name:          destName,
		Path:          destPath,
		SizeBytes:     info.Size(),
		SizeFormatted: formatBytes(info.Size()),
		Quant:         detectQuant(destName),
		ModifiedAt:    info.ModTime().Format("2006-01-02 15:04"),
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
