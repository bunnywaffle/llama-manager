package main

import (
	"archive/zip"
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

// ReleaseAsset describes a downloadable file in a GitHub release.
type ReleaseAsset struct {
	Name        string `json:"name"`
	Size        int64  `json:"size"`
	DownloadURL string `json:"downloadUrl"`
}

// ReleaseInfo holds info on a llama.cpp release.
type ReleaseInfo struct {
	Tag         string         `json:"tag"`
	Name        string         `json:"name"`
	PublishedAt string         `json:"publishedAt"`
	Assets      []ReleaseAsset `json:"assets"`
	IsCurrent   bool           `json:"isCurrent"`
}

// BackendStatus holds information about the currently installed llama.cpp backend.
type BackendStatus struct {
	Installed       bool   `json:"installed"`
	Version         string `json:"version"`
	Variant         string `json:"variant"` // "cuda", "vulkan", "cpu"
	ExecutablePath  string `json:"executablePath"`
	HasServer       bool   `json:"hasServer"`
	UpdateAvailable bool   `json:"updateAvailable"`
	LatestVersion   string `json:"latestVersion"`
}

// ProgressReport reports progress of a background download/extract operation.
type ProgressReport struct {
	Task       string  `json:"task"` // "download_llama", "download_model", etc.
	ItemName   string  `json:"itemName"`
	Status     string  `json:"status"` // "idle", "downloading", "extracting", "completed", "error"
	BytesRead  int64   `json:"bytesRead"`
	TotalBytes int64   `json:"totalBytes"`
	Percent    float64 `json:"percent"`
	SpeedMBps  float64 `json:"speedMBps"`
	ErrorMsg   string  `json:"errorMsg,omitempty"`
}

type ReleaseManager struct {
	binDir     string
	client     *http.Client
	mu         sync.Mutex
	isBusy     bool
	cancelChan chan struct{}
}

func NewReleaseManager(binDir string) *ReleaseManager {
	_ = os.MkdirAll(binDir, 0755)
	return &ReleaseManager{
		binDir: binDir,
		client: &http.Client{Timeout: 60 * time.Second},
	}
}

func (rm *ReleaseManager) SetBinDir(dir string) {
	rm.mu.Lock()
	defer rm.mu.Unlock()
	rm.binDir = dir
	_ = os.MkdirAll(dir, 0755)
}

func (rm *ReleaseManager) GetInstalledBackend() BackendStatus {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	serverExe := filepath.Join(rm.binDir, "llama-server.exe")
	versionFile := filepath.Join(rm.binDir, "version.txt")

	status := BackendStatus{
		ExecutablePath: serverExe,
	}

	if _, err := os.Stat(serverExe); err == nil {
		status.HasServer = true
		status.Installed = true
	}

	if data, err := os.ReadFile(versionFile); err == nil {
		lines := strings.Split(strings.TrimSpace(string(data)), "\n")
		if len(lines) > 0 {
			status.Version = strings.TrimSpace(lines[0])
		}
		if len(lines) > 1 {
			status.Variant = strings.TrimSpace(lines[1])
		}
	} else if status.HasServer {
		status.Version = "Custom / Unspecified"
		status.Variant = "unknown"
	}

	return status
}

func (rm *ReleaseManager) CheckReleases() ([]ReleaseInfo, error) {
	req, err := http.NewRequest("GET", "https://api.github.com/repos/ggerganov/llama.cpp/releases?per_page=10", nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", "llama-manager")

	resp, err := rm.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("GitHub API returned HTTP %d", resp.StatusCode)
	}

	var rawReleases []struct {
		TagName     string `json:"tag_name"`
		Name        string `json:"name"`
		PublishedAt string `json:"published_at"`
		Assets      []struct {
			Name               string `json:"name"`
			Size               int64  `json:"size"`
			BrowserDownloadURL string `json:"browser_download_url"`
		} `json:"assets"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&rawReleases); err != nil {
		return nil, err
	}

	current := rm.GetInstalledBackend()
	var releases []ReleaseInfo

	for _, r := range rawReleases {
		rel := ReleaseInfo{
			Tag:         r.TagName,
			Name:        r.Name,
			PublishedAt: r.PublishedAt,
			IsCurrent:   current.Version == r.TagName,
		}

		for _, a := range r.Assets {
			// Filter to Windows assets
			if strings.Contains(a.Name, "win") && strings.HasSuffix(a.Name, ".zip") {
				rel.Assets = append(rel.Assets, ReleaseAsset{
					Name:        a.Name,
					Size:        a.Size,
					DownloadURL: a.BrowserDownloadURL,
				})
			}
		}

		if len(rel.Assets) > 0 {
			releases = append(releases, rel)
		}
	}

	return releases, nil
}

// DownloadAndInstall downloads the requested release tag and variant (cuda, vulkan, cpu).
func (rm *ReleaseManager) DownloadAndInstall(tag, variant string, onProgress func(ProgressReport)) error {
	rm.mu.Lock()
	if rm.isBusy {
		rm.mu.Unlock()
		return fmt.Errorf("a download operation is already in progress")
	}
	rm.isBusy = true
	rm.cancelChan = make(chan struct{})
	rm.mu.Unlock()

	defer func() {
		rm.mu.Lock()
		rm.isBusy = false
		rm.cancelChan = nil
		rm.mu.Unlock()
	}()

	// Query releases to find exact assets
	releases, err := rm.CheckReleases()
	if err != nil {
		return err
	}

	var targetRelease *ReleaseInfo
	for i := range releases {
		if releases[i].Tag == tag {
			targetRelease = &releases[i]
			break
		}
	}
	if targetRelease == nil && len(releases) > 0 {
		targetRelease = &releases[0] // default to latest
		tag = targetRelease.Tag
	}
	if targetRelease == nil {
		return fmt.Errorf("release tag %s not found", tag)
	}

	// Identify primary asset and optional cudart asset
	var primaryAsset *ReleaseAsset
	var cudartAsset *ReleaseAsset

	var needle string
	switch variant {
	case "cuda":
		needle = "cuda-12.4-x64"
	case "vulkan":
		needle = "vulkan-x64"
	case "cpu":
		needle = "cpu-x64"
	default:
		needle = "cuda-12.4-x64"
	}

	for i := range targetRelease.Assets {
		a := &targetRelease.Assets[i]
		if strings.HasPrefix(a.Name, "llama-") && strings.Contains(a.Name, needle) {
			primaryAsset = a
		} else if strings.HasPrefix(a.Name, "cudart-") && strings.Contains(a.Name, needle) {
			cudartAsset = a
		}
	}

	if primaryAsset == nil {
		// Fallback to any win-x64 zip
		for i := range targetRelease.Assets {
			a := &targetRelease.Assets[i]
			if strings.Contains(a.Name, "win") && strings.Contains(a.Name, "x64") {
				primaryAsset = a
				break
			}
		}
	}

	if primaryAsset == nil {
		return fmt.Errorf("no matching binary found for variant %s in release %s", variant, tag)
	}

	tempDir := filepath.Join(rm.binDir, "tmp_download")
	_ = os.RemoveAll(tempDir)
	_ = os.MkdirAll(tempDir, 0755)
	defer os.RemoveAll(tempDir)

	// Step 1: Download Primary Llama Binary Zip
	primaryZip := filepath.Join(tempDir, primaryAsset.Name)
	if err := rm.downloadFile(primaryAsset.DownloadURL, primaryZip, primaryAsset.Name, primaryAsset.Size, onProgress); err != nil {
		return err
	}

	// Step 2: Download CUDA runtime DLLs if applicable
	if variant == "cuda" && cudartAsset != nil {
		cudartZip := filepath.Join(tempDir, cudartAsset.Name)
		if err := rm.downloadFile(cudartAsset.DownloadURL, cudartZip, cudartAsset.Name, cudartAsset.Size, onProgress); err != nil {
			return err
		}
		// Extract cudart
		if err := rm.extractZip(cudartZip, rm.binDir, onProgress); err != nil {
			return err
		}
	}

	// Step 3: Extract Primary Llama Binaries
	if err := rm.extractZip(primaryZip, rm.binDir, onProgress); err != nil {
		return err
	}

	// Write version tag and variant
	versionFile := filepath.Join(rm.binDir, "version.txt")
	_ = os.WriteFile(versionFile, []byte(fmt.Sprintf("%s\n%s\n", tag, variant)), 0644)

	onProgress(ProgressReport{
		Task:     "download_llama",
		ItemName: primaryAsset.Name,
		Status:   "completed",
		Percent:  100,
	})

	return nil
}

func (rm *ReleaseManager) downloadFile(url, dst, name string, totalBytes int64, onProgress func(ProgressReport)) error {
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return err
	}
	req.Header.Set("User-Agent", "llama-manager")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download returned HTTP %d", resp.StatusCode)
	}

	if totalBytes <= 0 && resp.ContentLength > 0 {
		totalBytes = resp.ContentLength
	}

	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer out.Close()

	buf := make([]byte, 64*1024)
	var downloaded int64
	startTime := time.Now()
	lastReport := time.Now()

	for {
		select {
		case <-rm.cancelChan:
			return fmt.Errorf("download cancelled")
		default:
		}

		n, err := resp.Body.Read(buf)
		if n > 0 {
			if _, wErr := out.Write(buf[:n]); wErr != nil {
				return wErr
			}
			downloaded += int64(n)

			if time.Since(lastReport) > 200*time.Millisecond {
				lastReport = time.Now()
				elapsedSec := time.Since(startTime).Seconds()
				var speed float64
				if elapsedSec > 0 {
					speed = (float64(downloaded) / (1024 * 1024)) / elapsedSec
				}
				var pct float64
				if totalBytes > 0 {
					pct = float64(downloaded) / float64(totalBytes) * 100
				}
				onProgress(ProgressReport{
					Task:       "download_llama",
					ItemName:   name,
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

	return nil
}

func (rm *ReleaseManager) extractZip(srcZip, dstDir string, onProgress func(ProgressReport)) error {
	r, err := zip.OpenReader(srcZip)
	if err != nil {
		return err
	}
	defer r.Close()

	onProgress(ProgressReport{
		Task:     "download_llama",
		ItemName: filepath.Base(srcZip),
		Status:   "extracting",
		Percent:  98,
	})

	for _, f := range r.File {
		// Flatten or preserve relative path
		cleanName := filepath.Clean(f.Name)
		if strings.HasPrefix(cleanName, "..") {
			continue
		}
		targetPath := filepath.Join(dstDir, filepath.Base(cleanName))

		if f.FileInfo().IsDir() {
			continue
		}

		rc, err := f.Open()
		if err != nil {
			return err
		}

		outFile, err := os.OpenFile(targetPath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, f.Mode())
		if err != nil {
			rc.Close()
			return err
		}

		_, err = io.Copy(outFile, rc)
		outFile.Close()
		rc.Close()
		if err != nil {
			return err
		}
	}

	return nil
}
