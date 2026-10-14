package main

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"
)

// getServerExeName returns the binary name for the current operating system.
func getServerExeName() string {
	if runtime.GOOS == "windows" {
		return "llama-server.exe"
	}
	return "llama-server"
}

// isAssetCompatibleWithHost ensures only releases built for the detected OS and architecture are downloaded.
func isAssetCompatibleWithHost(assetName string) bool {
	name := strings.ToLower(assetName)
	switch runtime.GOOS {
	case "windows":
		if !strings.Contains(name, "win") {
			return false
		}
		if runtime.GOARCH == "arm64" {
			return strings.Contains(name, "arm64")
		}
		// x64 / amd64 Windows: reject arm64, linux, macos, android, ubuntu
		return (strings.Contains(name, "x64") || strings.Contains(name, "win")) &&
			!strings.Contains(name, "arm64") &&
			!strings.Contains(name, "ubuntu") &&
			!strings.Contains(name, "macos")
	case "linux":
		if !strings.Contains(name, "ubuntu") && !strings.Contains(name, "linux") {
			return false
		}
		if runtime.GOARCH == "arm64" {
			return strings.Contains(name, "arm64")
		}
		return strings.Contains(name, "x64") && !strings.Contains(name, "arm64")
	case "darwin":
		if !strings.Contains(name, "macos") {
			return false
		}
		if runtime.GOARCH == "arm64" {
			return strings.Contains(name, "arm64")
		}
		return strings.Contains(name, "x64")
	default:
		return strings.Contains(name, runtime.GOOS)
	}
}

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

// BackendVariantInfo details each supported backend variant and whether it is installed/active.
type BackendVariantInfo struct {
	Variant        string `json:"variant"` // "cuda", "vulkan", "cpu", "metal", "rocm"
	Name           string `json:"name"`
	Version        string `json:"version"`
	ExecutablePath string `json:"executablePath"`
	Installed      bool   `json:"installed"`
	IsActive       bool   `json:"isActive"`
	Devices        string `json:"devices,omitempty"`
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
	appDataDir string
	client     *http.Client
	mu         sync.Mutex
	isBusy     bool
	cancelChan chan struct{}
}

func NewReleaseManager(binDir, appDataDir string) *ReleaseManager {
	_ = os.MkdirAll(binDir, 0755)
	_ = os.MkdirAll(filepath.Join(appDataDir, "backends"), 0755)
	return &ReleaseManager{
		binDir:     binDir,
		appDataDir: appDataDir,
		client:     &http.Client{Timeout: 60 * time.Second},
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

	serverExe := filepath.Join(rm.binDir, getServerExeName())
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
		status.Version = "Installed"
		status.Variant = "unknown"
	}

	return status
}

func (rm *ReleaseManager) GetInstalledBackends(activeVariant string) []BackendVariantInfo {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	backendsDir := filepath.Join(rm.appDataDir, "backends")
	var variants []string
	switch runtime.GOOS {
	case "darwin":
		variants = []string{"metal", "cpu"}
	case "linux":
		variants = []string{"cuda", "rocm", "vulkan", "cpu"}
	default:
		variants = []string{"cuda", "vulkan", "cpu"}
	}

	var results []BackendVariantInfo
	for _, v := range variants {
		info := BackendVariantInfo{
			Variant:  v,
			Name:     formatBackendDisplayName(v),
			IsActive: (activeVariant == v),
		}

		targetDir := filepath.Join(backendsDir, v)
		targetExe := filepath.Join(targetDir, getServerExeName())
		legacyExe := filepath.Join(rm.binDir, getServerExeName())

		var foundExe string
		if _, err := os.Stat(targetExe); err == nil {
			foundExe = targetExe
		} else if activeVariant == v {
			if _, err := os.Stat(legacyExe); err == nil {
				foundExe = legacyExe
				targetDir = rm.binDir
			}
		}

		if foundExe != "" {
			info.Installed = true
			info.ExecutablePath = foundExe
			if vData, err := os.ReadFile(filepath.Join(targetDir, "version.txt")); err == nil {
				lines := strings.Split(strings.TrimSpace(string(vData)), "\n")
				if len(lines) > 0 {
					info.Version = strings.TrimSpace(lines[0])
				}
			} else {
				info.Version = "Installed"
			}
		}

		results = append(results, info)
	}

	return results
}

func formatBackendDisplayName(v string) string {
	switch v {
	case "cuda":
		return "NVIDIA CUDA (RTX / GTX)"
	case "vulkan":
		return "Vulkan (Universal GPU)"
	case "cpu":
		return "CPU (Native SIMD)"
	case "metal":
		return "Apple Silicon Metal"
	case "rocm":
		return "AMD ROCm / HIP"
	default:
		return strings.ToUpper(v)
	}
}

func (rm *ReleaseManager) UninstallBackend(variant, activeVariant string) error {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	backendsDir := filepath.Join(rm.appDataDir, "backends")
	targetDir := filepath.Join(backendsDir, variant)
	_ = os.RemoveAll(targetDir)

	if variant == activeVariant || activeVariant == "" {
		_ = os.RemoveAll(rm.binDir)
		_ = os.MkdirAll(rm.binDir, 0755)
	}

	return nil
}

func (rm *ReleaseManager) UninstallAllBackends() error {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	backendsDir := filepath.Join(rm.appDataDir, "backends")
	_ = os.RemoveAll(backendsDir)
	_ = os.MkdirAll(backendsDir, 0755)

	_ = os.RemoveAll(rm.binDir)
	_ = os.MkdirAll(rm.binDir, 0755)

	return nil
}

func (rm *ReleaseManager) ActivateBackend(variant string) error {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	backendsDir := filepath.Join(rm.appDataDir, "backends")
	targetDir := filepath.Join(backendsDir, variant)
	targetExe := filepath.Join(targetDir, getServerExeName())

	if _, err := os.Stat(targetExe); err != nil {
		return fmt.Errorf("backend %s is not installed in %s", variant, targetDir)
	}

	_ = os.RemoveAll(rm.binDir)
	_ = os.MkdirAll(rm.binDir, 0755)

	return copyDir(targetDir, rm.binDir)
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
			// Strict filter to only assets compatible with current host OS & architecture
			if isAssetCompatibleWithHost(a.Name) {
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

	// Match primary asset and optional cudart asset
	var primaryAsset *ReleaseAsset
	var cudartAsset *ReleaseAsset

	// Step 1: Find primary binary archive for variant
	for i := range targetRelease.Assets {
		a := &targetRelease.Assets[i]
		if !isAssetCompatibleWithHost(a.Name) {
			continue
		}
		lower := strings.ToLower(a.Name)
		if !strings.HasPrefix(lower, "llama-") {
			continue
		}

		switch variant {
		case "metal":
			if strings.Contains(lower, "macos") {
				primaryAsset = a
			}
		case "cuda":
			if strings.Contains(lower, "cuda") || strings.Contains(lower, "cu12") || strings.Contains(lower, "cu13") || strings.Contains(lower, "cu11") {
				if primaryAsset == nil || strings.Contains(lower, "13.4") || strings.Contains(lower, "12.4") {
					primaryAsset = a
				}
			}
		case "rocm":
			if strings.Contains(lower, "rocm") {
				primaryAsset = a
			}
		case "vulkan":
			if strings.Contains(lower, "vulkan") {
				primaryAsset = a
			}
		case "cpu":
			if strings.Contains(lower, "cpu") || strings.Contains(lower, "avx2") ||
				(!strings.Contains(lower, "cuda") && !strings.Contains(lower, "vulkan") && !strings.Contains(lower, "rocm") && !strings.Contains(lower, "sycl")) {
				primaryAsset = a
			}
		}
	}

	if primaryAsset == nil {
		// Fallback to any host-compatible zip/tar.gz
		for i := range targetRelease.Assets {
			a := &targetRelease.Assets[i]
			if isAssetCompatibleWithHost(a.Name) && strings.HasPrefix(strings.ToLower(a.Name), "llama-") {
				primaryAsset = a
				break
			}
		}
	}

	if primaryAsset == nil {
		return fmt.Errorf("no matching binary found for %s on %s/%s in release %s", variant, runtime.GOOS, runtime.GOARCH, tag)
	}

	// Step 2: For CUDA, match cudart archive with EXACT version harmony
	if variant == "cuda" && primaryAsset != nil {
		targetVer := ""
		pLower := strings.ToLower(primaryAsset.Name)
		if strings.Contains(pLower, "13.4") {
			targetVer = "13.4"
		} else if strings.Contains(pLower, "12.4") {
			targetVer = "12.4"
		} else if strings.Contains(pLower, "12.2") {
			targetVer = "12.2"
		} else if strings.Contains(pLower, "11.8") {
			targetVer = "11.8"
		}

		for i := range targetRelease.Assets {
			a := &targetRelease.Assets[i]
			if !isAssetCompatibleWithHost(a.Name) {
				continue
			}
			lower := strings.ToLower(a.Name)
			if strings.HasPrefix(lower, "cudart-") {
				if targetVer != "" && strings.Contains(lower, targetVer) {
					cudartAsset = a
					break
				} else if cudartAsset == nil {
					cudartAsset = a
				}
			}
		}
	}

	backendsDir := filepath.Join(rm.appDataDir, "backends")
	targetDir := filepath.Join(backendsDir, variant)
	_ = os.RemoveAll(targetDir)
	_ = os.MkdirAll(targetDir, 0755)

	tempDir := filepath.Join(rm.appDataDir, "tmp_download")
	_ = os.RemoveAll(tempDir)
	_ = os.MkdirAll(tempDir, 0755)
	defer os.RemoveAll(tempDir)

	// Step 3: Download & Extract CUDA runtime DLLs if applicable
	if variant == "cuda" && cudartAsset != nil {
		cudartArchive := filepath.Join(tempDir, cudartAsset.Name)
		if err := rm.downloadFile(cudartAsset.DownloadURL, cudartArchive, cudartAsset.Name, cudartAsset.Size, onProgress); err != nil {
			return err
		}
		if err := rm.extractArchive(cudartArchive, targetDir, onProgress); err != nil {
			return err
		}
	}

	// Step 4: Download & Extract Primary Llama Binaries
	primaryArchive := filepath.Join(tempDir, primaryAsset.Name)
	if err := rm.downloadFile(primaryAsset.DownloadURL, primaryArchive, primaryAsset.Name, primaryAsset.Size, onProgress); err != nil {
		return err
	}
	if err := rm.extractArchive(primaryArchive, targetDir, onProgress); err != nil {
		return err
	}

	// Ensure llama-server executable is directly at the root of targetDir
	serverExeName := getServerExeName()
	serverExe := filepath.Join(targetDir, serverExeName)
	if _, err := os.Stat(serverExe); err != nil {
		_ = filepath.Walk(targetDir, func(path string, info os.FileInfo, err error) error {
			if err == nil && !info.IsDir() && info.Name() == serverExeName {
				_ = os.Rename(path, serverExe)
			}
			return nil
		})
	}

	if runtime.GOOS != "windows" {
		_ = os.Chmod(serverExe, 0755)
	}

	// Write version tag and variant in targetDir
	versionFile := filepath.Join(targetDir, "version.txt")
	_ = os.WriteFile(versionFile, []byte(fmt.Sprintf("%s\n%s\n", tag, variant)), 0644)

	// Step 5: Mirror targetDir cleanly into rm.binDir as the active backend
	_ = os.RemoveAll(rm.binDir)
	_ = os.MkdirAll(rm.binDir, 0755)
	_ = copyDir(targetDir, rm.binDir)

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

func (rm *ReleaseManager) extractArchive(srcArchive, dstDir string, onProgress func(ProgressReport)) error {
	lower := strings.ToLower(srcArchive)
	if strings.HasSuffix(lower, ".zip") {
		return rm.extractZip(srcArchive, dstDir, onProgress)
	}
	if strings.HasSuffix(lower, ".tar.gz") || strings.HasSuffix(lower, ".tgz") {
		return rm.extractTarGz(srcArchive, dstDir, onProgress)
	}
	return fmt.Errorf("unsupported archive format: %s", filepath.Base(srcArchive))
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

func (rm *ReleaseManager) extractTarGz(srcTarGz, dstDir string, onProgress func(ProgressReport)) error {
	f, err := os.Open(srcTarGz)
	if err != nil {
		return err
	}
	defer f.Close()

	gzReader, err := gzip.NewReader(f)
	if err != nil {
		return err
	}
	defer gzReader.Close()

	tarReader := tar.NewReader(gzReader)

	onProgress(ProgressReport{
		Task:     "download_llama",
		ItemName: filepath.Base(srcTarGz),
		Status:   "extracting",
		Percent:  98,
	})

	for {
		header, err := tarReader.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}

		cleanName := filepath.Clean(header.Name)
		if strings.HasPrefix(cleanName, "..") {
			continue
		}
		targetPath := filepath.Join(dstDir, filepath.Base(cleanName))

		if header.Typeflag == tar.TypeDir {
			continue
		}

		outFile, err := os.OpenFile(targetPath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, os.FileMode(header.Mode))
		if err != nil {
			return err
		}

		if _, err := io.Copy(outFile, tarReader); err != nil {
			outFile.Close()
			return err
		}
		outFile.Close()
	}

	return nil
}

func copyDir(src, dst string) error {
	_ = os.MkdirAll(dst, 0755)
	entries, err := os.ReadDir(src)
	if err != nil {
		return err
	}
	for _, entry := range entries {
		srcPath := filepath.Join(src, entry.Name())
		dstPath := filepath.Join(dst, entry.Name())
		if entry.IsDir() {
			if err := copyDir(srcPath, dstPath); err != nil {
				return err
			}
		} else {
			in, err := os.Open(srcPath)
			if err != nil {
				return err
			}
			out, err := os.OpenFile(dstPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0755)
			if err != nil {
				in.Close()
				return err
			}
			_, err = io.Copy(out, in)
			in.Close()
			out.Close()
			if err != nil {
				return err
			}
		}
	}
	return nil
}

