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
	"os/exec"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"sort"
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

// defaultBackendForHost returns a safe default backend variant for the current
// OS without assuming any GPU vendor: Metal on macOS, CPU elsewhere.
// Callers should prefer the detected HardwareInfo.RecommendedBackend and only
// fall back to this when detection yields nothing.
func defaultBackendForHost() string {
	if runtime.GOOS == "darwin" {
		return "metal"
	}
	return "cpu"
}

// isAssetCompatibleWithHost ensures only releases built for the detected OS and architecture are downloaded.
func isAssetCompatibleWithHost(assetName string) bool {
	name := strings.ToLower(assetName)
	// Never treat framework/installer/android artifacts as server backends.
	if strings.Contains(name, "xcframework") || strings.HasSuffix(name, ".dmg") || strings.Contains(name, "android") {
		return false
	}
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
		if strings.Contains(name, "s390x") || strings.Contains(name, "snapdragon") {
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

// isSpecializedBuild reports whether a host-compatible asset is a specialized
// accelerator build (OpenVINO, SYCL, OpenCL/Adreno, Snapdragon, s390x) rather
// than one of the generic variants this app manages (cuda, rocm, vulkan,
// cpu, metal). Such builds must never be selected as a generic "cpu" backend.
func isSpecializedBuild(lower string) bool {
	special := []string{"openvino", "sycl", "opencl", "adreno", "snapdragon", "s390x", "xcframework"}
	for _, s := range special {
		if strings.Contains(lower, s) {
			return true
		}
	}
	return false
}

// cudaVersionOf extracts a "major.minor" CUDA version from an asset name,
// handling both "cuda-12.8" and "cu12"/"cu13" styles. Returns "" if none found.
func cudaVersionOf(lower string) string {
	if idx := strings.Index(lower, "cuda-"); idx != -1 {
		rest := lower[idx+len("cuda-"):]
		dots := 0
		end := 0
		for end < len(rest) && ((rest[end] >= '0' && rest[end] <= '9') || rest[end] == '.') {
			if rest[end] == '.' {
				dots++
				if dots > 1 {
					break
				}
			}
			end++
		}
		if end > 0 {
			return strings.Trim(rest[:end], ".")
		}
	}
	// Fallback: cu12 / cu13 / cu11 style markers (no minor version).
	for _, v := range []string{"cu13", "cu12", "cu11"} {
		if strings.Contains(lower, v) {
			return v
		}
	}
	return ""
}

// scorePrimaryAsset ranks a host-compatible "llama-*" asset for the requested
// variant. Higher is better; negative means "not a match". The scoring prefers
// generic builds and exact CPU markers over incidental substring matches so
// that e.g. an OpenVINO Windows build is never chosen as the "cpu" backend.
func scorePrimaryAsset(lower, variant string) int {
	if !strings.HasPrefix(lower, "llama-") {
		return -1
	}
	switch variant {
	case "metal":
		if !strings.Contains(lower, "macos") {
			return -1
		}
		return 10
	case "cuda":
		if !(strings.Contains(lower, "cuda") || strings.Contains(lower, "cu12") || strings.Contains(lower, "cu13") || strings.Contains(lower, "cu11")) {
			return -1
		}
		if isSpecializedBuild(lower) {
			return -1
		}
		score := 10
		// Prefer newer CUDA toolkits when several are published with descending weights.
		cudaScores := map[string]int{
			"13.4": 25,
			"13.0": 24,
			"12.8": 20,
			"12.6": 18,
			"12.4": 16,
			"12.2": 14,
			"11.8": 10,
		}
		bonus := 0
		for ver, pts := range cudaScores {
			if strings.Contains(lower, ver) {
				if pts > bonus {
					bonus = pts
				}
			}
		}
		return score + bonus
	case "rocm":
		if !strings.Contains(lower, "rocm") {
			return -1
		}
		score := 10
		for _, v := range []string{"10.0", "6.2", "6.1", "6.0", "5.7"} {
			if strings.Contains(lower, v) {
				score += 5
				break
			}
		}
		return score
	case "vulkan":
		if !strings.Contains(lower, "vulkan") {
			return -1
		}
		if isSpecializedBuild(lower) {
			return -1
		}
		return 10
	case "cpu":
		if isSpecializedBuild(lower) {
			return -1
		}
		if strings.Contains(lower, "cuda") || strings.Contains(lower, "vulkan") ||
			strings.Contains(lower, "rocm") || strings.Contains(lower, "sycl") ||
			strings.Contains(lower, "cudart") {
			return -1
		}
		// Explicit CPU builds win over generic fallback builds.
		if strings.Contains(lower, "cpu") || strings.Contains(lower, "avx") {
			return 20
		}
		// Generic per-OS builds: ubuntu-x64 tarball, macOS archive.
		if strings.Contains(lower, "macos") {
			return 10
		}
		return 5
	default:
		return -1
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

// InstalledVersionInfo represents a specific installed build of llama.cpp.
type InstalledVersionInfo struct {
	Id             string `json:"id"`             // e.g. "b4000-cuda"
	Tag            string `json:"tag"`            // e.g. "b4000"
	Variant        string `json:"variant"`        // e.g. "cuda", "vulkan", "cpu", "metal", "rocm"
	VariantName    string `json:"variantName"`    // e.g. "NVIDIA CUDA (RTX / GTX)"
	ExecutablePath string `json:"executablePath"`
	SizeBytes      int64  `json:"sizeBytes"`
	SizeDisplay    string `json:"sizeDisplay"`    // e.g. "342 MB"
	InstalledAt    string `json:"installedAt"`    // Formatted date string
	IsActive       bool   `json:"isActive"`
}

func sanitizeDirName(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return "default"
	}
	invalid := []string{"/", "\\", ":", "*", "?", "\"", "<", ">", "|", " "}
	for _, inv := range invalid {
		s = strings.ReplaceAll(s, inv, "_")
	}
	return s
}

func dirSize(path string) int64 {
	var total int64
	_ = filepath.Walk(path, func(_ string, info os.FileInfo, err error) error {
		if err == nil && !info.IsDir() {
			total += info.Size()
		}
		return nil
	})
	return total
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
	binDir      string
	backendsDir string
	appDataDir  string
	client      *http.Client
	mu          sync.Mutex
	isBusy      bool
	cancelChan  chan struct{}
}

func NewReleaseManager(binDir, appDataDir string, backendsDir ...string) *ReleaseManager {
	bDir := ""
	if len(backendsDir) > 0 && backendsDir[0] != "" {
		bDir = backendsDir[0]
	} else {
		bDir = filepath.Join(appDataDir, "backends")
	}
	_ = os.MkdirAll(binDir, 0755)
	_ = os.MkdirAll(bDir, 0755)
	return &ReleaseManager{
		binDir:      binDir,
		backendsDir: bDir,
		appDataDir:  appDataDir,
		client:      &http.Client{Timeout: 60 * time.Second},
	}
}

func (rm *ReleaseManager) SetBinDir(dir string) {
	rm.mu.Lock()
	defer rm.mu.Unlock()
	rm.binDir = dir
	_ = os.MkdirAll(dir, 0755)
}

func (rm *ReleaseManager) SetBackendsDir(dir string) {
	rm.mu.Lock()
	defer rm.mu.Unlock()
	rm.backendsDir = dir
	_ = os.MkdirAll(dir, 0755)
}

func (rm *ReleaseManager) getBackendsDirLocked() string {
	if rm.backendsDir != "" {
		return rm.backendsDir
	}
	return filepath.Join(rm.appDataDir, "backends")
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

	// Auto-recovery: If rm.binDir does not have the executable, inspect backendsDir
	if !status.Installed {
		backendsDirs := []string{rm.getBackendsDirLocked()}
		if rm.appDataDir != "" && filepath.Join(rm.appDataDir, "backends") != rm.getBackendsDirLocked() {
			backendsDirs = append(backendsDirs, filepath.Join(rm.appDataDir, "backends"))
		}
		for _, bDir := range backendsDirs {
			entries, _ := os.ReadDir(bDir)
			for _, e := range entries {
				if !e.IsDir() {
					continue
				}
			candDir := filepath.Join(bDir, e.Name())
			candExe := filepath.Join(candDir, getServerExeName())
			if _, err := os.Stat(candExe); err == nil {
				status.HasServer = true
				status.Installed = true
				status.ExecutablePath = candExe
				if vData, err := os.ReadFile(filepath.Join(candDir, "version.txt")); err == nil {
					lines := strings.Split(strings.TrimSpace(string(vData)), "\n")
					if len(lines) > 0 {
						status.Version = strings.TrimSpace(lines[0])
					}
					if len(lines) > 1 {
						status.Variant = strings.TrimSpace(lines[1])
					}
				} else {
					status.Version = "Installed"
					status.Variant = "unknown"
				}
				break
			}
		}
		if status.Installed {
			break
		}
	}
}

	return status
}

func (rm *ReleaseManager) GetInstalledBackends(activeVariant string) []BackendVariantInfo {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	backendsDir := rm.getBackendsDirLocked()
	var variants []string
	switch runtime.GOOS {
	case "darwin":
		variants = []string{"metal", "cpu"}
	case "linux":
		variants = []string{"cuda", "rocm", "vulkan", "cpu"}
	default:
		variants = []string{"cuda", "vulkan", "cpu"}
	}

	serverExeName := getServerExeName()
	var results []BackendVariantInfo
	for _, v := range variants {
		info := BackendVariantInfo{
			Variant:  v,
			Name:     formatBackendDisplayName(v),
			IsActive: (activeVariant == v),
		}

		targetDir := filepath.Join(backendsDir, v)
		targetExe := filepath.Join(targetDir, serverExeName)
		legacyExe := filepath.Join(rm.binDir, serverExeName)

		var foundExe string
		var foundDir string
		if _, err := os.Stat(targetExe); err == nil {
			foundExe = targetExe
			foundDir = targetDir
		} else if activeVariant == v {
			if _, err := os.Stat(legacyExe); err == nil {
				foundExe = legacyExe
				foundDir = rm.binDir
			}
		}

		// Also check if any backends/*-<variant> is installed
		if foundExe == "" {
			entries, _ := os.ReadDir(backendsDir)
			for _, e := range entries {
				if !e.IsDir() {
					continue
				}
				candDir := filepath.Join(backendsDir, e.Name())
				candExe := filepath.Join(candDir, serverExeName)
				if _, err := os.Stat(candExe); err != nil {
					continue
				}
				if vData, err := os.ReadFile(filepath.Join(candDir, "version.txt")); err == nil {
					lines := strings.Split(strings.TrimSpace(string(vData)), "\n")
					if len(lines) > 1 && strings.TrimSpace(lines[1]) == v {
						foundExe = candExe
						foundDir = candDir
						break
					}
				} else if strings.HasSuffix(e.Name(), "-"+v) {
					foundExe = candExe
					foundDir = candDir
					break
				}
			}
		}

		if foundExe != "" {
			info.Installed = true
			info.ExecutablePath = foundExe
			if vData, err := os.ReadFile(filepath.Join(foundDir, "version.txt")); err == nil {
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

// GetInstalledVersions returns all installed llama.cpp builds across versions and backends.
func (rm *ReleaseManager) GetInstalledVersions() []InstalledVersionInfo {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	backendsDir := rm.getBackendsDirLocked()
	_ = os.MkdirAll(backendsDir, 0755)

	serverExeName := getServerExeName()

	activeTag := ""
	activeVariant := ""
	activeId := ""
	if bData, err := os.ReadFile(filepath.Join(rm.binDir, "version.txt")); err == nil {
		lines := strings.Split(strings.TrimSpace(string(bData)), "\n")
		if len(lines) > 0 {
			activeTag = strings.TrimSpace(lines[0])
		}
		if len(lines) > 1 {
			activeVariant = strings.TrimSpace(lines[1])
		}
		if len(lines) > 2 {
			activeId = strings.TrimSpace(lines[2])
		}
	}

	hasBinExe := false
	if _, err := os.Stat(filepath.Join(rm.binDir, serverExeName)); err == nil {
		hasBinExe = true
	}

	backendsDirs := []string{backendsDir}
	if rm.appDataDir != "" && filepath.Join(rm.appDataDir, "backends") != backendsDir {
		backendsDirs = append(backendsDirs, filepath.Join(rm.appDataDir, "backends"))
	}

	var results []InstalledVersionInfo
	foundActive := false
	seenIds := make(map[string]bool)

	for _, bDir := range backendsDirs {
		entries, err := os.ReadDir(bDir)
		if err != nil {
			continue
		}
		for _, e := range entries {
			if !e.IsDir() || seenIds[e.Name()] {
				continue
			}
			seenIds[e.Name()] = true
			subDir := filepath.Join(bDir, e.Name())
		exePath := filepath.Join(subDir, serverExeName)
		fi, err := os.Stat(exePath)
		if err != nil {
			continue
		}

		tag := ""
		variant := ""
		id := e.Name()

		if vData, err := os.ReadFile(filepath.Join(subDir, "version.txt")); err == nil {
			lines := strings.Split(strings.TrimSpace(string(vData)), "\n")
			if len(lines) > 0 {
				tag = strings.TrimSpace(lines[0])
			}
			if len(lines) > 1 {
				variant = strings.TrimSpace(lines[1])
			}
			if len(lines) > 2 && strings.TrimSpace(lines[2]) != "" {
				id = strings.TrimSpace(lines[2])
			}
		}

		if tag == "" {
			if idx := strings.LastIndex(e.Name(), "-"); idx > 0 {
				tag = e.Name()[:idx]
				if variant == "" {
					variant = e.Name()[idx+1:]
				}
			} else {
				tag = e.Name()
			}
		}
		if variant == "" {
			variant = "cpu"
		}

		canonicalId := fmt.Sprintf("%s-%s", sanitizeDirName(tag), variant)
		if e.Name() == variant {
			// Legacy bare variant folder (e.g. "cuda")
			canonicalDir := filepath.Join(backendsDir, canonicalId)
			if _, statErr := os.Stat(canonicalDir); statErr == nil {
				// Canonical versioned folder already exists!
				if activeId == e.Name() || activeId == variant {
					activeId = canonicalId
				}
				_ = os.RemoveAll(subDir)
				continue
			} else {
				// Migrate bare variant folder to canonical folder
				if err := os.Rename(subDir, canonicalDir); err == nil {
					subDir = canonicalDir
					id = canonicalId
					_ = os.WriteFile(filepath.Join(subDir, "version.txt"), []byte(fmt.Sprintf("%s\n%s\n%s\n", tag, variant, id)), 0644)
					if activeId == e.Name() || activeId == variant {
						activeId = canonicalId
					}
				}
			}
		}

		isActive := false
		if hasBinExe {
			if activeId != "" && (activeId == id || activeId == e.Name() || activeId == canonicalId) {
				isActive = true
			} else if activeId == "" && activeTag == tag && (activeVariant == variant || activeVariant == "") {
				isActive = true
			}
		}

		if isActive {
			foundActive = true
		}

		sz := dirSize(subDir)
		modTime := fi.ModTime().Format("2006-01-02 15:04")

		results = append(results, InstalledVersionInfo{
			Id:             id,
			Tag:            tag,
			Variant:        variant,
			VariantName:    formatBackendDisplayName(variant),
			ExecutablePath: exePath,
			SizeBytes:      sz,
			SizeDisplay:    formatBytes(sz),
			InstalledAt:    modTime,
			IsActive:       isActive,
		})
		}
	}

	// Legacy fallback: If binDir has executable but backendsDir was empty
	if hasBinExe && len(results) == 0 {
		tag := activeTag
		if tag == "" {
			tag = "Installed"
		}
		variant := activeVariant
		if variant == "" {
			variant = defaultBackendForHost()
		}
		id := fmt.Sprintf("%s-%s", sanitizeDirName(tag), variant)
		legacyTarget := filepath.Join(backendsDir, id)
		_ = os.MkdirAll(legacyTarget, 0755)
		_ = copyDir(rm.binDir, legacyTarget)
		_ = os.WriteFile(filepath.Join(legacyTarget, "version.txt"), []byte(fmt.Sprintf("%s\n%s\n%s\n", tag, variant, id)), 0644)
		_ = os.WriteFile(filepath.Join(rm.binDir, "version.txt"), []byte(fmt.Sprintf("%s\n%s\n%s\n", tag, variant, id)), 0644)

		sz := dirSize(legacyTarget)
		results = append(results, InstalledVersionInfo{
			Id:             id,
			Tag:            tag,
			Variant:        variant,
			VariantName:    formatBackendDisplayName(variant),
			ExecutablePath: filepath.Join(legacyTarget, serverExeName),
			SizeBytes:      sz,
			SizeDisplay:    formatBytes(sz),
			InstalledAt:    time.Now().Format("2006-01-02 15:04"),
			IsActive:       true,
		})
		foundActive = true
	} else if hasBinExe && !foundActive && len(results) > 0 {
		for i := range results {
			if results[i].Tag == activeTag || results[i].Variant == activeVariant {
				results[i].IsActive = true
				break
			}
		}
	}

	sort.Slice(results, func(i, j int) bool {
		if results[i].IsActive != results[j].IsActive {
			return results[i].IsActive
		}
		if results[i].Tag != results[j].Tag {
			return results[i].Tag > results[j].Tag
		}
		return results[i].Variant < results[j].Variant
	})

	return results
}

func removeDirOrLink(path string) error {
	fi, err := os.Lstat(path)
	if err != nil {
		return nil // doesn't exist
	}

	if runtime.GOOS == "windows" {
		// If junction or symlink, rmdir cleanly unlinks without deleting target contents
		if fi.Mode()&os.ModeSymlink != 0 {
			_ = os.Remove(path)
		}
		cmd := prepareCmd(exec.Command("cmd", "/c", "rmdir", path))
		if err := cmd.Run(); err == nil {
			return nil
		}
		// If regular directory, try rmdir /s /q with backoff retries
		for attempt := 0; attempt < 3; attempt++ {
			cmdAll := prepareCmd(exec.Command("cmd", "/c", "rmdir", "/s", "/q", path))
			if err := cmdAll.Run(); err == nil {
				if _, statErr := os.Stat(path); os.IsNotExist(statErr) {
					return nil
				}
			}
			time.Sleep(100 * time.Millisecond)
		}
		return os.RemoveAll(path)
	}

	_ = os.Remove(path)
	return os.RemoveAll(path)
}

func linkOrMirrorDir(targetDir, linkDir string) error {
	_ = removeDirOrLink(linkDir)

	// 1. On Windows, try creating an NTFS directory junction (instant, no admin elevation required)
	if runtime.GOOS == "windows" {
		cmd := prepareCmd(exec.Command("cmd", "/c", "mklink", "/J", linkDir, targetDir))
		if out, err := cmd.CombinedOutput(); err == nil && !strings.Contains(strings.ToLower(string(out)), "error") {
			return nil
		}
	}

	// 2. On Unix (macOS / Linux), try creating a standard directory symlink
	if runtime.GOOS != "windows" {
		if err := os.Symlink(targetDir, linkDir); err == nil {
			if runtime.GOOS == "darwin" {
				_ = exec.Command("xattr", "-cr", linkDir).Run()
				if realTarget, rErr := filepath.EvalSymlinks(linkDir); rErr == nil && realTarget != linkDir {
					_ = exec.Command("xattr", "-cr", realTarget).Run()
				}
			}
			return nil
		}
	}

	// 3. Fallback: Mirror directory files directly
	_ = os.MkdirAll(linkDir, 0755)
	return copyDir(targetDir, linkDir)
}

func (rm *ReleaseManager) GetVersionDir(id string) string {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	backendsDir := rm.getBackendsDirLocked()
	targetDir := filepath.Join(backendsDir, id)
	if _, err := os.Stat(filepath.Join(targetDir, getServerExeName())); err == nil {
		return targetDir
	}

	entries, _ := os.ReadDir(backendsDir)
	for _, e := range entries {
		if e.IsDir() && (e.Name() == id || strings.HasSuffix(e.Name(), "-"+id)) {
			cand := filepath.Join(backendsDir, e.Name())
			if _, err := os.Stat(filepath.Join(cand, getServerExeName())); err == nil {
				return cand
			}
		}
	}
	return ""
}

func (rm *ReleaseManager) ActivateVersion(id string) error {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	backendsDir := rm.getBackendsDirLocked()
	targetDir := filepath.Join(backendsDir, id)
	targetExe := filepath.Join(targetDir, getServerExeName())

	if _, err := os.Stat(targetExe); err != nil {
		backendsDirs := []string{backendsDir}
		if rm.appDataDir != "" && filepath.Join(rm.appDataDir, "backends") != backendsDir {
			backendsDirs = append(backendsDirs, filepath.Join(rm.appDataDir, "backends"))
		}
		found := false
		for _, bDir := range backendsDirs {
			entries, _ := os.ReadDir(bDir)
			for _, e := range entries {
				if e.IsDir() && (e.Name() == id || strings.HasSuffix(e.Name(), "-"+id)) {
					cand := filepath.Join(bDir, e.Name())
					if _, err := os.Stat(filepath.Join(cand, getServerExeName())); err == nil {
						targetDir = cand
						targetExe = filepath.Join(cand, getServerExeName())
						id = e.Name()
						found = true
						break
					}
				}
			}
			if found {
				break
			}
		}
		if !found {
			return fmt.Errorf("version %s is not installed in %s", id, targetDir)
		}
	}

	if runtime.GOOS != "windows" {
		_ = os.Chmod(targetExe, 0755)
		entries, _ := os.ReadDir(targetDir)
		for _, e := range entries {
			if !e.IsDir() {
				_ = os.Chmod(filepath.Join(targetDir, e.Name()), 0755)
			}
		}
	}
	if runtime.GOOS == "darwin" {
		_ = exec.Command("xattr", "-cr", targetDir).Run()
		if realT, err := filepath.EvalSymlinks(targetDir); err == nil && realT != targetDir {
			_ = exec.Command("xattr", "-cr", realT).Run()
		}
	}

	tag := ""
	variant := ""
	if vData, err := os.ReadFile(filepath.Join(targetDir, "version.txt")); err == nil {
		lines := strings.Split(strings.TrimSpace(string(vData)), "\n")
		if len(lines) > 0 {
			tag = strings.TrimSpace(lines[0])
		}
		if len(lines) > 1 {
			variant = strings.TrimSpace(lines[1])
		}
	}
	_ = os.WriteFile(filepath.Join(targetDir, "version.txt"), []byte(fmt.Sprintf("%s\n%s\n%s\n", tag, variant, id)), 0644)

	// Link or mirror into rm.binDir
	if err := linkOrMirrorDir(targetDir, rm.binDir); err != nil {
		return err
	}

	_ = os.WriteFile(filepath.Join(rm.binDir, "version.txt"), []byte(fmt.Sprintf("%s\n%s\n%s\n", tag, variant, id)), 0644)

	return nil
}

func (rm *ReleaseManager) UninstallVersion(id string) error {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	backendsDir := rm.getBackendsDirLocked()
	targetDir := filepath.Join(backendsDir, id)
	_ = removeDirOrLink(targetDir)

	binVersionFile := filepath.Join(rm.binDir, "version.txt")
	isActive := false
	if data, err := os.ReadFile(binVersionFile); err == nil {
		lines := strings.Split(strings.TrimSpace(string(data)), "\n")
		if len(lines) >= 3 && strings.TrimSpace(lines[2]) == id {
			isActive = true
		} else if id == filepath.Base(targetDir) {
			if len(lines) >= 2 {
				tag := strings.TrimSpace(lines[0])
				variant := strings.TrimSpace(lines[1])
				expectedId := fmt.Sprintf("%s-%s", sanitizeDirName(tag), variant)
				if id == expectedId || id == variant {
					isActive = true
				}
			}
		}
	}

	if isActive {
		_ = removeDirOrLink(rm.binDir)
		_ = os.MkdirAll(rm.binDir, 0755)
	}

	return nil
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

	backendsDir := rm.getBackendsDirLocked()
	targetDir := filepath.Join(backendsDir, variant)
	_ = os.RemoveAll(targetDir)

	entries, _ := os.ReadDir(backendsDir)
	for _, e := range entries {
		if e.IsDir() && strings.HasSuffix(e.Name(), "-"+variant) {
			_ = os.RemoveAll(filepath.Join(backendsDir, e.Name()))
		}
	}

	if variant == activeVariant || activeVariant == "" {
		_ = os.RemoveAll(rm.binDir)
		_ = os.MkdirAll(rm.binDir, 0755)
	}

	return nil
}

func (rm *ReleaseManager) UninstallAllBackends() error {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	backendsDir := rm.getBackendsDirLocked()
	_ = os.RemoveAll(backendsDir)
	_ = os.MkdirAll(backendsDir, 0755)

	_ = os.RemoveAll(rm.binDir)
	_ = os.MkdirAll(rm.binDir, 0755)

	return nil
}

func (rm *ReleaseManager) ActivateBackend(variant string) error {
	backendsDir := rm.getBackendsDirLocked()
	targetDir := filepath.Join(backendsDir, variant)
	targetExe := filepath.Join(targetDir, getServerExeName())

	var chosenId string
	if _, err := os.Stat(targetExe); err == nil {
		chosenId = variant
	} else {
		entries, _ := os.ReadDir(backendsDir)
		for i := len(entries) - 1; i >= 0; i-- {
			e := entries[i]
			if !e.IsDir() {
				continue
			}
			candDir := filepath.Join(backendsDir, e.Name())
			if _, err := os.Stat(filepath.Join(candDir, getServerExeName())); err != nil {
				continue
			}
			if vData, err := os.ReadFile(filepath.Join(candDir, "version.txt")); err == nil {
				lines := strings.Split(strings.TrimSpace(string(vData)), "\n")
				if len(lines) > 1 && strings.TrimSpace(lines[1]) == variant {
					chosenId = e.Name()
					break
				}
			} else if strings.HasSuffix(e.Name(), "-"+variant) {
				chosenId = e.Name()
				break
			}
		}
	}

	if chosenId == "" {
		return fmt.Errorf("backend %s is not installed in %s", variant, backendsDir)
	}

	return rm.ActivateVersion(chosenId)
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

// DownloadAndInstall downloads the requested release tag and variant.
// Supported variants depend on the host OS: darwin offers "metal" and "cpu";
// linux offers "cuda", "rocm", "vulkan" and "cpu"; windows offers "cuda",
// "vulkan" and "cpu". An empty variant resolves to the OS default.
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
		debug.FreeOSMemory()
		trimProcessWorkingSet()
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

	if strings.TrimSpace(variant) == "" {
		variant = defaultBackendForHost()
	}

	// Match primary asset and optional CUDA runtime asset.
	// Scoring (see scorePrimaryAsset) picks the best host-compatible archive
	// for the variant and never mistakes specialized builds (OpenVINO, SYCL,
	// OpenCL, Snapdragon) for a generic backend.
	var primaryAsset *ReleaseAsset
	var cudartAsset *ReleaseAsset

	// Step 1: Find best primary binary archive for variant
	bestScore := -1
	for i := range targetRelease.Assets {
		a := &targetRelease.Assets[i]
		if !isAssetCompatibleWithHost(a.Name) {
			continue
		}
		lower := strings.ToLower(a.Name)
		if s := scorePrimaryAsset(lower, variant); s > bestScore {
			bestScore = s
			primaryAsset = a
		}
	}

	if primaryAsset == nil {
		return fmt.Errorf("no matching binary found for %s on %s/%s in release %s", variant, runtime.GOOS, runtime.GOARCH, tag)
	}

	// Step 2: For CUDA, match the CUDA runtime archive with the same toolkit
	// version as the primary asset (e.g. primary cuda-12.8 pairs with
	// cudart cuda-12.8), regardless of host OS archive layout.
	if variant == "cuda" && primaryAsset != nil {
		targetVer := cudaVersionOf(strings.ToLower(primaryAsset.Name))

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

	versionId := fmt.Sprintf("%s-%s", sanitizeDirName(tag), variant)
	if strings.TrimSpace(tag) == "" {
		versionId = variant
	}

	backendsDir := rm.getBackendsDirLocked()
	targetDir := filepath.Join(backendsDir, versionId)
	_ = os.RemoveAll(targetDir)
	_ = os.MkdirAll(targetDir, 0755)

	tempDir := filepath.Join(backendsDir, ".tmp_download")
	_ = os.RemoveAll(tempDir)
	_ = os.MkdirAll(tempDir, 0755)
	defer os.RemoveAll(tempDir)

	// Step 3: Download & extract CUDA runtime libraries if applicable
	if variant == "cuda" && cudartAsset != nil {
		cudartArchive := filepath.Join(tempDir, cudartAsset.Name)
		if err := rm.downloadFile(cudartAsset.DownloadURL, cudartArchive, cudartAsset.Name, cudartAsset.Size, onProgress); err != nil {
			return err
		}
		if err := rm.extractArchive(cudartArchive, targetDir, onProgress); err != nil {
			return err
		}
	}

	// Step 4: Download & extract primary llama binaries
	primaryArchive := filepath.Join(tempDir, primaryAsset.Name)
	if err := rm.downloadFile(primaryAsset.DownloadURL, primaryArchive, primaryAsset.Name, primaryAsset.Size, onProgress); err != nil {
		return err
	}
	if err := rm.extractArchive(primaryArchive, targetDir, onProgress); err != nil {
		return err
	}

	// Ensure llama-server executable and all companion libraries are directly at the root of targetDir
	serverExeName := getServerExeName()
	serverExe := filepath.Join(targetDir, serverExeName)
	if _, err := os.Stat(serverExe); err != nil {
		var subDirWithExe string
		_ = filepath.Walk(targetDir, func(path string, info os.FileInfo, err error) error {
			if err == nil && !info.IsDir() && info.Name() == serverExeName {
				subDirWithExe = filepath.Dir(path)
				return io.EOF
			}
			return nil
		})
		if subDirWithExe != "" && subDirWithExe != targetDir {
			entries, _ := os.ReadDir(subDirWithExe)
			for _, e := range entries {
				oldPath := filepath.Join(subDirWithExe, e.Name())
				newPath := filepath.Join(targetDir, e.Name())
				_ = os.Rename(oldPath, newPath)
			}
			_ = os.Remove(subDirWithExe)
		}
	}

	if runtime.GOOS != "windows" {
		_ = os.Chmod(serverExe, 0755)
		entries, _ := os.ReadDir(targetDir)
		for _, e := range entries {
			if !e.IsDir() {
				_ = os.Chmod(filepath.Join(targetDir, e.Name()), 0755)
			}
		}
	}
	if runtime.GOOS == "darwin" {
		_ = exec.Command("xattr", "-cr", targetDir).Run()
		if realT, err := filepath.EvalSymlinks(targetDir); err == nil && realT != targetDir {
			_ = exec.Command("xattr", "-cr", realT).Run()
		}
	}

	// Write version tag, variant, and versionId in targetDir
	versionFile := filepath.Join(targetDir, "version.txt")
	_ = os.WriteFile(versionFile, []byte(fmt.Sprintf("%s\n%s\n%s\n", tag, variant, versionId)), 0644)

	// Step 5: Link or mirror targetDir cleanly into rm.binDir as the active backend
	_ = linkOrMirrorDir(targetDir, rm.binDir)

	binVersionFile := filepath.Join(rm.binDir, "version.txt")
	_ = os.WriteFile(binVersionFile, []byte(fmt.Sprintf("%s\n%s\n%s\n", tag, variant, versionId)), 0644)

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

type deferredLink struct {
	targetPath string
	linkTarget string
	isSymlink  bool
}

func resolveFinalTarget(baseDir string, linkMap map[string]string, targetName string) string {
	currName := filepath.Base(targetName)
	visited := make(map[string]bool)
	for depth := 0; depth < 16; depth++ {
		if visited[currName] {
			break
		}
		visited[currName] = true
		if next, ok := linkMap[currName]; ok {
			currName = filepath.Base(next)
		} else {
			break
		}
	}

	currPath := filepath.Join(baseDir, currName)
	if filepath.IsAbs(targetName) && currName == filepath.Base(targetName) {
		if _, err := os.Lstat(targetName); err == nil {
			currPath = targetName
		}
	}

	for depth := 0; depth < 16; depth++ {
		fi, err := os.Lstat(currPath)
		if err != nil {
			break
		}
		if fi.Mode()&os.ModeSymlink != 0 {
			link, err := os.Readlink(currPath)
			if err != nil {
				break
			}
			if filepath.IsAbs(link) {
				currPath = link
			} else {
				currPath = filepath.Join(filepath.Dir(currPath), link)
			}
			continue
		}
		break
	}
	return currPath
}

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()

	_ = os.Chmod(dst, 0666)

	var out *os.File
	var openErr error
	for attempt := 0; attempt < 3; attempt++ {
		out, openErr = os.OpenFile(dst, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0755)
		if openErr == nil {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}
	if openErr != nil {
		return openErr
	}
	defer out.Close()

	_, err = io.Copy(out, in)
	return err
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

	var deferredLinks []deferredLink
	linkMap := make(map[string]string)

	for _, f := range r.File {
		cleanName := filepath.Clean(f.Name)
		if strings.HasPrefix(cleanName, "..") {
			continue
		}
		baseName := filepath.Base(cleanName)
		if baseName == "." || baseName == "/" || baseName == "\\" {
			continue
		}
		targetPath := filepath.Join(dstDir, baseName)

		if f.FileInfo().IsDir() {
			continue
		}

		rc, err := f.Open()
		if err != nil {
			return err
		}

		// Handle symlink entry in zip
		if f.Mode()&os.ModeSymlink != 0 {
			targetBytes, err := io.ReadAll(rc)
			rc.Close()
			if err == nil {
				targetName := filepath.Base(strings.TrimSpace(string(targetBytes)))
				deferredLinks = append(deferredLinks, deferredLink{
					targetPath: targetPath,
					linkTarget: targetName,
					isSymlink:  true,
				})
				linkMap[baseName] = targetName
			}
			continue
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

		if runtime.GOOS != "windows" {
			perm := f.Mode().Perm()
			if strings.HasPrefix(baseName, "llama-") || strings.HasPrefix(baseName, "ggml-") || strings.Contains(baseName, ".so") || strings.Contains(baseName, ".dylib") {
				perm |= 0755
			}
			_ = os.Chmod(targetPath, perm)
		}
	}

	if runtime.GOOS != "windows" {
		for _, l := range deferredLinks {
			if l.isSymlink {
				_ = os.Remove(l.targetPath)
				_ = os.Symlink(l.linkTarget, l.targetPath)
			}
		}
	}

	for _, l := range deferredLinks {
		needFallback := true
		if runtime.GOOS != "windows" && l.isSymlink {
			if _, err := os.Stat(l.targetPath); err == nil {
				needFallback = false
			} else {
				_ = os.Remove(l.targetPath)
			}
		}
		if needFallback {
			resolved := resolveFinalTarget(dstDir, linkMap, l.linkTarget)
			if _, err := os.Stat(resolved); err == nil {
				_ = copyFile(resolved, l.targetPath)
				if runtime.GOOS != "windows" {
					_ = os.Chmod(l.targetPath, 0755)
				}
			}
		}
	}

	if runtime.GOOS == "darwin" {
		_ = exec.Command("xattr", "-cr", dstDir).Run()
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

	var deferredLinks []deferredLink
	linkMap := make(map[string]string)

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
		baseName := filepath.Base(cleanName)
		if baseName == "." || baseName == "/" || baseName == "\\" {
			continue
		}
		targetPath := filepath.Join(dstDir, baseName)

		if header.Typeflag == tar.TypeDir {
			continue
		}

		// Handle Symlinks and Hardlinks
		if header.Typeflag == tar.TypeSymlink || header.Typeflag == tar.TypeLink {
			targetName := filepath.Base(header.Linkname)
			deferredLinks = append(deferredLinks, deferredLink{
				targetPath: targetPath,
				linkTarget: targetName,
				isSymlink:  (header.Typeflag == tar.TypeSymlink),
			})
			linkMap[baseName] = targetName
			continue
		}

		// Regular file
		outFile, err := os.OpenFile(targetPath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, os.FileMode(header.Mode))
		if err != nil {
			return err
		}

		if _, err := io.Copy(outFile, tarReader); err != nil {
			outFile.Close()
			return err
		}
		outFile.Close()

		if runtime.GOOS != "windows" {
			perm := os.FileMode(header.Mode).Perm()
			if strings.HasPrefix(baseName, "llama-") || strings.HasPrefix(baseName, "ggml-") || strings.Contains(baseName, ".so") || strings.Contains(baseName, ".dylib") {
				perm |= 0755
			}
			_ = os.Chmod(targetPath, perm)
		}
	}

	if runtime.GOOS != "windows" {
		for _, l := range deferredLinks {
			if l.isSymlink {
				_ = os.Remove(l.targetPath)
				_ = os.Symlink(l.linkTarget, l.targetPath)
			}
		}
	}

	for _, l := range deferredLinks {
		needFallback := true
		if runtime.GOOS != "windows" && l.isSymlink {
			if _, err := os.Stat(l.targetPath); err == nil {
				needFallback = false
			} else {
				_ = os.Remove(l.targetPath)
			}
		}
		if needFallback {
			resolved := resolveFinalTarget(dstDir, linkMap, l.linkTarget)
			if _, err := os.Stat(resolved); err == nil {
				_ = copyFile(resolved, l.targetPath)
				if runtime.GOOS != "windows" {
					_ = os.Chmod(l.targetPath, 0755)
				}
			}
		}
	}

	if runtime.GOOS == "darwin" {
		_ = exec.Command("xattr", "-cr", dstDir).Run()
	}

	return nil
}

func copyDir(src, dst string) error {
	_ = os.MkdirAll(dst, 0755)
	entries, err := os.ReadDir(src)
	if err != nil {
		return err
	}

	type symlinkEntry struct {
		dstPath string
		target  string
	}
	var symlinks []symlinkEntry
	linkMap := make(map[string]string)

	for _, entry := range entries {
		srcPath := filepath.Join(src, entry.Name())
		dstPath := filepath.Join(dst, entry.Name())

		lstat, err := os.Lstat(srcPath)
		if err != nil {
			continue
		}

		if lstat.Mode()&os.ModeSymlink != 0 {
			target, err := os.Readlink(srcPath)
			if err == nil {
				symlinks = append(symlinks, symlinkEntry{
					dstPath: dstPath,
					target:  target,
				})
				linkMap[entry.Name()] = filepath.Base(target)
			}
			continue
		}

		if entry.IsDir() {
			if err := copyDir(srcPath, dstPath); err != nil {
				return err
			}
		} else {
			if err := copyFile(srcPath, dstPath); err != nil {
				return err
			}
			if runtime.GOOS != "windows" {
				_ = os.Chmod(dstPath, lstat.Mode().Perm()|0755)
			}
		}
	}

	if runtime.GOOS != "windows" {
		for _, sl := range symlinks {
			_ = os.Remove(sl.dstPath)
			_ = os.Symlink(sl.target, sl.dstPath)
		}
	}

	for _, sl := range symlinks {
		needFallback := true
		if runtime.GOOS != "windows" {
			if _, err := os.Stat(sl.dstPath); err == nil {
				needFallback = false
			} else {
				_ = os.Remove(sl.dstPath)
			}
		}
		if needFallback {
			resolved := resolveFinalTarget(src, linkMap, sl.target)
			if _, err := os.Stat(resolved); err == nil {
				_ = copyFile(resolved, sl.dstPath)
				if runtime.GOOS != "windows" {
					_ = os.Chmod(sl.dstPath, 0755)
				}
			}
		}
	}

	if runtime.GOOS == "darwin" {
		_ = exec.Command("xattr", "-cr", dst).Run()
	}

	return nil
}

