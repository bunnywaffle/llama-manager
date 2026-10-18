package main

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"

	"github.com/egoist/mygo"
)

// Typed events emitted to the frontend
var (
	ServerLogEvent        = mygo.NewEvent[string]("server_log")
	DownloadProgressEvent = mygo.NewEvent[ProgressReport]("download_progress")
	ServerStatusEvent     = mygo.NewEvent[ServerStatus]("server_status")
)

type AppSettings struct {
	ModelsDir     string `json:"modelsDir"`
	AppDataDir    string `json:"appDataDir"`
	ActiveBackend string `json:"activeBackend"` // "cuda", "vulkan", "cpu"
	ActiveVersion string `json:"activeVersion,omitempty"`
	DefaultModel  string `json:"defaultModel"`
}

// LlamaService is the bound Go service callable from TypeScript.
type LlamaService struct {
	appDataDir  string
	settings    AppSettings
	hwInfo      HardwareInfo
	releaseMgr  *ReleaseManager
	modelMgr    *ModelManager
	serverMgr   *ServerManager
	win         *mygo.Window
	mu          sync.Mutex
	initialized bool
}

func NewLlamaService(appDataDir string) *LlamaService {
	binDir := filepath.Join(appDataDir, "bin")
	modelsDir := filepath.Join(appDataDir, "models")
	settingsPath := filepath.Join(appDataDir, "settings.json")

	hw := detectHardware()
	initialBackend := hw.RecommendedBackend
	if strings.TrimSpace(initialBackend) == "" {
		initialBackend = defaultBackendForHost()
	}

	settings := AppSettings{
		ModelsDir:     modelsDir,
		AppDataDir:    appDataDir,
		ActiveBackend: initialBackend,
		DefaultModel:  "",
	}

	if data, err := os.ReadFile(settingsPath); err == nil {
		_ = json.Unmarshal(data, &settings)
		if settings.ModelsDir != "" {
			modelsDir = settings.ModelsDir
		}
		if settings.AppDataDir != "" {
			appDataDir = settings.AppDataDir
			binDir = filepath.Join(appDataDir, "bin")
		}
	}

	svc := &LlamaService{
		appDataDir: appDataDir,
		settings:   settings,
		hwInfo:     hw,
		releaseMgr: NewReleaseManager(binDir, appDataDir),
		modelMgr:   NewModelManager(modelsDir, appDataDir),
		serverMgr:  NewServerManager(binDir, appDataDir),
	}

	svc.serverMgr.SetLogCallback(func(line string) {
		svc.mu.Lock()
		win := svc.win
		svc.mu.Unlock()
		if win != nil {
			_ = ServerLogEvent.Emit(win, line)
		}
	})

	svc.serverMgr.SetStatusCallback(func(st ServerStatus) {
		svc.mu.Lock()
		win := svc.win
		svc.mu.Unlock()
		if win != nil {
			_ = ServerStatusEvent.Emit(win, st)
		}
	})

	// Ensure any zombie server processes from prior unclean shutdowns are purged to free GPU VRAM
	_ = KillZombieProcesses()

	return svc
}

func (s *LlamaService) SetWindow(win *mygo.Window) {
	s.mu.Lock()
	s.win = win
	s.mu.Unlock()
}

// GetHardware returns CPU and detected GPU details.
func (s *LlamaService) GetHardware() HardwareInfo {
	return s.hwInfo
}

// GetSettings returns current application settings.
func (s *LlamaService) GetSettings() AppSettings {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.settings
}

// SaveSettings updates and persists the application settings.
func (s *LlamaService) SaveSettings(settings AppSettings) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.settings = settings
	if settings.ModelsDir != "" {
		_ = s.modelMgr.SetModelsDir(settings.ModelsDir)
	}
	if settings.AppDataDir != "" {
		s.appDataDir = settings.AppDataDir
		binDir := filepath.Join(settings.AppDataDir, "bin")
		s.releaseMgr.SetBinDir(binDir)
		s.serverMgr.SetBinDir(binDir)
	}

	return s.saveSettingsToFile()
}

func (s *LlamaService) saveSettingsToFile() error {
	settingsPath := filepath.Join(s.appDataDir, "settings.json")
	data, err := json.MarshalIndent(s.settings, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(settingsPath, data, 0644)
}

// SelectFolderDialog opens native folder picker to select a directory.
func (s *LlamaService) SelectFolderDialog(title string) (string, error) {
	s.mu.Lock()
	win := s.win
	s.mu.Unlock()

	paths, err := mygo.Dialog.Open(mygo.OpenDialogOptions{
		Parent:    win,
		Title:     title,
		Directory: true,
		Multiple:  false,
	})
	if err != nil {
		return "", err
	}
	if len(paths) == 0 || paths[0] == "" {
		return "", nil // cancelled
	}
	return paths[0], nil
}

// SelectFileDialog opens a native file picker to select a file with filter.
func (s *LlamaService) SelectFileDialog(title string, filterName string, extension string) (string, error) {
	s.mu.Lock()
	win := s.win
	s.mu.Unlock()

	filters := []mygo.FileFilter{
		{Name: fmt.Sprintf("%s (*.%s)", filterName, extension), Extensions: []string{extension}},
		{Name: "All Files (*.*)", Extensions: []string{"*"}},
	}

	paths, err := mygo.Dialog.Open(mygo.OpenDialogOptions{
		Parent:   win,
		Title:    title,
		Filters:  filters,
		Multiple: false,
	})
	if err != nil {
		return "", err
	}
	if len(paths) == 0 || paths[0] == "" {
		return "", nil // cancelled
	}
	return paths[0], nil
}

// SaveSnippetDialog opens a native save file dialog and writes the code snippet to disk.
func (s *LlamaService) SaveSnippetDialog(defaultFilename, content string) (string, error) {
	s.mu.Lock()
	win := s.win
	s.mu.Unlock()

	ext := filepath.Ext(defaultFilename)
	var filters []mygo.FileFilter
	if ext != "" {
		trimmedExt := strings.TrimPrefix(ext, ".")
		filters = append(filters, mygo.FileFilter{
			Name:       fmt.Sprintf("%s files (*%s)", strings.ToUpper(trimmedExt), ext),
			Extensions: []string{trimmedExt},
		})
	}
	filters = append(filters, mygo.FileFilter{
		Name:       "All Files (*.*)",
		Extensions: []string{"*"},
	})

	filePath, err := mygo.Dialog.Save(mygo.SaveDialogOptions{
		Parent:      win,
		Title:       "Save Code Snippet",
		DefaultPath: defaultFilename,
		Filters:     filters,
	})
	if err != nil {
		return "", err
	}
	if filePath == "" {
		return "", nil // user cancelled
	}

	if err := os.WriteFile(filePath, []byte(content), 0644); err != nil {
		return "", fmt.Errorf("failed to save file: %w", err)
	}

	return filePath, nil
}

// OpenFolderInExplorer opens the specified directory in the host OS's native file manager.
func (s *LlamaService) OpenFolderInExplorer(dir string) error {
	if dir == "" {
		dir = s.appDataDir
	}
	_ = os.MkdirAll(dir, 0755)

	switch runtime.GOOS {
	case "windows":
		return prepareCmd(exec.Command("explorer.exe", dir)).Start()
	case "darwin":
		return prepareCmd(exec.Command("open", dir)).Start()
	default:
		return prepareCmd(exec.Command("xdg-open", dir)).Start()
	}
}

// SwitchBackend downloads or switches the active backend variant
// (cuda, vulkan, cpu, metal, rocm depending on OS).
func (s *LlamaService) SwitchBackend(variant string) error {
	current := s.releaseMgr.GetInstalledBackend()
	tag := current.Version
	if tag == "" || tag == "Custom / Unspecified" || tag == "None" {
		tag = ""
	}

	err := s.releaseMgr.DownloadAndInstall(tag, variant, func(p ProgressReport) {
		s.mu.Lock()
		win := s.win
		s.mu.Unlock()
		if win != nil {
			_ = DownloadProgressEvent.Emit(win, p)
		}
	})

	if err == nil {
		s.mu.Lock()
		s.settings.ActiveBackend = variant
		_ = s.saveSettingsToFile()
		s.mu.Unlock()
	}

	return err
}

// repairVariantForHost resolves the backend variant a repair action should
// (re)install: the configured backend when usable on this OS, otherwise the
// OS default. Notably, CUDA/Metal are not available everywhere.
func (s *LlamaService) repairVariantForHost(configured string) string {
	v := strings.TrimSpace(configured)
	if v == "" {
		if rec := strings.TrimSpace(s.hwInfo.RecommendedBackend); rec != "" {
			v = rec
		} else {
			v = defaultBackendForHost()
		}
	}
	switch runtime.GOOS {
	case "darwin":
		if v == "cuda" || v == "vulkan" || v == "rocm" {
			return "metal"
		}
	case "windows":
		if v == "metal" || v == "rocm" {
			return "cpu"
		}
	case "linux":
		if v == "metal" {
			return "cpu"
		}
	}
	return v
}

// GetInstalledBackend checks if the server executable is installed and which version/variant.
func (s *LlamaService) GetInstalledBackend() BackendStatus {
	return s.releaseMgr.GetInstalledBackend()
}

// GetInstalledBackends returns all available backend variants and their installation/active status.
func (s *LlamaService) GetInstalledBackends() []BackendVariantInfo {
	s.mu.Lock()
	active := s.settings.ActiveBackend
	s.mu.Unlock()
	return s.releaseMgr.GetInstalledBackends(active)
}

// SetActiveBackend switches the current active backend to an installed variant.
func (s *LlamaService) SetActiveBackend(variant string) error {
	// Stop running server before switching
	_ = s.serverMgr.Stop()

	if err := s.releaseMgr.ActivateBackend(variant); err != nil {
		return err
	}

	s.mu.Lock()
	s.settings.ActiveBackend = variant
	_ = s.saveSettingsToFile()
	win := s.win
	s.mu.Unlock()

	if win != nil {
		_ = ServerStatusEvent.Emit(win, s.serverMgr.GetStatus())
	}

	return nil
}

// UninstallBackend uninstalls a specific backend variant and cleans up binaries.
func (s *LlamaService) UninstallBackend(variant string) error {
	s.mu.Lock()
	active := s.settings.ActiveBackend
	s.mu.Unlock()

	if active == variant {
		_ = s.serverMgr.Stop()
	}

	if err := s.releaseMgr.UninstallBackend(variant, active); err != nil {
		return err
	}

	// Update active backend if the uninstalled one was active
	if active == variant {
		installed := s.releaseMgr.GetInstalledBackends("")
		var newActive string
		for _, b := range installed {
			if b.Installed {
				newActive = b.Variant
				break
			}
		}

		if newActive != "" {
			_ = s.releaseMgr.ActivateBackend(newActive)
		}

		s.mu.Lock()
		s.settings.ActiveBackend = newActive
		_ = s.saveSettingsToFile()
		s.mu.Unlock()
	}

	s.mu.Lock()
	win := s.win
	s.mu.Unlock()
	if win != nil {
		_ = ServerStatusEvent.Emit(win, s.serverMgr.GetStatus())
	}

	return nil
}

// UninstallAllBackends completely deletes all installed backends and clears the bin folder.
func (s *LlamaService) UninstallAllBackends() error {
	_ = s.serverMgr.Stop()

	if err := s.releaseMgr.UninstallAllBackends(); err != nil {
		return err
	}

	s.mu.Lock()
	s.settings.ActiveBackend = ""
	_ = s.saveSettingsToFile()
	win := s.win
	s.mu.Unlock()

	if win != nil {
		_ = ServerStatusEvent.Emit(win, s.serverMgr.GetStatus())
	}

	return nil
}

// GetInstalledVersions returns all installed llama.cpp versions across all backends.
func (s *LlamaService) GetInstalledVersions() []InstalledVersionInfo {
	return s.releaseMgr.GetInstalledVersions()
}

// ActivateVersion sets a specific installed version (by ID) as the active/current runtime.
func (s *LlamaService) ActivateVersion(id string) error {
	_ = s.serverMgr.Stop()

	if err := s.releaseMgr.ActivateVersion(id); err != nil {
		return err
	}

	current := s.releaseMgr.GetInstalledBackend()
	s.mu.Lock()
	s.settings.ActiveBackend = current.Variant
	s.settings.ActiveVersion = current.Version
	_ = s.saveSettingsToFile()
	win := s.win
	s.mu.Unlock()

	if win != nil {
		_ = ServerStatusEvent.Emit(win, s.serverMgr.GetStatus())
	}
	return nil
}

// UninstallVersion uninstalls a specific installed version by ID.
func (s *LlamaService) UninstallVersion(id string) error {
	current := s.releaseMgr.GetInstalledBackend()
	binVersionFile := filepath.Join(s.releaseMgr.binDir, "version.txt")
	isActive := false
	if data, err := os.ReadFile(binVersionFile); err == nil {
		lines := strings.Split(strings.TrimSpace(string(data)), "\n")
		if len(lines) >= 3 && strings.TrimSpace(lines[2]) == id {
			isActive = true
		} else if strings.Contains(id, current.Version) && strings.Contains(id, current.Variant) {
			isActive = true
		} else if id == current.Variant {
			isActive = true
		}
	}

	if isActive {
		_ = s.serverMgr.Stop()
	}

	if err := s.releaseMgr.UninstallVersion(id); err != nil {
		return err
	}

	if isActive {
		remaining := s.releaseMgr.GetInstalledVersions()
		if len(remaining) > 0 {
			_ = s.releaseMgr.ActivateVersion(remaining[0].Id)
			s.mu.Lock()
			s.settings.ActiveBackend = remaining[0].Variant
			s.settings.ActiveVersion = remaining[0].Tag
			_ = s.saveSettingsToFile()
			s.mu.Unlock()
		} else {
			s.mu.Lock()
			s.settings.ActiveBackend = ""
			s.settings.ActiveVersion = ""
			_ = s.saveSettingsToFile()
			s.mu.Unlock()
		}
	}

	s.mu.Lock()
	win := s.win
	s.mu.Unlock()
	if win != nil {
		_ = ServerStatusEvent.Emit(win, s.serverMgr.GetStatus())
	}
	return nil
}

// CheckReleases queries GitHub for recent llama.cpp releases.
func (s *LlamaService) CheckReleases() ([]ReleaseInfo, error) {
	return s.releaseMgr.CheckReleases()
}

// DownloadBackend downloads and installs the selected release tag and variant.
func (s *LlamaService) DownloadBackend(tag, variant string) error {
	_ = s.serverMgr.Stop()

	err := s.releaseMgr.DownloadAndInstall(tag, variant, func(p ProgressReport) {
		s.mu.Lock()
		win := s.win
		s.mu.Unlock()
		if win != nil {
			_ = DownloadProgressEvent.Emit(win, p)
		}
	})
	if err == nil {
		current := s.releaseMgr.GetInstalledBackend()
		s.mu.Lock()
		s.settings.ActiveBackend = current.Variant
		s.settings.ActiveVersion = current.Version
		_ = s.saveSettingsToFile()
		win := s.win
		s.mu.Unlock()

		if win != nil {
			_ = ServerStatusEvent.Emit(win, s.serverMgr.GetStatus())
		}
	}
	return err
}

// ListModels scans the models directory and returns all .gguf files.
func (s *LlamaService) ListModels() ([]ModelInfo, error) {
	return s.modelMgr.ListModels()
}

// FetchHFRepo queries Hugging Face API to list .gguf files in a repository.
func (s *LlamaService) FetchHFRepo(repoID string) ([]HuggingFaceFile, error) {
	return s.modelMgr.FetchHFRepoGGUFs(repoID)
}

// DownloadModel downloads a .gguf file from a URL.
func (s *LlamaService) DownloadModel(url, filename string) error {
	return s.modelMgr.DownloadModel(url, filename, func(p ProgressReport) {
		s.mu.Lock()
		win := s.win
		s.mu.Unlock()
		if win != nil {
			_ = DownloadProgressEvent.Emit(win, p)
		}
	})
}

// ImportModelDialog opens the native file picker to select and import a .gguf file into models/.
func (s *LlamaService) ImportModelDialog() (*ModelInfo, error) {
	s.mu.Lock()
	win := s.win
	s.mu.Unlock()

	paths, err := mygo.Dialog.Open(mygo.OpenDialogOptions{
		Parent: win,
		Title:  "Import GGUF Model",
		Filters: []mygo.FileFilter{
			{Name: "GGUF Models (*.gguf)", Extensions: []string{"gguf"}},
			{Name: "All Files (*.*)", Extensions: []string{"*"}},
		},
		Multiple: false,
	})
	if err != nil {
		return nil, err
	}
	if len(paths) == 0 || paths[0] == "" {
		return nil, nil // user cancelled
	}

	info, err := s.modelMgr.ImportModel(paths[0])
	if err != nil {
		return nil, err
	}
	return &info, nil
}

// DeleteModel removes a local model file.
func (s *LlamaService) DeleteModel(filename string) error {
	return s.modelMgr.DeleteModel(filename)
}

// GetModelsDir returns the current models directory path.
func (s *LlamaService) GetModelsDir() string {
	return s.modelMgr.GetModelsDir()
}

// SetModelsDir updates the models directory path and persists in settings.
func (s *LlamaService) SetModelsDir(dir string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if err := s.modelMgr.SetModelsDir(dir); err != nil {
		return err
	}
	s.settings.ModelsDir = dir
	return s.saveSettingsToFile()
}

// SetAppDataDir updates the working app data directory and persists in settings.
func (s *LlamaService) SetAppDataDir(dir string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}
	s.appDataDir = dir
	s.settings.AppDataDir = dir
	binDir := filepath.Join(dir, "bin")
	s.releaseMgr.SetBinDir(binDir)
	s.serverMgr.SetBinDir(binDir)
	return s.saveSettingsToFile()
}

// GetServerConfig returns the active server configuration.
func (s *LlamaService) GetServerConfig() ServerConfig {
	return s.serverMgr.GetConfig()
}

// GetModelConfig retrieves the unique configuration JSON for a specific model file.
func (s *LlamaService) GetModelConfig(modelPath string) (ServerConfig, error) {
	return s.serverMgr.GetModelConfig(modelPath)
}

// SaveModelConfig saves configuration for a specific model to its individual JSON file.
func (s *LlamaService) SaveModelConfig(modelPath string, cfg ServerConfig) error {
	return s.serverMgr.SaveModelConfig(modelPath, cfg)
}

// SaveServerConfig updates the server configuration without starting it.
func (s *LlamaService) SaveServerConfig(cfg ServerConfig) error {
	if cfg.ModelPath != "" {
		return s.serverMgr.SaveModelConfig(cfg.ModelPath, cfg)
	}
	s.serverMgr.SaveConfig(cfg)
	return nil
}

// StartServer launches the server executable with the specified configuration.
func (s *LlamaService) StartServer(cfg ServerConfig) error {
	if cfg.RouterMode && cfg.ModelsDir == "" {
		cfg.ModelsDir = s.modelMgr.GetModelsDir()
	}

	err := s.serverMgr.Start(cfg)
	if err == nil {
		s.mu.Lock()
		win := s.win
		s.mu.Unlock()
		if win != nil {
			_ = ServerStatusEvent.Emit(win, s.serverMgr.GetStatus())
		}
	}
	return err
}

// StopServer stops the running server process.
func (s *LlamaService) StopServer() error {
	err := s.serverMgr.Stop()
	if err == nil {
		s.mu.Lock()
		win := s.win
		s.mu.Unlock()
		if win != nil {
			_ = ServerStatusEvent.Emit(win, s.serverMgr.GetStatus())
		}
	}
	return err
}

// GetServerStatus returns the status of the server.
func (s *LlamaService) GetServerStatus() ServerStatus {
	return s.serverMgr.GetStatus()
}

// GetRecentLogs returns the buffered lines of console output.
func (s *LlamaService) GetRecentLogs() []string {
	return s.serverMgr.GetRecentLogs()
}

// SendChat proxies a chat message list to the running server with an optional reasoning effort.
func (s *LlamaService) SendChat(messages []ChatMessage, temperature float64, reasoningEffort string) (string, error) {
	return s.serverMgr.SendChatMessage(messages, temperature, reasoningEffort)
}

// SendChatWithModel proxies chat messages to llama-server with an optional model name (for Router Mode) and reasoning effort.
func (s *LlamaService) SendChatWithModel(model string, messages []ChatMessage, temperature float64, reasoningEffort string) (string, error) {
	return s.serverMgr.SendChatMessageWithModel(model, messages, temperature, reasoningEffort)
}

// RunDiagnostics performs a comprehensive health scan of the backend and environment.
func (s *LlamaService) RunDiagnostics() TroubleshootReport {
	s.mu.Lock()
	binDir := s.releaseMgr.binDir
	modelsDir := s.modelMgr.GetModelsDir()
	variant := s.settings.ActiveBackend
	port := s.serverMgr.cfg.Port
	hw := s.hwInfo
	s.mu.Unlock()

	return RunDiagnostics(binDir, modelsDir, variant, port, hw)
}

// RepairBackend executes an automated repair action for reported issues,
// resolving the target backend for the current OS (CUDA requested on macOS
// repairs Metal instead, etc.).
func (s *LlamaService) RepairBackend(action string) error {
	switch action {
	case "kill_zombies":
		return KillZombieProcesses()
	case "download_cudart":
		s.mu.Lock()
		variant := s.settings.ActiveBackend
		s.mu.Unlock()
		if strings.TrimSpace(variant) == "" || variant == "cuda" {
			// Missing CUDA runtime libraries: reinstall the CUDA backend
			// where supported, otherwise fall back per OS.
			if strings.TrimSpace(variant) == "" {
				variant = "cuda"
			}
		}
		return s.SwitchBackend(s.repairVariantForHost(variant))
	case "reinstall":
		s.mu.Lock()
		variant := s.settings.ActiveBackend
		s.mu.Unlock()
		return s.SwitchBackend(s.repairVariantForHost(variant))
	default:
		s.mu.Lock()
		variant := s.settings.ActiveBackend
		s.mu.Unlock()
		return s.SwitchBackend(s.repairVariantForHost(variant))
	}
}

// KillZombieServers terminates any orphaned server processes.
func (s *LlamaService) KillZombieServers() error {
	return KillZombieProcesses()
}

// AutoDetectMMProj searches for an associated vision projector (mmproj) for a model.
func (s *LlamaService) AutoDetectMMProj(modelPath string) string {
	return s.modelMgr.AutoDetectMMProj(modelPath)
}


