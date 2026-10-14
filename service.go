package main

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
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
	if initialBackend == "" {
		initialBackend = "cuda"
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
		releaseMgr: NewReleaseManager(binDir),
		modelMgr:   NewModelManager(modelsDir, appDataDir),
		serverMgr:  NewServerManager(binDir),
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

// SwitchBackend downloads or switches the active backend variant (cuda, vulkan, cpu).
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

// GetInstalledBackend checks if llama-server.exe is installed and which version/variant.
func (s *LlamaService) GetInstalledBackend() BackendStatus {
	return s.releaseMgr.GetInstalledBackend()
}

// CheckReleases queries GitHub for recent llama.cpp releases.
func (s *LlamaService) CheckReleases() ([]ReleaseInfo, error) {
	return s.releaseMgr.CheckReleases()
}

// DownloadBackend downloads and installs the selected release tag and variant (cuda, vulkan, cpu).
func (s *LlamaService) DownloadBackend(tag, variant string) error {
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

// SaveServerConfig updates the server configuration without starting it.
func (s *LlamaService) SaveServerConfig(cfg ServerConfig) {
	s.serverMgr.SaveConfig(cfg)
}

// StartServer launches llama-server.exe with the specified configuration.
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

// StopServer stops the running llama-server.exe process.
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

// SendChat proxies a chat message list to the running server.
func (s *LlamaService) SendChat(messages []ChatMessage, temperature float64) (string, error) {
	return s.serverMgr.SendChatMessage(messages, temperature)
}

// SendChatWithModel proxies chat messages to llama-server with an optional model name (for Router Mode).
func (s *LlamaService) SendChatWithModel(model string, messages []ChatMessage, temperature float64) (string, error) {
	return s.serverMgr.SendChatMessageWithModel(model, messages, temperature)
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

// RepairBackend executes an automated repair action for reported issues.
func (s *LlamaService) RepairBackend(action string) error {
	switch action {
	case "kill_zombies":
		return KillZombieProcesses()
	case "download_cudart":
		return s.SwitchBackend("cuda")
	case "reinstall":
		s.mu.Lock()
		variant := s.settings.ActiveBackend
		s.mu.Unlock()
		if variant == "" {
			variant = "cuda"
		}
		return s.SwitchBackend(variant)
	default:
		return s.SwitchBackend("cuda")
	}
}

// KillZombieServers terminates any orphaned llama-server.exe processes.
func (s *LlamaService) KillZombieServers() error {
	return KillZombieProcesses()
}

