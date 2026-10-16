package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

// ServerConfig holds the launch configuration parameters for llama-server.
type ServerConfig struct {
	ModelPath      string  `json:"modelPath"`
	Host           string  `json:"host"`
	Port           int     `json:"port"`
	GPULayers      int     `json:"gpuLayers"`
	ContextSize    int     `json:"contextSize"`
	Threads        int     `json:"threads"`
	BatchSize      int     `json:"batchSize"`
	FlashAttention bool    `json:"flashAttention"`
	ExtraArgs      string  `json:"extraArgs"`

	// Advanced Sampling Parameters
	Temperature      float64 `json:"temperature"`
	TopP             float64 `json:"topP"`
	TopK             int     `json:"topK"`
	MinP             float64 `json:"minP"`
	TypicalP         float64 `json:"typicalP"`
	RepeatPenalty    float64 `json:"repeatPenalty"`
	RepeatLastN      int     `json:"repeatLastN"`
	PresencePenalty  float64 `json:"presencePenalty"`
	FrequencyPenalty float64 `json:"frequencyPenalty"`
	MaxPredict       int     `json:"maxPredict"`
	Seed             int     `json:"seed"`

	// Memory & KV Cache
	MicroBatchSize int    `json:"microBatchSize"`
	CacheTypeK     string `json:"cacheTypeK"`
	CacheTypeV     string `json:"cacheTypeV"`
	MLock          bool   `json:"mlock"`
	NoMMap         bool   `json:"noMmap"`

	// Concurrency & Server
	ParallelSlots int  `json:"parallelSlots"`
	ContBatching  bool `json:"contBatching"`
	ThreadsHTTP   int  `json:"threadsHttp"`
	ThreadsBatch  int  `json:"threadsBatch"`
	Embedding     bool `json:"embedding"`
	Metrics       bool `json:"metrics"`

	// Multi-GPU & RoPE
	MainGPU      int     `json:"mainGpu"`
	SplitMode    string  `json:"splitMode"`
	RoPEScaling  string  `json:"ropeScaling"`
	RoPEScale    float64 `json:"ropeScale"`
	RoPEFreqBase float64 `json:"ropeFreqBase"`

	// Router Mode (Multi-model dynamic serving)
	RouterMode bool   `json:"routerMode"`
	ModelsDir  string `json:"modelsDir"`

	// Vision & Multimodal Projector
	MMProjPath string `json:"mmprojPath"`

	// Speculative Decoding & Drafter Model (MTP)
	DraftModelPath string `json:"draftModelPath"`
	DraftMaxTokens int    `json:"draftMaxTokens"`
	DraftGPULayers int    `json:"draftGpuLayers"`
	DraftSpecType  string `json:"draftSpecType"`

	// Reasoning & Thinking Settings (DeepSeek-R1, QwQ, etc.)
	ReasoningFormat string `json:"reasoningFormat"`
	ReasoningEffort string `json:"reasoningEffort"`
	EnableJinja     bool   `json:"enableJinja"`
}


// ServerStatus describes the runtime state of the backend.
type ServerStatus struct {
	Running       bool    `json:"running"`
	PID           int     `json:"pid"`
	Host          string  `json:"host"`
	Port          int     `json:"port"`
	ActiveModel   string  `json:"activeModel"`
	UptimeSeconds int64   `json:"uptimeSeconds"`
	TPS           float64 `json:"tps"`
	SlotsTotal    int     `json:"slotsTotal"`
	SlotsBusy     int     `json:"slotsBusy"`
	ErrorMsg      string  `json:"errorMsg,omitempty"`
	HasVision     bool    `json:"hasVision"`
	MMProjPath    string  `json:"mmprojPath,omitempty"`
}

type ServerManager struct {
	binDir         string
	appDataDir     string
	cmd            *exec.Cmd
	cfg            ServerConfig
	running        bool
	startTime      time.Time
	mu             sync.Mutex
	logBuffer      []string
	maxLogs        int
	logCallback    func(string)
	statusCallback func(ServerStatus)
	lastError      string
}

func getDefaultServerConfig() ServerConfig {
	return ServerConfig{
		Host:             "127.0.0.1",
		Port:             8080,
		GPULayers:        99,
		ContextSize:      4096,
		Threads:          6,
		BatchSize:        512,
		FlashAttention:   true,
		Temperature:      0.7,
		TopP:             0.9,
		TopK:             40,
		MinP:             0.05,
		TypicalP:         1.0,
		RepeatPenalty:    1.1,
		RepeatLastN:      64,
		MaxPredict:       -1,
		Seed:             -1,
		MicroBatchSize:   512,
		CacheTypeK:       "f16",
		CacheTypeV:       "f16",
		ParallelSlots:    1,
		ContBatching:     true,
		RoPEScaling:      "none",
		SplitMode:        "none",
		ReasoningFormat:  "deepseek-legacy",
		ReasoningEffort:  "default",
		EnableJinja:      true,
	}
}

func NewServerManager(binDir, appDataDir string) *ServerManager {
	return &ServerManager{
		binDir:     binDir,
		appDataDir: appDataDir,
		maxLogs:    1000,
		cfg:        getDefaultServerConfig(),
	}
}

func (sm *ServerManager) getModelConfigFilePaths(modelPath string) (primaryPath, fallbackPath string) {
	if modelPath == "" {
		return "", ""
	}
	dir := filepath.Dir(modelPath)
	base := filepath.Base(modelPath)
	baseNoExt := strings.TrimSuffix(base, filepath.Ext(base))

	// 1. Primary path alongside the model file (e.g. models/model-name.json)
	primaryPath = filepath.Join(dir, baseNoExt+".json")

	// 2. Fallback path in appDataDir/model_configs/<baseNoExt>.json
	if sm.appDataDir != "" {
		fallbackDir := filepath.Join(sm.appDataDir, "model_configs")
		_ = os.MkdirAll(fallbackDir, 0755)
		fallbackPath = filepath.Join(fallbackDir, baseNoExt+".json")
	}
	return primaryPath, fallbackPath
}

// GetModelConfig retrieves the configuration JSON for a specific model.
func (sm *ServerManager) GetModelConfig(modelPath string) (ServerConfig, error) {
	defaultCfg := getDefaultServerConfig()
	defaultCfg.ModelPath = modelPath

	if modelPath == "" {
		sm.mu.Lock()
		cur := sm.cfg
		sm.mu.Unlock()
		return cur, nil
	}

	primaryPath, fallbackPath := sm.getModelConfigFilePaths(modelPath)

	// Check primary path (<baseNoExt>.json)
	if data, err := os.ReadFile(primaryPath); err == nil {
		var cfg ServerConfig
		if err := json.Unmarshal(data, &cfg); err == nil {
			if cfg.ModelPath == "" {
				cfg.ModelPath = modelPath
			}
			return cfg, nil
		}
	}

	// Check <modelPath>.json (in case of double extension e.g. model.gguf.json)
	if data, err := os.ReadFile(modelPath + ".json"); err == nil {
		var cfg ServerConfig
		if err := json.Unmarshal(data, &cfg); err == nil {
			if cfg.ModelPath == "" {
				cfg.ModelPath = modelPath
			}
			return cfg, nil
		}
	}

	// Check fallback path in AppData
	if fallbackPath != "" {
		if data, err := os.ReadFile(fallbackPath); err == nil {
			var cfg ServerConfig
			if err := json.Unmarshal(data, &cfg); err == nil {
				if cfg.ModelPath == "" {
					cfg.ModelPath = modelPath
				}
				return cfg, nil
			}
		}
	}

	return defaultCfg, nil
}

// SaveModelConfig writes the model's configuration into its individual JSON file.
func (sm *ServerManager) SaveModelConfig(modelPath string, cfg ServerConfig) error {
	if modelPath == "" {
		modelPath = cfg.ModelPath
	}
	if modelPath == "" {
		return fmt.Errorf("no model path specified")
	}
	cfg.ModelPath = modelPath

	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}

	primaryPath, fallbackPath := sm.getModelConfigFilePaths(modelPath)

	var writeErr error
	if primaryPath != "" {
		writeErr = os.WriteFile(primaryPath, data, 0644)
	}

	if fallbackPath != "" {
		_ = os.WriteFile(fallbackPath, data, 0644)
	}

	sm.mu.Lock()
	sm.cfg = cfg
	sm.mu.Unlock()

	return writeErr
}

func (sm *ServerManager) SetBinDir(dir string) {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	sm.binDir = dir
}

func (sm *ServerManager) SetLogCallback(cb func(string)) {
	sm.mu.Lock()
	sm.logCallback = cb
	sm.mu.Unlock()
}

func (sm *ServerManager) SetStatusCallback(cb func(ServerStatus)) {
	sm.mu.Lock()
	sm.statusCallback = cb
	sm.mu.Unlock()
}

func (sm *ServerManager) GetConfig() ServerConfig {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	return sm.cfg
}

func (sm *ServerManager) SaveConfig(cfg ServerConfig) {
	sm.mu.Lock()
	sm.cfg = cfg
	sm.mu.Unlock()
}

func (sm *ServerManager) Start(cfg ServerConfig) error {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	// If a managed server instance is currently running, stop it cleanly first
	if sm.running && sm.cmd != nil && sm.cmd.Process != nil {
		pid := sm.cmd.Process.Pid
		terminateProcessTree(pid)
		sm.running = false
	}

	// Always terminate any lingering / orphaned llama-server processes (zombies from previous runs or crashes)
	// to prevent GPU VRAM exhaustion (cudaMalloc failed: out of memory) and socket port binding conflicts.
	_ = KillZombieProcesses()
	time.Sleep(150 * time.Millisecond)

	serverExe := filepath.Join(sm.binDir, getServerExeName())
	if _, err := os.Stat(serverExe); err != nil {
		return fmt.Errorf("%s not found in %s; please download the backend in Overview first", getServerExeName(), sm.binDir)
	}
	if runtime.GOOS != "windows" {
		_ = os.Chmod(serverExe, 0755)
	}
	if runtime.GOOS == "darwin" {
		_ = exec.Command("xattr", "-cr", sm.binDir).Run()
	}

	if !cfg.RouterMode {
		if cfg.ModelPath == "" {
			return fmt.Errorf("no model file selected")
		}
		st, err := os.Stat(cfg.ModelPath)
		if err != nil {
			return fmt.Errorf("selected model file does not exist: %s", cfg.ModelPath)
		}
		if st.Size() < 1024*1024 {
			return fmt.Errorf("selected model file is invalid or empty (%d bytes): %s", st.Size(), cfg.ModelPath)
		}
	} else {
		if cfg.ModelsDir == "" {
			return fmt.Errorf("models directory is not configured for Router Mode")
		}
	}

	sm.cfg = cfg
	sm.lastError = ""

	// Automatically save this configuration for the model to its individual JSON file
	if !cfg.RouterMode && cfg.ModelPath != "" {
		_ = sm.SaveModelConfig(cfg.ModelPath, cfg)
	}

	// Build arguments
	args := []string{
		"--host", cfg.Host,
		"--port", strconv.Itoa(cfg.Port),
		"-c", strconv.Itoa(cfg.ContextSize),
		"-ngl", strconv.Itoa(cfg.GPULayers),
		"-b", strconv.Itoa(cfg.BatchSize),
	}

	if cfg.RouterMode {
		args = append(args, "--models-path", cfg.ModelsDir)
		if cfg.ModelPath != "" {
			args = append(args, "-m", cfg.ModelPath)
		}
	} else {
		args = append(args, "-m", cfg.ModelPath)
	}

	// Vision Projector / Multimodal Encoder (--mmproj)
	if strings.TrimSpace(cfg.MMProjPath) != "" {
		if _, err := os.Stat(cfg.MMProjPath); err == nil {
			args = append(args, "--mmproj", cfg.MMProjPath)
		}
	}

	// Speculative Decoding & MTP Draft Model (-md)
	if strings.TrimSpace(cfg.DraftModelPath) != "" {
		if _, err := os.Stat(cfg.DraftModelPath); err == nil {
			args = append(args, "-md", cfg.DraftModelPath)
			if cfg.DraftMaxTokens > 0 {
				args = append(args, "--spec-draft-n-max", strconv.Itoa(cfg.DraftMaxTokens))
			}
			if cfg.DraftGPULayers > 0 {
				args = append(args, "--spec-draft-ngl", strconv.Itoa(cfg.DraftGPULayers))
			}
			if cfg.DraftSpecType != "" && cfg.DraftSpecType != "none" {
				args = append(args, "--spec-type", cfg.DraftSpecType)
			}
		}
	}

	if cfg.Threads > 0 {
		args = append(args, "-t", strconv.Itoa(cfg.Threads))
	}
	if cfg.FlashAttention {
		args = append(args, "-fa", "auto")
	} else {
		args = append(args, "-fa", "off")
	}

	// Sampling arguments
	if cfg.Temperature > 0 {
		args = append(args, "--temp", strconv.FormatFloat(cfg.Temperature, 'f', 2, 64))
	}
	if cfg.TopP > 0 && cfg.TopP <= 1.0 {
		args = append(args, "--top-p", strconv.FormatFloat(cfg.TopP, 'f', 2, 64))
	}
	if cfg.TopK > 0 {
		args = append(args, "--top-k", strconv.Itoa(cfg.TopK))
	}
	if cfg.MinP > 0 {
		args = append(args, "--min-p", strconv.FormatFloat(cfg.MinP, 'f', 2, 64))
	}
	if cfg.TypicalP > 0 && cfg.TypicalP != 1.0 {
		args = append(args, "--typical", strconv.FormatFloat(cfg.TypicalP, 'f', 2, 64))
	}
	if cfg.RepeatPenalty > 0 {
		args = append(args, "--repeat-penalty", strconv.FormatFloat(cfg.RepeatPenalty, 'f', 2, 64))
	}
	if cfg.RepeatLastN > 0 {
		args = append(args, "--repeat-last-n", strconv.Itoa(cfg.RepeatLastN))
	}
	if cfg.PresencePenalty != 0 {
		args = append(args, "--presence-penalty", strconv.FormatFloat(cfg.PresencePenalty, 'f', 2, 64))
	}
	if cfg.FrequencyPenalty != 0 {
		args = append(args, "--frequency-penalty", strconv.FormatFloat(cfg.FrequencyPenalty, 'f', 2, 64))
	}
	if cfg.MaxPredict > 0 {
		args = append(args, "-n", strconv.Itoa(cfg.MaxPredict))
	}
	if cfg.Seed >= 0 {
		args = append(args, "--seed", strconv.Itoa(cfg.Seed))
	}

	// Memory & KV Cache arguments
	if cfg.MicroBatchSize > 0 {
		args = append(args, "-ub", strconv.Itoa(cfg.MicroBatchSize))
	}
	if cfg.CacheTypeK != "" && cfg.CacheTypeK != "f16" {
		args = append(args, "--cache-type-k", cfg.CacheTypeK)
	}
	if cfg.CacheTypeV != "" && cfg.CacheTypeV != "f16" {
		args = append(args, "--cache-type-v", cfg.CacheTypeV)
	}
	if cfg.MLock {
		args = append(args, "--mlock")
	}
	if cfg.NoMMap {
		args = append(args, "--no-mmap")
	}

	// Concurrency & Server
	if cfg.ParallelSlots > 1 {
		args = append(args, "-np", strconv.Itoa(cfg.ParallelSlots))
	}
	if cfg.ContBatching {
		args = append(args, "-cb")
	}
	if cfg.ThreadsHTTP > 0 {
		args = append(args, "--threads-http", strconv.Itoa(cfg.ThreadsHTTP))
	}
	if cfg.ThreadsBatch > 0 {
		args = append(args, "-tb", strconv.Itoa(cfg.ThreadsBatch))
	}
	if cfg.Embedding {
		args = append(args, "--embedding")
	}
	if cfg.Metrics {
		args = append(args, "--metrics")
	}

	// Multi-GPU & RoPE
	if cfg.MainGPU > 0 {
		args = append(args, "--main-gpu", strconv.Itoa(cfg.MainGPU))
	}
	if cfg.SplitMode != "" && cfg.SplitMode != "none" {
		args = append(args, "--split-mode", cfg.SplitMode)
	}
	if cfg.RoPEScaling != "" && cfg.RoPEScaling != "none" {
		args = append(args, "--rope-scaling", cfg.RoPEScaling)
	}
	if cfg.RoPEScale > 0 {
		args = append(args, "--rope-scale", strconv.FormatFloat(cfg.RoPEScale, 'f', 4, 64))
	}
	if cfg.RoPEFreqBase > 0 {
		args = append(args, "--rope-freq-base", strconv.FormatFloat(cfg.RoPEFreqBase, 'f', 2, 64))
	}

	if strings.TrimSpace(cfg.ExtraArgs) != "" {
		extraParts := strings.Fields(cfg.ExtraArgs)
		args = append(args, extraParts...)
	}

	// Reasoning & Thinking model arguments (DeepSeek-R1, QwQ, etc.)
	if cfg.EnableJinja {
		args = append(args, "--jinja")
	}
	if cfg.ReasoningFormat != "" && cfg.ReasoningFormat != "none" {
		args = append(args, "--reasoning-format", cfg.ReasoningFormat)
	}
	if cfg.ReasoningEffort != "" && cfg.ReasoningEffort != "default" && cfg.ReasoningEffort != "auto" {
		args = append(args, "--reasoning-effort", cfg.ReasoningEffort)
	}

	cmd := prepareCmd(exec.Command(serverExe, args...))
	cmd.Dir = sm.binDir

	// Prepend binDir to PATH, LD_LIBRARY_PATH (Linux), and DYLD_LIBRARY_PATH (macOS)
	env := os.Environ()
	sep := string(os.PathListSeparator)

	pathUpdated := false
	ldUpdated := false
	dyldUpdated := false

	for i, e := range env {
		upper := strings.ToUpper(e)
		if strings.HasPrefix(upper, "PATH=") {
			env[i] = fmt.Sprintf("PATH=%s%s%s", sm.binDir, sep, e[5:])
			pathUpdated = true
		} else if strings.HasPrefix(upper, "LD_LIBRARY_PATH=") {
			env[i] = fmt.Sprintf("LD_LIBRARY_PATH=%s%s%s", sm.binDir, sep, e[16:])
			ldUpdated = true
		} else if strings.HasPrefix(upper, "DYLD_LIBRARY_PATH=") {
			env[i] = fmt.Sprintf("DYLD_LIBRARY_PATH=%s%s%s", sm.binDir, sep, e[18:])
			dyldUpdated = true
		}
	}
	if !pathUpdated {
		env = append(env, fmt.Sprintf("PATH=%s", sm.binDir))
	}
	if !ldUpdated {
		env = append(env, fmt.Sprintf("LD_LIBRARY_PATH=%s", sm.binDir))
	}
	if !dyldUpdated {
		env = append(env, fmt.Sprintf("DYLD_LIBRARY_PATH=%s", sm.binDir))
	}
	env = append(env, fmt.Sprintf("DYLD_FALLBACK_LIBRARY_PATH=%s", sm.binDir))
	cmd.Env = env

	stdoutPipe, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	stderrPipe, err := cmd.StderrPipe()
	if err != nil {
		return err
	}

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start llama-server: %w", err)
	}

	sm.cmd = cmd
	sm.running = true
	sm.startTime = time.Now()

	// Read logs from stdout and stderr
	go sm.streamLogs(stdoutPipe)
	go sm.streamLogs(stderrPipe)

	exitChan := make(chan error, 1)

	// Wait in background to track exit
	go func() {
		waitErr := cmd.Wait()
		sm.mu.Lock()
		sm.running = false
		if waitErr != nil {
			errReason := sm.getLastMeaningfulErrorLocked()
			if errReason != "" {
				sm.lastError = fmt.Sprintf("Process exited: %v (%s)", waitErr, errReason)
			} else {
				sm.lastError = fmt.Sprintf("Process exited: %v", waitErr)
			}
			sm.addLogLocked(fmt.Sprintf("[System] llama-server process (PID %d) exited: %v", cmd.Process.Pid, waitErr))
		} else {
			sm.lastError = ""
			sm.addLogLocked(fmt.Sprintf("[System] llama-server process (PID %d) stopped cleanly", cmd.Process.Pid))
		}

		cb := sm.statusCallback
		st := sm.getStatusLocked()
		sm.mu.Unlock()

		select {
		case exitChan <- waitErr:
		default:
		}

		if cb != nil {
			cb(st)
		}
	}()

	sm.addLogLocked(fmt.Sprintf("[System] Started llama-server with model %s on %s:%d", filepath.Base(cfg.ModelPath), cfg.Host, cfg.Port))

	// Release mutex temporarily while awaiting initial health confirmation
	sm.mu.Unlock()
	select {
	case err := <-exitChan:
		sm.mu.Lock()
		sm.running = false
		snippet := sm.getLastMeaningfulErrorLocked()
		if snippet == "" {
			snippet = sm.getLastErrorSnippetLocked()
		}
		return fmt.Errorf("llama-server failed to start (%v). %s", err, snippet)
	case <-time.After(1200 * time.Millisecond):
		sm.mu.Lock()
		if !sm.running {
			snippet := sm.getLastMeaningfulErrorLocked()
			if snippet == "" {
				snippet = sm.lastError
			}
			return fmt.Errorf("llama-server failed during startup: %s", snippet)
		}
	}

	return nil
}

func (sm *ServerManager) Stop() error {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	if !sm.running || sm.cmd == nil || sm.cmd.Process == nil {
		_ = KillZombieProcesses()
		return nil
	}

	pid := sm.cmd.Process.Pid
	terminateProcessTree(pid)

	sm.running = false
	sm.lastError = ""
	sm.addLogLocked(fmt.Sprintf("[System] Sent stop signal to llama-server (PID %d)", pid))

	// Clean up any lingering zombie child processes
	_ = KillZombieProcesses()

	cb := sm.statusCallback
	st := sm.getStatusLocked()
	if cb != nil {
		go cb(st)
	}

	return nil
}

func (sm *ServerManager) getStatusLocked() ServerStatus {
	hasVision := strings.TrimSpace(sm.cfg.MMProjPath) != ""
	if hasVision {
		if _, err := os.Stat(sm.cfg.MMProjPath); err != nil {
			hasVision = false
		}
	}

	status := ServerStatus{
		Running:     sm.running,
		Host:        sm.cfg.Host,
		Port:        sm.cfg.Port,
		ActiveModel: filepath.Base(sm.cfg.ModelPath),
		ErrorMsg:    sm.lastError,
		HasVision:   hasVision,
		MMProjPath:  sm.cfg.MMProjPath,
	}

	if sm.running && sm.cmd != nil && sm.cmd.Process != nil {
		status.PID = sm.cmd.Process.Pid
		status.UptimeSeconds = int64(time.Since(sm.startTime).Seconds())
	}

	return status
}

func (sm *ServerManager) getLastMeaningfulErrorLocked() string {
	if len(sm.logBuffer) == 0 {
		return ""
	}
	// Search backwards for the most informative error line
	for i := len(sm.logBuffer) - 1; i >= 0; i-- {
		line := strings.TrimSpace(sm.logBuffer[i])
		if line == "" || strings.HasPrefix(line, "[System]") {
			continue
		}
		lower := strings.ToLower(line)
		if strings.Contains(lower, "error") ||
			strings.Contains(lower, "failed") ||
			strings.Contains(lower, "cudamalloc") ||
			strings.Contains(lower, "out of memory") ||
			strings.Contains(lower, "cannot open") ||
			strings.Contains(lower, "exception") ||
			strings.Contains(lower, "invalid") ||
			strings.Contains(lower, "dyld") ||
			strings.Contains(lower, "library not loaded") ||
			strings.Contains(lower, "shared libraries") ||
			strings.Contains(lower, "permission denied") ||
			strings.Contains(lower, "segmentation fault") ||
			strings.Contains(lower, "illegal instruction") ||
			strings.Contains(lower, "abort") ||
			strings.Contains(lower, "killed") {
			if len(line) > 160 {
				line = line[:157] + "..."
			}
			return line
		}
	}

	// Fallback to the last non-system log line
	for i := len(sm.logBuffer) - 1; i >= 0; i-- {
		line := strings.TrimSpace(sm.logBuffer[i])
		if line != "" && !strings.HasPrefix(line, "[System]") {
			if len(line) > 160 {
				line = line[:157] + "..."
			}
			return line
		}
	}
	return ""
}

func (sm *ServerManager) getLastErrorSnippetLocked() string {
	if len(sm.logBuffer) == 0 {
		return ""
	}
	start := 0
	if len(sm.logBuffer) > 4 {
		start = len(sm.logBuffer) - 4
	}
	return strings.Join(sm.logBuffer[start:], " | ")
}

func (sm *ServerManager) GetStatus() ServerStatus {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	status := sm.getStatusLocked()
	if sm.running && sm.cmd != nil && sm.cmd.Process != nil {
		go sm.queryServerMetrics()
	}

	return status
}

func (sm *ServerManager) queryServerMetrics() {
	url := fmt.Sprintf("http://%s:%d/health", sm.cfg.Host, sm.cfg.Port)
	client := http.Client{Timeout: 500 * time.Millisecond}
	resp, err := client.Get(url)
	if err == nil {
		resp.Body.Close()
	}
}

func (sm *ServerManager) GetRecentLogs() []string {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	res := make([]string, len(sm.logBuffer))
	copy(res, sm.logBuffer)
	return res
}

func (sm *ServerManager) streamLogs(reader io.Reader) {
	scanner := bufio.NewScanner(reader)
	buf := make([]byte, 64*1024)
	scanner.Buffer(buf, 1024*1024)
	for scanner.Scan() {
		line := scanner.Text()
		sm.addLog(line)
	}
	if err := scanner.Err(); err != nil && err != io.EOF {
		sm.addLog(fmt.Sprintf("[System] Log scanner notice: %v", err))
	}
}

func (sm *ServerManager) addLogLocked(line string) {
	if len(sm.logBuffer) >= sm.maxLogs {
		sm.logBuffer = sm.logBuffer[1:]
	}
	sm.logBuffer = append(sm.logBuffer, line)
	cb := sm.logCallback
	if cb != nil {
		go cb(line)
	}
}

func (sm *ServerManager) addLog(line string) {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	sm.addLogLocked(line)
}

// ChatMessage represents a message in OpenAI compatible chat format.
type ChatMessage struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"`
}

// SendChatMessage proxies chat request to running llama-server with an optional reasoning effort.
func (sm *ServerManager) SendChatMessage(messages []ChatMessage, temperature float64, reasoningEffort string) (string, error) {
	return sm.SendChatMessageWithModel("", messages, temperature, reasoningEffort)
}

// SendChatMessageWithModel proxies chat request to running llama-server with an optional model name and reasoning effort.
func (sm *ServerManager) SendChatMessageWithModel(model string, messages []ChatMessage, temperature float64, reasoningEffort string) (string, error) {
	sm.mu.Lock()
	host := sm.cfg.Host
	port := sm.cfg.Port
	running := sm.running
	defaultEffort := sm.cfg.ReasoningEffort
	sm.mu.Unlock()

	if !running {
		return "", fmt.Errorf("server is not running")
	}

	reqBody := map[string]interface{}{
		"messages":    messages,
		"temperature": temperature,
		"stream":      false,
	}
	if model != "" {
		reqBody["model"] = model
	}

	effort := strings.TrimSpace(reasoningEffort)
	if effort == "" {
		effort = strings.TrimSpace(defaultEffort)
	}

	if effort != "" && effort != "default" && effort != "auto" {
		if effort == "off" || effort == "none" {
			reqBody["chat_template_kwargs"] = map[string]interface{}{"enable_thinking": false}
		} else {
			reqBody["reasoning_effort"] = effort
			reqBody["chat_template_kwargs"] = map[string]interface{}{"reasoning_effort": effort}
		}
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return "", err
	}

	url := fmt.Sprintf("http://%s:%d/v1/chat/completions", host, port)
	client := http.Client{Timeout: 180 * time.Second}
	resp, err := client.Post(url, "application/json", strings.NewReader(string(jsonData)))
	if err != nil {
		return "", fmt.Errorf("failed to call llama-server: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("llama-server returned HTTP %d: %s", resp.StatusCode, string(b))
	}

	var chatResp struct {
		Choices []struct {
			Message struct {
				Content          string `json:"content"`
				ReasoningContent string `json:"reasoning_content"`
				Reasoning        string `json:"reasoning"`
			} `json:"message"`
		} `json:"choices"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
		return "", err
	}

	if len(chatResp.Choices) > 0 {
		msg := chatResp.Choices[0].Message
		res := msg.Content
		reasoning := strings.TrimSpace(msg.ReasoningContent)
		if reasoning == "" {
			reasoning = strings.TrimSpace(msg.Reasoning)
		}

		if reasoning != "" {
			if !strings.Contains(res, "<think>") && !strings.Contains(res, "<thought>") {
				if res == "" {
					res = fmt.Sprintf("<think>\n%s\n</think>", reasoning)
				} else {
					res = fmt.Sprintf("<think>\n%s\n</think>\n%s", reasoning, res)
				}
			}
		}
		return res, nil
	}

	return "", fmt.Errorf("no response choices returned")
}
