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
}

type ServerManager struct {
	binDir         string
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

func NewServerManager(binDir string) *ServerManager {
	return &ServerManager{
		binDir:  binDir,
		maxLogs: 1000,
		cfg: ServerConfig{
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
		},
	}
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

	if sm.running && sm.cmd != nil && sm.cmd.Process != nil {
		return fmt.Errorf("llama-server is already running (PID %d)", sm.cmd.Process.Pid)
	}

	serverExe := filepath.Join(sm.binDir, getServerExeName())
	if _, err := os.Stat(serverExe); err != nil {
		return fmt.Errorf("%s not found in %s; please download the backend in Overview first", getServerExeName(), sm.binDir)
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

	if cfg.Threads > 0 {
		args = append(args, "-t", strconv.Itoa(cfg.Threads))
	}
	if cfg.FlashAttention {
		args = append(args, "-fa", "on")
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

	cmd := prepareCmd(exec.Command(serverExe, args...))
	cmd.Dir = sm.binDir

	// Prepend binDir to PATH so dynamic loaders reliably find cudart, cublas, and ggml libraries
	env := os.Environ()
	pathUpdated := false
	sep := string(os.PathListSeparator)
	for i, e := range env {
		if strings.HasPrefix(strings.ToUpper(e), "PATH=") {
			env[i] = fmt.Sprintf("PATH=%s%s%s", sm.binDir, sep, e[5:])
			pathUpdated = true
			break
		}
	}
	if !pathUpdated {
		env = append(env, fmt.Sprintf("PATH=%s", sm.binDir))
	}
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
			sm.lastError = fmt.Sprintf("Process exited: %v", waitErr)
			sm.mu.Unlock()
			sm.addLog(fmt.Sprintf("[System] llama-server process (PID %d) exited: %v", cmd.Process.Pid, waitErr))
		} else {
			sm.lastError = ""
			sm.mu.Unlock()
			sm.addLog(fmt.Sprintf("[System] llama-server process (PID %d) stopped cleanly", cmd.Process.Pid))
		}

		select {
		case exitChan <- waitErr:
		default:
		}

		sm.mu.Lock()
		cb := sm.statusCallback
		st := sm.getStatusLocked()
		sm.mu.Unlock()
		if cb != nil {
			cb(st)
		}
	}()

	sm.addLog(fmt.Sprintf("[System] Started llama-server with model %s on %s:%d", filepath.Base(cfg.ModelPath), cfg.Host, cfg.Port))

	// Release mutex temporarily while awaiting initial health confirmation
	sm.mu.Unlock()
	select {
	case err := <-exitChan:
		sm.mu.Lock()
		sm.running = false
		snippet := sm.getLastErrorSnippetLocked()
		return fmt.Errorf("llama-server failed to start (%v). Logs: %s", err, snippet)
	case <-time.After(600 * time.Millisecond):
		sm.mu.Lock()
	}

	return nil
}

func (sm *ServerManager) Stop() error {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	if !sm.running || sm.cmd == nil || sm.cmd.Process == nil {
		return nil
	}

	pid := sm.cmd.Process.Pid
	if runtime.GOOS == "windows" {
		// On Windows, use taskkill to kill process tree cleanly without flashing console
		killCmd := prepareCmd(exec.Command("taskkill", "/PID", strconv.Itoa(pid), "/T", "/F"))
		_ = killCmd.Run()
	} else {
		// On Unix (Linux / macOS), gracefully signal then terminate
		_ = sm.cmd.Process.Signal(os.Interrupt)
		time.Sleep(100 * time.Millisecond)
		_ = sm.cmd.Process.Kill()
	}

	sm.running = false
	sm.lastError = ""
	sm.addLog(fmt.Sprintf("[System] Sent stop signal to llama-server (PID %d)", pid))

	cb := sm.statusCallback
	st := sm.getStatusLocked()
	if cb != nil {
		go cb(st)
	}

	return nil
}

func (sm *ServerManager) getStatusLocked() ServerStatus {
	status := ServerStatus{
		Running:     sm.running,
		Host:        sm.cfg.Host,
		Port:        sm.cfg.Port,
		ActiveModel: filepath.Base(sm.cfg.ModelPath),
		ErrorMsg:    sm.lastError,
	}

	if sm.running && sm.cmd != nil && sm.cmd.Process != nil {
		status.PID = sm.cmd.Process.Pid
		status.UptimeSeconds = int64(time.Since(sm.startTime).Seconds())
	}

	return status
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
	for scanner.Scan() {
		line := scanner.Text()
		sm.addLog(line)
	}
}

func (sm *ServerManager) addLog(line string) {
	sm.mu.Lock()
	if len(sm.logBuffer) >= sm.maxLogs {
		sm.logBuffer = sm.logBuffer[1:]
	}
	sm.logBuffer = append(sm.logBuffer, line)
	cb := sm.logCallback
	sm.mu.Unlock()

	if cb != nil {
		cb(line)
	}
}

// ChatMessage represents a message in OpenAI compatible chat format.
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// SendChatMessage proxies chat request to running llama-server.
func (sm *ServerManager) SendChatMessage(messages []ChatMessage, temperature float64) (string, error) {
	return sm.SendChatMessageWithModel("", messages, temperature)
}

// SendChatMessageWithModel proxies chat request to running llama-server with an optional model name.
func (sm *ServerManager) SendChatMessageWithModel(model string, messages []ChatMessage, temperature float64) (string, error) {
	sm.mu.Lock()
	host := sm.cfg.Host
	port := sm.cfg.Port
	running := sm.running
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
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
		return "", err
	}

	if len(chatResp.Choices) > 0 {
		return chatResp.Choices[0].Message.Content, nil
	}

	return "", fmt.Errorf("no response choices returned")
}
