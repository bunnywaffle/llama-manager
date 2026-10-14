package main

import (
	"bytes"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
)

// HardwareInfo details system resources, detected GPU accelerators, and available backends across OSes.
type HardwareInfo struct {
	OS                 string   `json:"os"`                 // "windows", "darwin", "linux"
	Arch               string   `json:"arch"`               // "x64", "arm64"
	PlatformDisplay    string   `json:"platformDisplay"`    // e.g. "Windows 11 (x64)", "macOS (Apple Silicon)"
	CPUModel           string   `json:"cpuModel"`
	Cores              int      `json:"cores"`
	Threads            int      `json:"threads"`
	GPUName            string   `json:"gpuName"`
	VRAM               string   `json:"vram"`
	HasNvidia          bool     `json:"hasNvidia"`
	HasVulkan          bool     `json:"hasVulkan"`
	HasMetal           bool     `json:"hasMetal"`
	HasROCm            bool     `json:"hasRocm"`
	RecommendedBackend string   `json:"recommendedBackend"` // "metal", "cuda", "vulkan", "rocm", "cpu"
	AvailableBackends  []string `json:"availableBackends"`
}

var cachedHardware *HardwareInfo

func detectHardware() HardwareInfo {
	if cachedHardware != nil {
		return *cachedHardware
	}

	arch := runtime.GOARCH
	if arch == "amd64" {
		arch = "x64"
	}

	info := HardwareInfo{
		OS:                 runtime.GOOS,
		Arch:               arch,
		Threads:            runtime.NumCPU(),
		Cores:              runtime.NumCPU() / 2,
		RecommendedBackend: "cpu",
		AvailableBackends:  []string{"cpu"},
	}

	if info.Cores < 1 {
		info.Cores = 1
	}

	switch runtime.GOOS {
	case "darwin":
		detectHardwareDarwin(&info)
	case "linux":
		detectHardwareLinux(&info)
	case "windows":
		detectHardwareWindows(&info)
	default:
		info.PlatformDisplay = runtime.GOOS + " (" + arch + ")"
		info.CPUModel = "Standard Processor"
		info.GPUName = "Default Display Adapter"
	}

	cachedHardware = &info
	return info
}

// detectHardwareDarwin inspects macOS / Apple Silicon / Intel Mac hardware.
func detectHardwareDarwin(info *HardwareInfo) {
	if info.Arch == "arm64" {
		info.PlatformDisplay = "macOS (Apple Silicon)"
	} else {
		info.PlatformDisplay = "macOS (Intel x64)"
	}

	// 1. CPU Model
	cmdCPU := prepareCmd(exec.Command("sysctl", "-n", "machdep.cpu.brand_string"))
	if out, err := cmdCPU.Output(); err == nil && len(out) > 0 {
		info.CPUModel = strings.TrimSpace(string(out))
	} else {
		cmdModel := prepareCmd(exec.Command("sysctl", "-n", "hw.model"))
		if outModel, err := cmdModel.Output(); err == nil && len(outModel) > 0 {
			info.CPUModel = strings.TrimSpace(string(outModel))
		}
	}

	// 2. Physical Cores
	cmdCores := prepareCmd(exec.Command("sysctl", "-n", "hw.physicalcpu"))
	if out, err := cmdCores.Output(); err == nil {
		if c, err := strconv.Atoi(strings.TrimSpace(string(out))); err == nil && c > 0 {
			info.Cores = c
		}
	}

	// 3. Total Memory (Unified on Apple Silicon)
	cmdMem := prepareCmd(exec.Command("sysctl", "-n", "hw.memsize"))
	if out, err := cmdMem.Output(); err == nil {
		if bytesVal, err := strconv.ParseInt(strings.TrimSpace(string(out)), 10, 64); err == nil {
			info.VRAM = strconv.FormatInt(bytesVal/(1024*1024*1024), 10) + " GB (Unified)"
		}
	}

	// 4. Apple Silicon Metal or AMD/Intel Mac GPU
	if info.Arch == "arm64" {
		info.HasMetal = true
		info.GPUName = "Apple Silicon Metal GPU"
		info.RecommendedBackend = "metal"
		info.AvailableBackends = []string{"metal", "cpu"}
	} else {
		// Intel Mac: check for discrete GPU
		cmdGPU := prepareCmd(exec.Command("system_profiler", "SPDisplaysDataType"))
		if out, err := cmdGPU.Output(); err == nil {
			str := string(out)
			if strings.Contains(strings.ToLower(str), "metal") || strings.Contains(strings.ToLower(str), "amd") {
				info.HasMetal = true
				info.GPUName = "Mac Metal Accelerator"
				info.RecommendedBackend = "metal"
				info.AvailableBackends = []string{"metal", "cpu"}
			}
		}
	}
}

// detectHardwareLinux inspects Linux distributions, CPU info, and GPU drivers (NVIDIA, ROCm, Vulkan).
func detectHardwareLinux(info *HardwareInfo) {
	info.PlatformDisplay = "Linux (" + info.Arch + ")"

	// 1. CPU info from /proc/cpuinfo
	if data, err := os.ReadFile("/proc/cpuinfo"); err == nil {
		lines := strings.Split(string(data), "\n")
		coreCount := 0
		for _, line := range lines {
			if strings.HasPrefix(line, "model name") {
				parts := strings.Split(line, ":")
				if len(parts) >= 2 && info.CPUModel == "" {
					info.CPUModel = strings.TrimSpace(parts[1])
				}
			}
			if strings.HasPrefix(line, "processor") {
				coreCount++
			}
		}
		if coreCount > 0 {
			info.Threads = coreCount
			info.Cores = coreCount / 2
			if info.Cores < 1 {
				info.Cores = 1
			}
		}
	}

	// 2. Check for NVIDIA GPU
	cmdSmi := prepareCmd(exec.Command("nvidia-smi", "--query-gpu=gpu_name,memory.total", "--format=csv,noheader,nounits"))
	var smiOut bytes.Buffer
	cmdSmi.Stdout = &smiOut
	if err := cmdSmi.Run(); err == nil {
		parts := strings.Split(strings.TrimSpace(smiOut.String()), ",")
		if len(parts) >= 2 {
			info.GPUName = strings.TrimSpace(parts[0])
			if mb, err := strconv.Atoi(strings.TrimSpace(parts[1])); err == nil {
				info.VRAM = strconv.Itoa(mb/1024) + " GB"
			}
			info.HasNvidia = true
			info.RecommendedBackend = "cuda"
			info.AvailableBackends = append(info.AvailableBackends, "cuda")
		}
	}

	// 3. Check for AMD ROCm
	if _, err := os.Stat("/dev/kfd"); err == nil {
		info.HasROCm = true
		if !info.HasNvidia {
			info.GPUName = "AMD Radeon (ROCm)"
			info.RecommendedBackend = "rocm"
		}
		info.AvailableBackends = append(info.AvailableBackends, "rocm")
	}

	// 4. Check for Vulkan
	if _, err := os.Stat("/dev/dri"); err == nil {
		info.HasVulkan = true
		info.AvailableBackends = append(info.AvailableBackends, "vulkan")
		if info.RecommendedBackend == "cpu" {
			info.RecommendedBackend = "vulkan"
		}
	}
}

// detectHardwareWindows inspects Windows CPU, NVIDIA GPUs, and AMD/Intel controllers.
func detectHardwareWindows(info *HardwareInfo) {
	info.PlatformDisplay = "Windows (" + info.Arch + ")"

	// 1. Detect CPU Name & Cores via PowerShell CIM
	cmdCPU := prepareCmd(exec.Command("powershell", "-NoProfile", "-Command",
		"Get-CimInstance Win32_Processor | Select-Object -First 1 Name, NumberOfCores, NumberOfLogicalProcessors | ConvertTo-Json"))
	if out, err := cmdCPU.Output(); err == nil {
		str := string(out)
		if strings.Contains(str, `"Name":`) {
			info.CPUModel = extractJsonString(str, "Name")
		}
		if coresStr := extractJsonValue(str, "NumberOfCores"); coresStr != "" {
			if c, err := strconv.Atoi(coresStr); err == nil && c > 0 {
				info.Cores = c
			}
		}
		if threadsStr := extractJsonValue(str, "NumberOfLogicalProcessors"); threadsStr != "" {
			if t, err := strconv.Atoi(threadsStr); err == nil && t > 0 {
				info.Threads = t
			}
		}
	}

	// 2. Detect GPU via PowerShell CIM
	cmdGPU := prepareCmd(exec.Command("powershell", "-NoProfile", "-Command",
		"Get-CimInstance Win32_VideoController | Select-Object Name, AdapterRAM | ConvertTo-Json"))
	if out, err := cmdGPU.Output(); err == nil {
		str := string(out)
		lower := strings.ToLower(str)
		if strings.Contains(lower, "nvidia") || strings.Contains(lower, "geforce") || strings.Contains(lower, "rtx") {
			info.HasNvidia = true
			info.HasVulkan = true
			info.RecommendedBackend = "cuda"
			info.GPUName = extractGPUName(str)
			info.VRAM = "12 GB"
			info.AvailableBackends = []string{"cuda", "vulkan", "cpu"}
		} else if strings.Contains(lower, "amd") || strings.Contains(lower, "radeon") || strings.Contains(lower, "intel") {
			info.HasVulkan = true
			info.RecommendedBackend = "vulkan"
			info.GPUName = extractGPUName(str)
			info.VRAM = "Shared / System"
			info.AvailableBackends = []string{"vulkan", "cpu"}
		} else {
			info.GPUName = "Standard Graphics Adapter"
			info.RecommendedBackend = "cpu"
			info.AvailableBackends = []string{"cpu"}
		}
	}

	// 3. Double check nvidia-smi if available for precise VRAM
	cmdSmi := prepareCmd(exec.Command("nvidia-smi", "--query-gpu=gpu_name,memory.total", "--format=csv,noheader,nounits"))
	var smiOut bytes.Buffer
	cmdSmi.Stdout = &smiOut
	if err := cmdSmi.Run(); err == nil {
		parts := strings.Split(strings.TrimSpace(smiOut.String()), ",")
		if len(parts) >= 2 {
			info.GPUName = strings.TrimSpace(parts[0])
			if mb, err := strconv.Atoi(strings.TrimSpace(parts[1])); err == nil {
				info.VRAM = strconv.Itoa(mb/1024) + " GB"
			}
			info.HasNvidia = true
			info.RecommendedBackend = "cuda"
			info.AvailableBackends = []string{"cuda", "vulkan", "cpu"}
		}
	}
}

func extractJsonString(jsonStr, key string) string {
	idx := strings.Index(jsonStr, `"`+key+`":`)
	if idx == -1 {
		return ""
	}
	rem := jsonStr[idx+len(key)+3:]
	q1 := strings.Index(rem, `"`)
	if q1 == -1 {
		return ""
	}
	rem2 := rem[q1+1:]
	q2 := strings.Index(rem2, `"`)
	if q2 == -1 {
		return ""
	}
	return strings.TrimSpace(rem2[:q2])
}

func extractJsonValue(jsonStr, key string) string {
	idx := strings.Index(jsonStr, `"`+key+`":`)
	if idx == -1 {
		return ""
	}
	rem := strings.TrimSpace(jsonStr[idx+len(key)+3:])
	end := strings.IndexAny(rem, ",\r\n}")
	if end == -1 {
		return rem
	}
	return strings.TrimSpace(rem[:end])
}

func extractGPUName(jsonStr string) string {
	lines := strings.Split(jsonStr, "\n")
	for _, l := range lines {
		if strings.Contains(l, `"Name":`) {
			parts := strings.Split(l, ":")
			if len(parts) >= 2 {
				name := strings.Trim(parts[1], " \",\r\t")
				if !strings.Contains(strings.ToLower(name), "virtual") {
					return name
				}
			}
		}
	}
	return "Graphics Controller"
}
