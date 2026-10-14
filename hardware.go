package main

import (
	"bytes"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
)

// HardwareInfo details system resources and detected GPU accelerator.
type HardwareInfo struct {
	OS                 string `json:"os"`
	Arch               string `json:"arch"`
	CPUModel           string `json:"cpuModel"`
	Cores              int    `json:"cores"`
	Threads            int    `json:"threads"`
	GPUName            string `json:"gpuName"`
	VRAM               string `json:"vram"`
	HasNvidia          bool   `json:"hasNvidia"`
	HasVulkan          bool   `json:"hasVulkan"`
	RecommendedBackend string `json:"recommendedBackend"`
}

func detectHardware() HardwareInfo {
	info := HardwareInfo{
		OS:                 runtime.GOOS,
		Arch:               runtime.GOARCH,
		Threads:            runtime.NumCPU(),
		Cores:              runtime.NumCPU() / 2,
		RecommendedBackend: "cpu",
	}

	if info.Cores < 1 {
		info.Cores = 1
	}

	// Detect CPU Name & Cores on Windows via PowerShell CIM
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

	// Detect GPU on Windows
	cmdGPU := prepareCmd(exec.Command("powershell", "-NoProfile", "-Command",
		"Get-CimInstance Win32_VideoController | Select-Object Name, AdapterRAM | ConvertTo-Json"))
	if out, err := cmdGPU.Output(); err == nil {
		str := string(out)
		// Check for NVIDIA GPU
		if strings.Contains(strings.ToLower(str), "nvidia") || strings.Contains(strings.ToLower(str), "geforce") || strings.Contains(strings.ToLower(str), "rtx") {
			info.HasNvidia = true
			info.HasVulkan = true
			info.RecommendedBackend = "cuda"
			info.GPUName = extractGPUName(str)
			info.VRAM = "12 GB" // Typical for RTX 3060 desktop or auto-detected
		} else if strings.Contains(strings.ToLower(str), "amd") || strings.Contains(strings.ToLower(str), "radeon") || strings.Contains(strings.ToLower(str), "intel") {
			info.HasVulkan = true
			info.RecommendedBackend = "vulkan"
			info.GPUName = extractGPUName(str)
			info.VRAM = "Shared / System"
		} else {
			info.GPUName = "Standard Graphics Adapter"
			info.RecommendedBackend = "cpu"
		}
	}

	// Double check nvidia-smi if available for precise VRAM
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
		}
	}

	return info
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
	return "NVIDIA Graphics Card"
}
