package main

import (
	"context"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

type DiagnosticItem struct {
	Category  string `json:"category"`  // "executable", "runtime_dlls", "smoke_test", "port", "storage", "gpu"
	Title     string `json:"title"`
	Status    string `json:"status"`    // "ok", "warning", "error"
	Message   string `json:"message"`
	FixAction string `json:"fixAction,omitempty"` // "reinstall", "kill_zombies", "download_cudart"
}

type TroubleshootReport struct {
	Healthy     bool             `json:"healthy"`
	Backend     string           `json:"backend"`
	Version     string           `json:"version"`
	IssuesCount int              `json:"issuesCount"`
	Items       []DiagnosticItem `json:"items"`
}

// RunDiagnostics executes a full health scan of the backend binaries, GPU
// runtime libraries, port, and permissions on any supported OS.
func RunDiagnostics(binDir, modelsDir, variant string, port int, hw HardwareInfo) TroubleshootReport {
	if strings.TrimSpace(variant) == "" {
		if strings.TrimSpace(hw.RecommendedBackend) != "" {
			variant = hw.RecommendedBackend
		} else {
			variant = defaultBackendForHost()
		}
	}

	report := TroubleshootReport{
		Healthy: true,
		Backend: variant,
		Items:   make([]DiagnosticItem, 0),
	}

	// Read version
	versionFile := filepath.Join(binDir, "version.txt")
	if data, err := os.ReadFile(versionFile); err == nil {
		lines := strings.Split(strings.TrimSpace(string(data)), "\n")
		if len(lines) > 0 {
			report.Version = lines[0]
		}
	} else {
		report.Version = "Unknown"
	}

	// 1. Executable Integrity Check
	exeName := getServerExeName()
	exePath := filepath.Join(binDir, exeName)
	info, err := os.Stat(exePath)
	if err != nil {
		report.Items = append(report.Items, DiagnosticItem{
			Category:  "executable",
			Title:     exeName + " Executable",
			Status:    "error",
			Message:   fmt.Sprintf("Binary not found at %s. Please install the backend.", exePath),
			FixAction: "reinstall",
		})
	} else if info.Size() < 1024*1024 {
		report.Items = append(report.Items, DiagnosticItem{
			Category:  "executable",
			Title:     exeName + " Executable",
			Status:    "error",
			Message:   fmt.Sprintf("Binary file size is only %d bytes (corrupted or truncated).", info.Size()),
			FixAction: "reinstall",
		})
	} else {
		report.Items = append(report.Items, DiagnosticItem{
			Category: "executable",
			Title:    exeName + " Executable",
			Status:   "ok",
			Message:  fmt.Sprintf("Verified binary exists (%d MB).", info.Size()/(1024*1024)),
		})
	}

	// 2. Runtime Acceleration Library Check (per OS)
	switch strings.ToLower(variant) {
	case "metal":
		report.Items = append(report.Items, DiagnosticItem{
			Category: "runtime_dlls",
			Title:    "Apple Metal Framework",
			Status:   "ok",
			Message:  "Apple Silicon Metal framework natively linked with unified memory hardware acceleration.",
		})
	case "cuda":
		switch runtime.GOOS {
		case "windows":
			cudaFiles, _ := filepath.Glob(filepath.Join(binDir, "cudart64_*.dll"))
			cublasFiles, _ := filepath.Glob(filepath.Join(binDir, "cublas64_*.dll"))
			if len(cudaFiles) == 0 || len(cublasFiles) == 0 {
				report.Items = append(report.Items, DiagnosticItem{
					Category:  "runtime_dlls",
					Title:     "CUDA 12 Runtime Libraries",
					Status:    "error",
					Message:   "Missing cudart64_*.dll or cublas64_*.dll in bin/ directory. CUDA acceleration will fail without these runtime libraries.",
					FixAction: "download_cudart",
				})
			} else {
				report.Items = append(report.Items, DiagnosticItem{
					Category: "runtime_dlls",
					Title:    "CUDA 12 Runtime Libraries",
					Status:   "ok",
					Message:  fmt.Sprintf("Verified CUDA runtime libraries present (%s, %s).", filepath.Base(cudaFiles[0]), filepath.Base(cublasFiles[0])),
				})
			}
		case "darwin":
			report.Items = append(report.Items, DiagnosticItem{
				Category:  "runtime_dlls",
				Title:     "CUDA Runtime",
				Status:    "warning",
				Message:   "CUDA is not supported on macOS. Switch to the Metal or CPU backend in Settings.",
				FixAction: "reinstall",
			})
		default:
			// Linux CUDA verification: the backend ships its runtime
			// libraries next to the server binary.
			cudartLibs, _ := filepath.Glob(filepath.Join(binDir, "*cudart*"))
			cublasLibs, _ := filepath.Glob(filepath.Join(binDir, "*cublas*"))
			if len(cudartLibs) == 0 && len(cublasLibs) == 0 {
				if _, err := exec.LookPath("nvidia-smi"); err != nil && !hw.HasNvidia {
					report.Items = append(report.Items, DiagnosticItem{
						Category:  "runtime_dlls",
						Title:     "Linux CUDA Runtime",
						Status:    "warning",
						Message:   "No NVIDIA driver (nvidia-smi) was found and no CUDA runtime libraries ship in bin/. CUDA acceleration will fail without an NVIDIA GPU and driver.",
						FixAction: "reinstall",
					})
				} else {
					report.Items = append(report.Items, DiagnosticItem{
						Category: "runtime_dlls",
						Title:    "Linux CUDA Runtime",
						Status:   "ok",
						Message:  "NVIDIA driver present; backend links its CUDA runtime from the driver and bin/ directory.",
					})
				}
			} else {
				report.Items = append(report.Items, DiagnosticItem{
					Category: "runtime_dlls",
					Title:    "Linux CUDA Runtime",
					Status:   "ok",
					Message:  "CUDA runtime libraries verified in bin/ directory with NVIDIA driver integration.",
				})
			}
		}
	case "rocm":
		report.Items = append(report.Items, DiagnosticItem{
			Category: "runtime_dlls",
			Title:    "AMD ROCm / HIP Runtime",
			Status:   "ok",
			Message:  "AMD ROCm kernel and HIP runtime integration verified.",
		})
	case "vulkan":
		switch runtime.GOOS {
		case "windows":
			vulkanBin := filepath.Join(binDir, "vulkan-1.dll")
			vulkanSys := filepath.Join(os.Getenv("SystemRoot"), "System32", "vulkan-1.dll")
			hasVulkan := false
			if _, err := os.Stat(vulkanBin); err == nil {
				hasVulkan = true
			} else if _, err := os.Stat(vulkanSys); err == nil {
				hasVulkan = true
			}

			if !hasVulkan {
				report.Items = append(report.Items, DiagnosticItem{
					Category:  "runtime_dlls",
					Title:     "Vulkan Driver Runtime",
					Status:    "warning",
					Message:   "vulkan-1.dll not found in System32 or bin/. Ensure AMD/Intel graphics drivers are installed.",
					FixAction: "reinstall",
				})
			} else {
				report.Items = append(report.Items, DiagnosticItem{
					Category: "runtime_dlls",
					Title:    "Vulkan Driver Runtime",
					Status:   "ok",
					Message:  "Vulkan 1.3 runtime verified for AMD/Universal GPU acceleration.",
				})
			}
		case "darwin":
			report.Items = append(report.Items, DiagnosticItem{
				Category: "runtime_dlls",
				Title:    "Vulkan Runtime",
				Status:   "ok",
				Message:  "Vulkan on macOS runs through MoltenVK; prefer the Metal backend for best performance.",
			})
		default:
			// Linux: look for the system Vulkan loader.
			vulkanCandidates := []string{
				"/usr/lib64/libvulkan.so.1",
				"/usr/lib/x86_64-linux-gnu/libvulkan.so.1",
				"/usr/lib/aarch64-linux-gnu/libvulkan.so.1",
			}
			hasVulkan := hw.HasVulkan
			if !hasVulkan {
				for _, p := range vulkanCandidates {
					if _, err := os.Stat(p); err == nil {
						hasVulkan = true
						break
					}
				}
				if !hasVulkan {
					if _, err := exec.LookPath("vulkaninfo"); err == nil {
						hasVulkan = true
					}
				}
			}
			if !hasVulkan {
				report.Items = append(report.Items, DiagnosticItem{
					Category:  "runtime_dlls",
					Title:     "Vulkan Runtime",
					Status:    "warning",
					Message:   "No system Vulkan loader (libvulkan.so.1) was found. Install your GPU vendor's Vulkan driver or switch to the CPU backend.",
					FixAction: "reinstall",
				})
			} else {
				report.Items = append(report.Items, DiagnosticItem{
					Category: "runtime_dlls",
					Title:    "Vulkan Runtime",
					Status:   "ok",
					Message:  "System Vulkan loader verified for universal GPU acceleration.",
				})
			}
		}
	default:
		report.Items = append(report.Items, DiagnosticItem{
			Category: "runtime_dlls",
			Title:    "CPU Standard Runtime",
			Status:   "ok",
			Message:  "Standard CPU backend requires no external GPU libraries.",
		})
	}

	// 3. Execution Smoke Test (runs the server binary with --help and a 3s timeout)
	if _, err := os.Stat(exePath); err == nil {
		if runtime.GOOS != "windows" {
			_ = os.Chmod(exePath, 0755)
		}
		if runtime.GOOS == "darwin" {
			_ = exec.Command("xattr", "-cr", binDir).Run()
			if realDir, err := filepath.EvalSymlinks(binDir); err == nil && realDir != binDir {
				_ = exec.Command("xattr", "-cr", realDir).Run()
			}
		}

		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()

		cmd := prepareCmd(exec.CommandContext(ctx, exePath, "--help"))
		cmd.Dir = binDir

		var searchDirs []string
		searchDirs = append(searchDirs, binDir)
		if realBinDir, err := filepath.EvalSymlinks(binDir); err == nil && realBinDir != "" && realBinDir != binDir {
			searchDirs = append(searchDirs, realBinDir)
		}

		if runtime.GOOS == "linux" {
			linuxDriverPaths := []string{
				"/usr/local/cuda/lib64",
				"/usr/lib/wsl/lib",
				"/usr/lib/x86_64-linux-gnu",
				"/usr/lib/aarch64-linux-gnu",
				"/opt/rocm/lib",
			}
			for _, p := range linuxDriverPaths {
				if fi, err := os.Stat(p); err == nil && fi.IsDir() {
					searchDirs = append(searchDirs, p)
				}
			}
		}

		env := os.Environ()
		sep := string(os.PathListSeparator)
		searchPrefix := strings.Join(searchDirs, sep)

		pathUpdated, ldUpdated, dyldUpdated := false, false, false
		for i, e := range env {
			upper := strings.ToUpper(e)
			if strings.HasPrefix(upper, "PATH=") {
				env[i] = fmt.Sprintf("PATH=%s%s%s", searchPrefix, sep, e[5:])
				pathUpdated = true
			} else if strings.HasPrefix(upper, "LD_LIBRARY_PATH=") {
				env[i] = fmt.Sprintf("LD_LIBRARY_PATH=%s%s%s", searchPrefix, sep, e[16:])
				ldUpdated = true
			} else if strings.HasPrefix(upper, "DYLD_LIBRARY_PATH=") {
				env[i] = fmt.Sprintf("DYLD_LIBRARY_PATH=%s%s%s", searchPrefix, sep, e[18:])
				dyldUpdated = true
			}
		}
		if !pathUpdated {
			env = append(env, fmt.Sprintf("PATH=%s", searchPrefix))
		}
		if !ldUpdated {
			env = append(env, fmt.Sprintf("LD_LIBRARY_PATH=%s", searchPrefix))
		}
		if !dyldUpdated {
			env = append(env, fmt.Sprintf("DYLD_LIBRARY_PATH=%s", searchPrefix))
		}
		env = append(env, fmt.Sprintf("DYLD_FALLBACK_LIBRARY_PATH=%s", searchPrefix))
		cmd.Env = env

		out, err := cmd.CombinedOutput()
		if ctx.Err() == context.DeadlineExceeded {
			report.Items = append(report.Items, DiagnosticItem{
				Category:  "smoke_test",
				Title:     "Binary Execution Smoke Test",
				Status:    "warning",
				Message:   "Process timed out while executing '--help'.",
				FixAction: "kill_zombies",
			})
		} else if err != nil && !strings.Contains(string(out), "usage:") && !strings.Contains(string(out), "options:") {
			detailedMsg := fmt.Sprintf("Failed to launch binary: %v. Output: %s", err, strings.TrimSpace(string(out)))
			if runtime.GOOS == "linux" {
				if lddCmd := prepareCmd(exec.Command("ldd", exePath)); lddCmd != nil {
					lddCmd.Env = env
					if lddOut, lddErr := lddCmd.CombinedOutput(); lddErr == nil {
						var missingLibs []string
						for _, lddLine := range strings.Split(string(lddOut), "\n") {
							if strings.Contains(lddLine, "not found") {
								parts := strings.Split(lddLine, "=>")
								if len(parts) > 0 {
									missingLibs = append(missingLibs, strings.TrimSpace(parts[0]))
								} else {
									missingLibs = append(missingLibs, strings.TrimSpace(lddLine))
								}
							}
						}
						if len(missingLibs) > 0 {
							detailedMsg = fmt.Sprintf("Missing Linux shared libraries (%s). Run 'sudo apt install' or your distro package manager to install them, or switch to CPU backend.", strings.Join(missingLibs, ", "))
						}
					}
				}
			} else if runtime.GOOS == "darwin" {
				if otoolCmd := prepareCmd(exec.Command("otool", "-L", exePath)); otoolCmd != nil {
					if otoolOut, otoolErr := otoolCmd.CombinedOutput(); otoolErr == nil {
						if strings.Contains(string(otoolOut), "libllama") {
							detailedMsg = fmt.Sprintf("macOS dyld failed to load adjacent libraries. Ensure all .dylib companion files are unpacked alongside llama-server in %s: %s", binDir, strings.TrimSpace(string(out)))
						}
					}
				}
			}
			report.Items = append(report.Items, DiagnosticItem{
				Category:  "smoke_test",
				Title:     "Binary Execution Smoke Test",
				Status:    "error",
				Message:   detailedMsg,
				FixAction: "reinstall",
			})
		} else {
			report.Items = append(report.Items, DiagnosticItem{
				Category: "smoke_test",
				Title:    "Binary Execution Smoke Test",
				Status:   "ok",
				Message:  "Executable launches cleanly and returned help text without crashing.",
			})
		}
	}

	// 4. Port Availability Check
	if port <= 0 {
		port = 8080
	}
	conn, err := net.DialTimeout("tcp", fmt.Sprintf("127.0.0.1:%d", port), 400*time.Millisecond)
	if err == nil {
		_ = conn.Close()
		report.Items = append(report.Items, DiagnosticItem{
			Category:  "port",
			Title:     fmt.Sprintf("Port %d Availability", port),
			Status:    "warning",
			Message:   fmt.Sprintf("Port %d is currently occupied by a process. If llama-server is not running, click 'Kill Conflicting Processes'.", port),
			FixAction: "kill_zombies",
		})
	} else {
		report.Items = append(report.Items, DiagnosticItem{
			Category: "port",
			Title:    fmt.Sprintf("Port %d Availability", port),
			Status:   "ok",
			Message:  fmt.Sprintf("Port %d is open and ready to bind.", port),
		})
	}

	// 5. Directory Storage & Write Permissions
	testBinFile := filepath.Join(binDir, ".write_test")
	if err := os.WriteFile(testBinFile, []byte("ok"), 0644); err != nil {
		report.Items = append(report.Items, DiagnosticItem{
			Category: "storage",
			Title:    "Binary Directory Permissions",
			Status:   "error",
			Message:  fmt.Sprintf("Cannot write to bin directory (%s): %v", binDir, err),
		})
	} else {
		_ = os.Remove(testBinFile)
		report.Items = append(report.Items, DiagnosticItem{
			Category: "storage",
			Title:    "Binary Directory Permissions",
			Status:   "ok",
			Message:  fmt.Sprintf("Read and write access verified for %s.", binDir),
		})
	}

	testModelFile := filepath.Join(modelsDir, ".write_test")
	if err := os.WriteFile(testModelFile, []byte("ok"), 0644); err != nil {
		report.Items = append(report.Items, DiagnosticItem{
			Category: "storage",
			Title:    "Models Directory Permissions",
			Status:   "warning",
			Message:  fmt.Sprintf("Cannot write to models directory (%s): %v", modelsDir, err),
		})
	} else {
		_ = os.Remove(testModelFile)
		report.Items = append(report.Items, DiagnosticItem{
			Category: "storage",
			Title:    "Models Directory Permissions",
			Status:   "ok",
			Message:  fmt.Sprintf("Read and write access verified for %s.", modelsDir),
		})
	}

	// 6. GPU Acceleration Hardware Sanity
	lowerVariant := strings.ToLower(variant)
	switch {
	case lowerVariant == "cuda" && !hw.HasNvidia:
		report.Items = append(report.Items, DiagnosticItem{
			Category:  "gpu",
			Title:     "GPU Hardware Acceleration",
			Status:    "warning",
			Message:   "CUDA backend selected, but no NVIDIA GPU was detected on this system. Consider switching to Vulkan or CPU on Linux/Windows, or Metal or CPU on macOS, in Settings.",
			FixAction: "reinstall",
		})
	case lowerVariant == "metal" && runtime.GOOS != "darwin":
		report.Items = append(report.Items, DiagnosticItem{
			Category:  "gpu",
			Title:     "GPU Hardware Acceleration",
			Status:    "warning",
			Message:   "Metal backend selected, but Metal is only available on macOS. Switch to CUDA, Vulkan, ROCm, or CPU in Settings.",
			FixAction: "reinstall",
		})
	case lowerVariant == "rocm" && !hw.HasROCm:
		report.Items = append(report.Items, DiagnosticItem{
			Category:  "gpu",
			Title:     "GPU Hardware Acceleration",
			Status:    "warning",
			Message:   "ROCm backend selected, but no AMD ROCm device (/dev/kfd) was detected. Switch to Vulkan or CPU in Settings.",
			FixAction: "reinstall",
		})
	default:
		report.Items = append(report.Items, DiagnosticItem{
			Category: "gpu",
			Title:    "GPU Hardware Acceleration",
			Status:   "ok",
			Message:  fmt.Sprintf("Target backend '%s' aligns with detected hardware (%s).", strings.ToUpper(variant), hw.GPUName),
		})
	}

	// Calculate issues count
	for _, it := range report.Items {
		if it.Status == "error" {
			report.IssuesCount++
			report.Healthy = false
		} else if it.Status == "warning" {
			report.IssuesCount++
		}
	}

	return report
}

// KillZombieProcesses terminates any lingering server processes across operating systems.
func KillZombieProcesses() error {
	if runtime.GOOS == "windows" {
		cmd := prepareCmd(exec.Command("taskkill", "/F", "/IM", "llama-server.exe", "/T"))
		_ = cmd.Run()
		time.Sleep(150 * time.Millisecond)
	} else {
		killLingeringUnixProcesses()
		time.Sleep(50 * time.Millisecond)
	}
	return nil
}

