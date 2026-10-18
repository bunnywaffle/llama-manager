//go:build !windows

package main

import (
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"syscall"
	"time"
)

// prepareCmd sets process group attributes on Unix platforms (Linux & macOS)
// so child processes can be reliably terminated without leaving orphaned worker threads or GPU zombies.
func prepareCmd(cmd *exec.Cmd) *exec.Cmd {
	if cmd == nil {
		return nil
	}
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.Setpgid = true
	return cmd
}

// assignProcessToJob is a no-op on Unix platforms as process group PGID handles containment.
func assignProcessToJob(p *os.Process) {
}

// trimProcessWorkingSet is a no-op on Unix as Go runtime FreeOSMemory handles madvise.
func trimProcessWorkingSet() {
}

// terminateProcessTree terminates a process and all its child processes in its process group on Unix.
func terminateProcessTree(pid int) {
	if pid <= 0 {
		return
	}
	// Signal the process group with SIGINT first for graceful cleanup
	_ = syscall.Kill(-pid, syscall.SIGINT)
	time.Sleep(100 * time.Millisecond)
	// Force kill the process group
	_ = syscall.Kill(-pid, syscall.SIGKILL)
}

func killLingeringUnixProcesses() {
	cmd := prepareCmd(exec.Command("pkill", "-x", "llama-server"))
	_ = cmd.Run()
	cmd2 := prepareCmd(exec.Command("killall", "-9", "llama-server"))
	_ = cmd2.Run()

	// On Linux, scan /proc for any lingering llama-server PID to ensure complete cleanup
	if entries, err := os.ReadDir("/proc"); err == nil {
		for _, e := range entries {
			if !e.IsDir() {
				continue
			}
			pid, pErr := strconv.Atoi(e.Name())
			if pErr != nil || pid <= 1 {
				continue
			}
			if cmdline, err := os.ReadFile(fmt.Sprintf("/proc/%d/cmdline", pid)); err == nil {
				if strings.Contains(string(cmdline), "llama-server") {
					_ = syscall.Kill(pid, syscall.SIGKILL)
				}
			}
		}
	}
}
