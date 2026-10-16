//go:build !windows

package main

import (
	"os/exec"
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
