//go:build windows

package main

import (
	"os/exec"
	"strconv"
	"syscall"
)

// prepareCmd ensures background child processes and CLI tools on Windows
// run completely hidden without flashing or opening terminal/console windows.
func prepareCmd(cmd *exec.Cmd) *exec.Cmd {
	if cmd == nil {
		return nil
	}
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.HideWindow = true
	cmd.SysProcAttr.CreationFlags |= 0x08000000 // CREATE_NO_WINDOW
	return cmd
}

// terminateProcessTree terminates a process and all its child processes cleanly on Windows.
func terminateProcessTree(pid int) {
	if pid <= 0 {
		return
	}
	killCmd := prepareCmd(exec.Command("taskkill", "/PID", strconv.Itoa(pid), "/T", "/F"))
	_ = killCmd.Run()
}

func killLingeringUnixProcesses() {
	// No-op on Windows: taskkill handles all process trees
}
