//go:build !windows

package main

import "os/exec"

func prepareCmd(cmd *exec.Cmd) *exec.Cmd {
	return cmd
}
