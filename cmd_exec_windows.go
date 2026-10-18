//go:build windows

package main

import (
	"os"
	"os/exec"
	"strconv"
	"sync"
	"syscall"
	"unsafe"
)

var (
	kernel32                     = syscall.NewLazyDLL("kernel32.dll")
	procCreateJobObject          = kernel32.NewProc("CreateJobObjectW")
	procSetInformationJobObject  = kernel32.NewProc("SetInformationJobObject")
	procAssignProcessToJobObject = kernel32.NewProc("AssignProcessToJobObject")
	procSetProcessWorkingSetSize = kernel32.NewProc("SetProcessWorkingSetSize")
	procGetCurrentProcess        = kernel32.NewProc("GetCurrentProcess")

	globalJobObject uintptr
	jobOnce         sync.Once
)

const (
	jobObjectInfoClassExtendedLimit = 9
	jobObjectLimitKillOnJobClose    = 0x00002000
	processSetQuotaAndTerminate     = 0x0100 | 0x0001
)

type jobObjectBasicLimitInformation struct {
	PerProcessUserTimeLimit int64
	PerJobUserTimeLimit     int64
	LimitFlags              uint32
	MinimumWorkingSetSize   uintptr
	MaximumWorkingSetSize   uintptr
	ActiveProcessLimit      uint32
	Affinity                uintptr
	PriorityClass           uint32
	SchedulingClass         uint32
}

type ioCounters struct {
	ReadOperationCount  uint64
	WriteOperationCount uint64
	OtherOperationCount uint64
	ReadTransferCount   uint64
	WriteTransferCount  uint64
	OtherTransferCount  uint64
}

type jobObjectExtendedLimitInformation struct {
	BasicLimitInformation jobObjectBasicLimitInformation
	IoInfo                ioCounters
	ProcessMemoryLimit    uintptr
	JobMemoryLimit        uintptr
	PeakProcessMemoryUsed uintptr
	PeakJobMemoryUsed     uintptr
}

func initJobObject() {
	jobOnce.Do(func() {
		hJob, _, _ := procCreateJobObject.Call(0, 0)
		if hJob == 0 {
			return
		}
		var info jobObjectExtendedLimitInformation
		info.BasicLimitInformation.LimitFlags = jobObjectLimitKillOnJobClose
		r, _, _ := procSetInformationJobObject.Call(
			hJob,
			uintptr(jobObjectInfoClassExtendedLimit),
			uintptr(unsafe.Pointer(&info)),
			uintptr(unsafe.Sizeof(info)),
		)
		if r != 0 {
			globalJobObject = hJob
		}
	})
}

// assignProcessToJob registers the child process with a Windows Job Object configured
// with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE so the OS kernel guarantees child termination.
func assignProcessToJob(p *os.Process) {
	if p == nil {
		return
	}
	initJobObject()
	if globalJobObject == 0 {
		return
	}
	hProcess, err := syscall.OpenProcess(processSetQuotaAndTerminate, false, uint32(p.Pid))
	if err == nil && hProcess != 0 {
		_, _, _ = procAssignProcessToJobObject.Call(globalJobObject, uintptr(hProcess))
		_ = syscall.CloseHandle(hProcess)
	}
}

// trimProcessWorkingSet flushes unused pages from the process working set back to Windows.
func trimProcessWorkingSet() {
	hProc, _, _ := procGetCurrentProcess.Call()
	if hProc != 0 {
		_, _, _ = procSetProcessWorkingSetSize.Call(hProc, ^uintptr(0), ^uintptr(0))
	}
}

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
	// No-op on Windows: taskkill and Job Object handle all process trees
}
