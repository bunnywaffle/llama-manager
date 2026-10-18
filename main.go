package main

import (
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"runtime/debug"
	"syscall"
	"time"

	"github.com/egoist/mygo"
)

func getAppDataDir() string {
	// 1. Check if portable mode is enabled via .portable file next to the executable
	if exePath, err := os.Executable(); err == nil {
		exeDir := filepath.Dir(exePath)
		if _, err := os.Stat(filepath.Join(exeDir, ".portable")); err == nil {
			return exeDir
		}
	}

	// 2. Standard user AppData / config directory (writable without elevation)
	if configDir, err := os.UserConfigDir(); err == nil && configDir != "" {
		dir := filepath.Join(configDir, "llama-manager")
		_ = os.MkdirAll(dir, 0755)
		return dir
	}

	if appData := os.Getenv("APPDATA"); appData != "" {
		dir := filepath.Join(appData, "llama-manager")
		_ = os.MkdirAll(dir, 0755)
		return dir
	}

	if home, err := os.UserHomeDir(); err == nil && home != "" {
		dir := filepath.Join(home, ".llama-manager")
		_ = os.MkdirAll(dir, 0755)
		return dir
	}

	return "."
}

func main() {
	// Minimize Go runtime memory overhead: trigger GC when heap grows by 30% (default is 100%)
	debug.SetGCPercent(30)
	debug.FreeOSMemory()
	trimProcessWorkingSet()

	appDataDir := getAppDataDir()
	svc := NewLlamaService(appDataDir)
	mygo.Bind(svc)

	// Clean up any lingering background servers and child processes on app exit
	mygo.App.OnBeforeQuit(func(e *mygo.QuitEvent) {
		_ = svc.StopServer()
		_ = KillZombieProcesses()
	})
	mygo.App.OnWindowAllClosed(func() {
		_ = svc.StopServer()
		_ = KillZombieProcesses()
		mygo.App.Quit()
	})
	mygo.App.OnQuit(func() {
		_ = svc.StopServer()
		_ = KillZombieProcesses()
	})

	// Intercept OS termination signals (SIGINT, SIGTERM) to guarantee nothing is left open
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-sigChan
		_ = svc.StopServer()
		_ = KillZombieProcesses()
		os.Exit(0)
	}()

	// Periodic background memory trim: release unused virtual memory pages back to operating system
	go func() {
		ticker := time.NewTicker(30 * time.Second)
		defer ticker.Stop()
		for range ticker.C {
			debug.FreeOSMemory()
			trimProcessWorkingSet()
		}
	}()

	mygo.App.WhenReady(func() {
		win := mygo.NewWindow(mygo.WindowOptions{
			Title:           "llama.cpp Manager",
			Width:           1140,
			Height:          740,
			MinWidth:        880,
			MinHeight:       560,
			BackgroundColor: "light-dark(#f8fafc, #090d16)",
			StateKey:        "llama-manager-main",
			URL:             "/",
		})

		svc.SetWindow(win)

		// Periodic server status ticker to keep UI metrics updated
		go func() {
			ticker := time.NewTicker(2 * time.Second)
			defer ticker.Stop()
			for range ticker.C {
				st := svc.GetServerStatus()
				if ServerStatusEvent.Emit(win, st) != nil {
					return // Window closed
				}
			}
		}()
	})

	if err := mygo.App.Run(); err != nil {
		log.Fatal(err)
	}
}
