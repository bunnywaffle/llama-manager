package main

import (
	"log"
	"os"
	"path/filepath"
	"time"

	"github.com/egoist/mygo"
)

func main() {
	// Determine working directory for persistent app data (binaries, models, configs)
	exePath, err := os.Executable()
	appDataDir := "."
	if err == nil {
		appDataDir = filepath.Dir(exePath)
	}

	svc := NewLlamaService(appDataDir)
	mygo.Bind(svc)

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
