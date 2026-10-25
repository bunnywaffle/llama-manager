package main

import (
	"archive/tar"
	"compress/gzip"
	"os"
	"path/filepath"
	"testing"
)

func TestModelConfigPersistence(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "llama_cfg_test_*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	appData := filepath.Join(tempDir, "appdata")
	binDir := filepath.Join(tempDir, "bin")
	_ = os.MkdirAll(appData, 0755)
	_ = os.MkdirAll(binDir, 0755)

	sm := NewServerManager(binDir, appData)

	fakeModel := filepath.Join(tempDir, "MyTestModel-Q4_K_M.gguf")
	if err := os.WriteFile(fakeModel, []byte("fake gguf"), 0644); err != nil {
		t.Fatalf("Failed to write fake model: %v", err)
	}

	// 1. GetModelConfig before any JSON exists -> returns default with modelPath set
	cfg, err := sm.GetModelConfig(fakeModel)
	if err != nil {
		t.Fatalf("GetModelConfig failed: %v", err)
	}
	if cfg.ModelPath != fakeModel {
		t.Errorf("Expected ModelPath %s, got %s", fakeModel, cfg.ModelPath)
	}

	// 2. Modify specific model parameters and save
	cfg.Temperature = 0.42
	cfg.MinP = 0.15
	cfg.TopP = 0.85
	cfg.TopK = 50
	cfg.BatchSize = 1024
	cfg.MicroBatchSize = 256
	cfg.ReasoningFormat = "deepseek-legacy"
	cfg.ReasoningEffort = "high"

	if err := sm.SaveModelConfig(fakeModel, cfg); err != nil {
		t.Fatalf("SaveModelConfig failed: %v", err)
	}

	// Verify JSON file exists next to model
	expectedJson := filepath.Join(tempDir, "MyTestModel-Q4_K_M.json")
	if _, err := os.Stat(expectedJson); os.IsNotExist(err) {
		t.Fatalf("Expected json file %s does not exist", expectedJson)
	}

	// 3. Load again and verify loaded fields
	loadedCfg, err := sm.GetModelConfig(fakeModel)
	if err != nil {
		t.Fatalf("GetModelConfig re-load failed: %v", err)
	}
	if loadedCfg.Temperature != 0.42 {
		t.Errorf("Expected Temperature 0.42, got %f", loadedCfg.Temperature)
	}
	if loadedCfg.MinP != 0.15 {
		t.Errorf("Expected MinP 0.15, got %f", loadedCfg.MinP)
	}
	if loadedCfg.BatchSize != 1024 {
		t.Errorf("Expected BatchSize 1024, got %d", loadedCfg.BatchSize)
	}
	if loadedCfg.MicroBatchSize != 256 {
		t.Errorf("Expected MicroBatchSize 256, got %d", loadedCfg.MicroBatchSize)
	}
	if loadedCfg.ReasoningEffort != "high" {
		t.Errorf("Expected ReasoningEffort 'high', got %s", loadedCfg.ReasoningEffort)
	}
}

func TestTarGzExtractionWithSymlinks(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "llama_targz_test_*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	tarGzPath := filepath.Join(tempDir, "test.tar.gz")
	f, err := os.Create(tarGzPath)
	if err != nil {
		t.Fatalf("Failed to create test archive: %v", err)
	}

	gzWriter := gzip.NewWriter(f)
	tarWriter := tar.NewWriter(gzWriter)

	data := []byte("dylib-binary-content-12345")

	// 1. First write symlink libllama.dylib -> libllama.0.dylib
	_ = tarWriter.WriteHeader(&tar.Header{
		Name:     "llama-b999/libllama.dylib",
		Mode:     0755,
		Typeflag: tar.TypeSymlink,
		Linkname: "libllama.0.dylib",
	})

	// 2. Write symlink libllama.0.dylib -> libllama.0.7.0.dylib
	_ = tarWriter.WriteHeader(&tar.Header{
		Name:     "llama-b999/libllama.0.dylib",
		Mode:     0755,
		Typeflag: tar.TypeSymlink,
		Linkname: "libllama.0.7.0.dylib",
	})

	// 3. Write regular file libllama.0.7.0.dylib
	_ = tarWriter.WriteHeader(&tar.Header{
		Name:     "llama-b999/libllama.0.7.0.dylib",
		Mode:     0755,
		Size:     int64(len(data)),
		Typeflag: tar.TypeReg,
	})
	_, _ = tarWriter.Write(data)

	// 4. Write llama-server executable
	serverData := []byte("fake-server-bin")
	_ = tarWriter.WriteHeader(&tar.Header{
		Name:     "llama-b999/llama-server",
		Mode:     0755,
		Size:     int64(len(serverData)),
		Typeflag: tar.TypeReg,
	})
	_, _ = tarWriter.Write(serverData)

	_ = tarWriter.Close()
	_ = gzWriter.Close()
	_ = f.Close()

	// Extract
	dstDir := filepath.Join(tempDir, "extracted")
	_ = os.MkdirAll(dstDir, 0755)

	rm := NewReleaseManager(dstDir, tempDir)
	err = rm.extractTarGz(tarGzPath, dstDir, func(p ProgressReport) {})
	if err != nil {
		t.Fatalf("extractTarGz failed: %v", err)
	}

	// Verify libllama.0.7.0.dylib exists and non-empty
	pReal := filepath.Join(dstDir, "libllama.0.7.0.dylib")
	infoReal, err := os.Stat(pReal)
	if err != nil || infoReal.Size() != int64(len(data)) {
		t.Fatalf("Expected real dylib with size %d, got err=%v", len(data), err)
	}

	// Verify libllama.0.dylib exists and non-empty (resolved or copied)
	p0 := filepath.Join(dstDir, "libllama.0.dylib")
	info0, err := os.Stat(p0)
	if err != nil {
		t.Fatalf("Failed to stat libllama.0.dylib: %v", err)
	}
	if info0.Size() == 0 {
		t.Fatalf("libllama.0.dylib was extracted as an empty 0-byte file!")
	}

	// Verify libllama.dylib exists and non-empty (resolved or copied)
	pMain := filepath.Join(dstDir, "libllama.dylib")
	infoMain, err := os.Stat(pMain)
	if err != nil {
		t.Fatalf("Failed to stat libllama.dylib: %v", err)
	}
	if infoMain.Size() == 0 {
		t.Fatalf("libllama.dylib was extracted as an empty 0-byte file!")
	}

	// Verify llama-server exists
	pServer := filepath.Join(dstDir, "llama-server")
	if _, err := os.Stat(pServer); err != nil {
		t.Fatalf("Failed to find extracted llama-server: %v", err)
	}
}

func TestVersionManagementLifecycle(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "llama_version_test_*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	binDir := filepath.Join(tempDir, "bin")
	backendsDir := filepath.Join(tempDir, "backends")
	_ = os.MkdirAll(binDir, 0755)
	_ = os.MkdirAll(backendsDir, 0755)

	serverExeName := getServerExeName()

	// 1. Create mock version b4000-cuda
	dir4000 := filepath.Join(backendsDir, "b4000-cuda")
	_ = os.MkdirAll(dir4000, 0755)
	_ = os.WriteFile(filepath.Join(dir4000, serverExeName), []byte("mock-exe-4000"), 0755)
	_ = os.WriteFile(filepath.Join(dir4000, "version.txt"), []byte("b4000\ncuda\nb4000-cuda\n"), 0644)

	// 2. Create mock version b3900-cpu
	dir3900 := filepath.Join(backendsDir, "b3900-cpu")
	_ = os.MkdirAll(dir3900, 0755)
	_ = os.WriteFile(filepath.Join(dir3900, serverExeName), []byte("mock-exe-3900"), 0755)
	_ = os.WriteFile(filepath.Join(dir3900, "version.txt"), []byte("b3900\ncpu\nb3900-cpu\n"), 0644)

	rm := NewReleaseManager(binDir, tempDir)

	// Activate b4000-cuda
	if err := rm.ActivateVersion("b4000-cuda"); err != nil {
		t.Fatalf("ActivateVersion failed: %v", err)
	}

	// Verify binDir has b4000-cuda
	binContent, err := os.ReadFile(filepath.Join(binDir, serverExeName))
	if err != nil || string(binContent) != "mock-exe-4000" {
		t.Fatalf("Expected mock-exe-4000 in binDir, got %s, err: %v", string(binContent), err)
	}

	// Check GetInstalledVersions
	versions := rm.GetInstalledVersions()
	if len(versions) != 2 {
		t.Fatalf("Expected 2 installed versions, got %d", len(versions))
	}

	var activeVersion, inactiveVersion *InstalledVersionInfo
	for i := range versions {
		if versions[i].IsActive {
			activeVersion = &versions[i]
		} else {
			inactiveVersion = &versions[i]
		}
	}

	if activeVersion == nil || activeVersion.Id != "b4000-cuda" {
		t.Fatalf("Expected b4000-cuda to be active, got %+v", activeVersion)
	}
	if inactiveVersion == nil || inactiveVersion.Id != "b3900-cpu" {
		t.Fatalf("Expected b3900-cpu to be inactive, got %+v", inactiveVersion)
	}

	// Switch active version to b3900-cpu
	if err := rm.ActivateVersion("b3900-cpu"); err != nil {
		t.Fatalf("Failed to activate b3900-cpu: %v", err)
	}

	binContent2, err := os.ReadFile(filepath.Join(binDir, serverExeName))
	if err != nil || string(binContent2) != "mock-exe-3900" {
		t.Fatalf("Expected mock-exe-3900 in binDir after switch, got %s, err: %v", string(binContent2), err)
	}

	// Uninstall b4000-cuda
	if err := rm.UninstallVersion("b4000-cuda"); err != nil {
		t.Fatalf("UninstallVersion failed: %v", err)
	}

	if _, err := os.Stat(dir4000); !os.IsNotExist(err) {
		t.Fatalf("Expected dir4000 to be deleted, err: %v", err)
	}

	versionsAfter := rm.GetInstalledVersions()
	if len(versionsAfter) != 1 || versionsAfter[0].Id != "b3900-cpu" || !versionsAfter[0].IsActive {
		t.Fatalf("Expected 1 active version (b3900-cpu), got %+v", versionsAfter)
	}
}

func TestRealStartServerIfExists(t *testing.T) {
	modelPath := `E:\LLM\LLamaManager\FrogNano-4B-2609-Q6_K_L.gguf`
	if _, err := os.Stat(modelPath); err != nil {
		t.Skip("skipping real server test: local model not present")
	}
	appData := filepath.Join(os.Getenv("APPDATA"), "llama-manager")
	svc := NewLlamaService(appData)
	backend := svc.GetInstalledBackend()
	if !backend.Installed {
		t.Skip("skipping real server test: backend not installed")
	}
	cfg, err := svc.GetModelConfig(modelPath)
	if err != nil {
		t.Fatalf("GetModelConfig failed: %v", err)
	}
	if err := svc.StartServer(cfg); err != nil {
		t.Fatalf("StartServer failed: %v", err)
	}
	defer func() {
		_ = svc.StopServer()
	}()

	reply, err := svc.SendChat([]ChatMessage{
		{Role: "user", Content: "Say hi in one word"},
	}, 0.7, "")
	if err != nil {
		t.Fatalf("SendChat failed: %v", err)
	}
	if reply == "" {
		t.Errorf("Expected non-empty reply from chat completion, got empty")
	}
}



