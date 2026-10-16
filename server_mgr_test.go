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
