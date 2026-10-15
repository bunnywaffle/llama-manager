package main

import (
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
