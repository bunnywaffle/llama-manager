package main

import (
	"runtime"
	"strings"
	"testing"
)

func TestCudaVersionOf(t *testing.T) {
	cases := map[string]string{
		"llama-b11683-bin-ubuntu-cuda-12.8-x64.tar.gz": "12.8",
		"llama-b11683-bin-ubuntu-cuda-13.4-x64.tar.gz": "13.4",
		"cudart-llama-bin-win-cuda-12.4-x64.zip":       "12.4",
		"llama-b11683-bin-ubuntu-x64.tar.gz":           "",
		"llama-b11683-bin-macos-arm64.tar.gz":          "",
		"llama-b11683-bin-win-cpu-x64.zip":             "",
	}
	for asset, want := range cases {
		if got := cudaVersionOf(strings.ToLower(asset)); got != want {
			t.Errorf("cudaVersionOf(%q) = %q, want %q", asset, got, want)
		}
	}
}

// TestCpuMatcherRejectsSpecializedBuilds guards against the core
// Windows-bias bug: generic "cpu" requests must never resolve to OpenVINO,
// SYCL, OpenCL or other specialized accelerator builds.
func TestCpuMatcherRejectsSpecializedBuilds(t *testing.T) {
	rejected := []string{
		"llama-b11683-bin-win-openvino-2026.4.1-x64.zip",
		"llama-b11683-bin-ubuntu-sycl-fp16-x64.tar.gz",
		"llama-b11683-bin-win-sycl-x64.zip",
		"llama-b11683-bin-win-opencl-adreno-arm64.zip",
		"llama-b11683-bin-ubuntu-cuda-12.8-x64.tar.gz",
		"llama-b11683-bin-ubuntu-vulkan-x64.tar.gz",
		"llama-b11683-bin-ubuntu-rocm-10.0-x64.tar.gz",
	}
	for _, asset := range rejected {
		if s := scorePrimaryAsset(strings.ToLower(asset), "cpu"); s >= 0 {
			t.Errorf("scorePrimaryAsset(%q, cpu) = %d, want negative", asset, s)
		}
	}

	accepted := []string{
		"llama-b11683-bin-win-cpu-x64.zip",
		"llama-b11683-bin-ubuntu-x64.tar.gz",
		"llama-b11683-bin-macos-arm64.tar.gz",
	}
	for _, asset := range accepted {
		if s := scorePrimaryAsset(strings.ToLower(asset), "cpu"); s < 0 {
			t.Errorf("scorePrimaryAsset(%q, cpu) = %d, want non-negative", asset, s)
		}
	}
}

// TestHostFilterMatchesCurrentOS ensures release assets for other operating
// systems (and installer-only artifacts) are never offered on this host.
func TestHostFilterMatchesCurrentOS(t *testing.T) {
	if runtime.GOOS != "linux" || runtime.GOARCH != "amd64" {
		t.Skip("host-specific assertions target linux/amd64 CI hosts")
	}
	compatible := []string{
		"llama-b11683-bin-ubuntu-x64.tar.gz",
		"llama-b11683-bin-ubuntu-cuda-12.8-x64.tar.gz",
		"llama-b11683-bin-ubuntu-vulkan-x64.tar.gz",
		"llama-b11683-bin-ubuntu-rocm-10.0-x64.tar.gz",
		"cudart-llama-b11683-bin-ubuntu-cuda-12.8-x64.tar.gz",
	}
	for _, asset := range compatible {
		if !isAssetCompatibleWithHost(asset) {
			t.Errorf("isAssetCompatibleWithHost(%q) = false, want true", asset)
		}
	}
	incompatible := []string{
		"llama-b11683-bin-win-cpu-x64.zip",
		"llama-b11683-bin-win-cuda-12.4-x64.zip",
		"llama-b11683-bin-macos-arm64.tar.gz",
		"llama-b11683-bin-ubuntu-arm64.tar.gz",
		"llama-b11683-bin-ubuntu-s390x.tar.gz",
		"llama-b11683-xcframework.zip",
	}
	for _, asset := range incompatible {
		if isAssetCompatibleWithHost(asset) {
			t.Errorf("isAssetCompatibleWithHost(%q) = true, want false", asset)
		}
	}
}

func TestDefaultBackendForHost(t *testing.T) {
	got := defaultBackendForHost()
	switch runtime.GOOS {
	case "darwin":
		if got != "metal" {
			t.Errorf("defaultBackendForHost() = %q on darwin, want metal", got)
		}
	default:
		if got != "cpu" {
			t.Errorf("defaultBackendForHost() = %q, want cpu", got)
		}
	}
}



func TestCudaWeightedScoringReal(t *testing.T) {
	s134 := scorePrimaryAsset("llama-b11718-bin-win-cuda-13.4-x64.zip", "cuda")
	s128 := scorePrimaryAsset("llama-b11718-bin-win-cuda-12.8-x64.zip", "cuda")
	s124 := scorePrimaryAsset("llama-b11718-bin-win-cuda-12.4-x64.zip", "cuda")
	s118 := scorePrimaryAsset("llama-b11718-bin-win-cuda-11.8-x64.zip", "cuda")

	if !(s134 > s128 && s128 > s124 && s124 > s118) {
		t.Errorf("expected descending CUDA scores: 13.4 (%d) > 12.8 (%d) > 12.4 (%d) > 11.8 (%d)", s134, s128, s124, s118)
	}
}

func TestHostFilterExcludesAndroidAndSnapdragon(t *testing.T) {
	rejected := []string{
		"llama-b11718-bin-android-arm64.tar.gz",
		"llama-b11718-bin-android-arm64-snapdragon.tar.gz",
		"llama-b11718-bin-linux-arm64-snapdragon.tar.gz",
		"llama-b11718-xcframework.zip",
	}
	for _, a := range rejected {
		if isAssetCompatibleWithHost(a) {
			t.Errorf("isAssetCompatibleWithHost(%q) = true, want false", a)
		}
	}
}
