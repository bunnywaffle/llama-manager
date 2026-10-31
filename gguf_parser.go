package main

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"
)

type DetectedModelParams struct {
	Architecture       string   `json:"architecture"`
	BlockCount         int      `json:"blockCount"`
	ContextLength      int      `json:"contextLength"`
	RecommendedContext int      `json:"recommendedContext"`
	EmbeddingLength    int      `json:"embeddingLength"`
	HasSampling        bool     `json:"hasSampling"`
	Temperature        *float64 `json:"temperature,omitempty"`
	TopP               *float64 `json:"topP,omitempty"`
	TopK               *int     `json:"topK,omitempty"`
	MinP               *float64 `json:"minP,omitempty"`
	RepeatPenalty      *float64 `json:"repeatPenalty,omitempty"`
	RoPEFreqBase       float64  `json:"ropeFreqBase,omitempty"`
	RoPEScale          float64  `json:"ropeScale,omitempty"`
	HasThinking        bool     `json:"hasThinking"`
	ReasoningFormat    string   `json:"reasoningFormat"`
	EnableJinja        bool     `json:"enableJinja"`
	MMProjPath         string   `json:"mmprojPath,omitempty"`
	Summary            string   `json:"summary"`
}

// discardBytes reads and discards n bytes efficiently from a buffered reader.
func discardBytes(br *bufio.Reader, n int64) error {
	for n > 0 {
		step := int(n)
		if n > 1<<30 {
			step = 1 << 30
		}
		discarded, err := br.Discard(step)
		n -= int64(discarded)
		if err != nil {
			return err
		}
	}
	return nil
}

// ParseGGUFMetadata reads the header and metadata key-value pairs from a GGUF file.
// It stops immediately after the metadata section without reading tensor data.
func ParseGGUFMetadata(filePath string) (map[string]interface{}, error) {
	f, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open GGUF file: %w", err)
	}
	defer f.Close()

	// 2MB buffer to drastically reduce system call overhead
	br := bufio.NewReaderSize(f, 2*1024*1024)

	var magic [4]byte
	if _, err := io.ReadFull(br, magic[:]); err != nil {
		return nil, fmt.Errorf("failed to read magic header: %w", err)
	}
	if string(magic[:]) != "GGUF" {
		return nil, fmt.Errorf("not a valid GGUF file (magic header %q)", string(magic[:]))
	}

	var version uint32
	if err := binary.Read(br, binary.LittleEndian, &version); err != nil {
		return nil, fmt.Errorf("failed to read GGUF version: %w", err)
	}
	if version < 2 || version > 3 {
		return nil, fmt.Errorf("unsupported GGUF version %d (only v2 and v3 supported)", version)
	}

	var tensorCount uint64
	var kvCount uint64
	if err := binary.Read(br, binary.LittleEndian, &tensorCount); err != nil {
		return nil, err
	}
	if err := binary.Read(br, binary.LittleEndian, &kvCount); err != nil {
		return nil, err
	}

	meta := make(map[string]interface{}, kvCount)

	readGGUFString := func() (string, error) {
		var strLen uint64
		if err := binary.Read(br, binary.LittleEndian, &strLen); err != nil {
			return "", err
		}
		if strLen > 10*1024*1024 { // Sanity limit: 10MB max string length
			return "", fmt.Errorf("string length exceeds sanity threshold: %d", strLen)
		}
		buf := make([]byte, strLen)
		if _, err := io.ReadFull(br, buf); err != nil {
			return "", err
		}
		return string(buf), nil
	}

	var skipValue func(valType uint32) error
	skipValue = func(valType uint32) error {
		switch valType {
		case 0, 1, 7:
			return discardBytes(br, 1)
		case 2, 3:
			return discardBytes(br, 2)
		case 4, 5, 6:
			return discardBytes(br, 4)
		case 10, 11, 12:
			return discardBytes(br, 8)
		case 8: // string
			var sLen uint64
			if err := binary.Read(br, binary.LittleEndian, &sLen); err != nil {
				return err
			}
			return discardBytes(br, int64(sLen))
		case 9: // array
			var arrType uint32
			var arrLen uint64
			if err := binary.Read(br, binary.LittleEndian, &arrType); err != nil {
				return err
			}
			if err := binary.Read(br, binary.LittleEndian, &arrLen); err != nil {
				return err
			}
			switch arrType {
			case 0, 1, 7:
				return discardBytes(br, int64(arrLen))
			case 2, 3:
				return discardBytes(br, int64(arrLen*2))
			case 4, 5, 6:
				return discardBytes(br, int64(arrLen*4))
			case 10, 11, 12:
				return discardBytes(br, int64(arrLen*8))
			case 8: // array of strings (e.g. tokenizer vocabs)
				for i := uint64(0); i < arrLen; i++ {
					var strL uint64
					if err := binary.Read(br, binary.LittleEndian, &strL); err != nil {
						return err
					}
					if err := discardBytes(br, int64(strL)); err != nil {
						return err
					}
				}
				return nil
			default:
				for i := uint64(0); i < arrLen; i++ {
					if err := skipValue(arrType); err != nil {
						return err
					}
				}
				return nil
			}
		default:
			return fmt.Errorf("unknown GGUF value type %d", valType)
		}
	}

	for i := uint64(0); i < kvCount; i++ {
		key, err := readGGUFString()
		if err != nil {
			return nil, fmt.Errorf("error reading metadata key #%d: %w", i, err)
		}

		var valType uint32
		if err := binary.Read(br, binary.LittleEndian, &valType); err != nil {
			return nil, fmt.Errorf("error reading value type for key %q: %w", key, err)
		}

		// Read scalar or string values we care about; skip large arrays (tokens, imatrix)
		switch valType {
		case 0: // uint8
			var v uint8
			_ = binary.Read(br, binary.LittleEndian, &v)
			meta[key] = uint64(v)
		case 1: // int8
			var v int8
			_ = binary.Read(br, binary.LittleEndian, &v)
			meta[key] = int64(v)
		case 2: // uint16
			var v uint16
			_ = binary.Read(br, binary.LittleEndian, &v)
			meta[key] = uint64(v)
		case 3: // int16
			var v int16
			_ = binary.Read(br, binary.LittleEndian, &v)
			meta[key] = int64(v)
		case 4: // uint32
			var v uint32
			_ = binary.Read(br, binary.LittleEndian, &v)
			meta[key] = uint64(v)
		case 5: // int32
			var v int32
			_ = binary.Read(br, binary.LittleEndian, &v)
			meta[key] = int64(v)
		case 6: // float32
			var v float32
			_ = binary.Read(br, binary.LittleEndian, &v)
			meta[key] = float64(v)
		case 7: // bool
			var v uint8
			_ = binary.Read(br, binary.LittleEndian, &v)
			meta[key] = (v != 0)
		case 8: // string
			strVal, err := readGGUFString()
			if err != nil {
				return nil, fmt.Errorf("error reading string value for key %q: %w", key, err)
			}
			meta[key] = strVal
		case 10: // uint64
			var v uint64
			_ = binary.Read(br, binary.LittleEndian, &v)
			meta[key] = v
		case 11: // int64
			var v int64
			_ = binary.Read(br, binary.LittleEndian, &v)
			meta[key] = v
		case 12: // float64
			var v float64
			_ = binary.Read(br, binary.LittleEndian, &v)
			meta[key] = v
		case 9: // array (we skip arrays like tokens to maintain lightning fast parsing)
			if err := skipValue(9); err != nil {
				return nil, fmt.Errorf("error skipping array for key %q: %w", key, err)
			}
		default:
			return nil, fmt.Errorf("unknown GGUF value type %d for key %q", valType, key)
		}
	}

	return meta, nil
}

func getMetaInt(meta map[string]interface{}, keys ...string) (int, bool) {
	for _, k := range keys {
		if raw, ok := meta[k]; ok {
			switch v := raw.(type) {
			case uint64:
				return int(v), true
			case int64:
				return int(v), true
			case float64:
				return int(v), true
			}
		}
	}
	return 0, false
}

func getMetaFloat(meta map[string]interface{}, keys ...string) (float64, bool) {
	for _, k := range keys {
		if raw, ok := meta[k]; ok {
			switch v := raw.(type) {
			case float64:
				return v, true
			case uint64:
				return float64(v), true
			case int64:
				return float64(v), true
			}
		}
	}
	return 0, false
}

func getMetaString(meta map[string]interface{}, keys ...string) (string, bool) {
	for _, k := range keys {
		if raw, ok := meta[k]; ok {
			if s, ok := raw.(string); ok && strings.TrimSpace(s) != "" {
				return strings.TrimSpace(s), true
			}
		}
	}
	return "", false
}

// ExtractModelParameters parses GGUF metadata and derives optimal parameters.
func ExtractModelParameters(filePath string) (*DetectedModelParams, error) {
	start := time.Now()
	meta, err := ParseGGUFMetadata(filePath)
	if err != nil {
		return nil, err
	}
	elapsed := time.Since(start)

	arch, _ := getMetaString(meta, "general.architecture")
	if arch == "" {
		arch = "model"
	}

	// Block count (transformer layers)
	blockCount, _ := getMetaInt(meta,
		arch+".block_count",
		"general.block_count",
		"llm.block_count",
	)

	// Native context length
	contextLength, _ := getMetaInt(meta,
		arch+".context_length",
		"general.context_length",
		"llm.context_length",
	)
	if contextLength <= 0 {
		contextLength = 4096
	}

	// Recommended context for balanced performance
	recommendedCtx := contextLength
	if recommendedCtx > 32768 {
		recommendedCtx = 16384 // Safe balanced default
	}

	// Embedding length
	embeddingLength, _ := getMetaInt(meta,
		arch+".embedding_length",
		"general.embedding_length",
	)

	// Sampling parameters
	params := &DetectedModelParams{
		Architecture:       arch,
		BlockCount:         blockCount,
		ContextLength:      contextLength,
		RecommendedContext: recommendedCtx,
		EmbeddingLength:    embeddingLength,
		ReasoningFormat:    "none",
	}

	// RoPE parameters
	if freqBase, ok := getMetaFloat(meta, arch+".rope.freq_base", "general.rope.freq_base"); ok && freqBase > 0 {
		params.RoPEFreqBase = freqBase
	}
	if ropeScale, ok := getMetaFloat(meta, arch+".rope.scale_linear", "general.rope.scale_linear"); ok && ropeScale > 0 {
		params.RoPEScale = ropeScale
	}

	// Sampling parameters embedded in GGUF
	if temp, ok := getMetaFloat(meta, "general.sampling.temp", "general.sampling.temperature"); ok && temp >= 0 {
		// Round to 2 decimals
		t := math.Round(temp*100) / 100
		params.Temperature = &t
		params.HasSampling = true
	}
	if topP, ok := getMetaFloat(meta, "general.sampling.top_p"); ok && topP > 0 && topP <= 1.0 {
		p := math.Round(topP*100) / 100
		params.TopP = &p
		params.HasSampling = true
	}
	if topK, ok := getMetaInt(meta, "general.sampling.top_k"); ok && topK > 0 {
		params.TopK = &topK
		params.HasSampling = true
	}
	if minP, ok := getMetaFloat(meta, "general.sampling.min_p"); ok && minP > 0 && minP <= 1.0 {
		p := math.Round(minP*100) / 100
		params.MinP = &p
		params.HasSampling = true
	}
	if repPen, ok := getMetaFloat(meta, "general.sampling.repeat_penalty", "general.sampling.repetition_penalty"); ok && repPen > 0 {
		rp := math.Round(repPen*100) / 100
		params.RepeatPenalty = &rp
		params.HasSampling = true
	}

	// Chat template & reasoning detection
	chatTemplate, _ := getMetaString(meta, "tokenizer.chat_template")
	lowerTemplate := strings.ToLower(chatTemplate)

	hasJinja := strings.Contains(chatTemplate, "{%") || strings.Contains(chatTemplate, "{{")
	params.EnableJinja = hasJinja

	hasThinking := strings.Contains(lowerTemplate, "<think>") ||
		strings.Contains(lowerTemplate, "preserve_thinking") ||
		strings.Contains(lowerTemplate, "reasoning_content") ||
		strings.Contains(lowerTemplate, "deepseek") ||
		strings.Contains(lowerTemplate, "thought")

	params.HasThinking = hasThinking
	if hasThinking {
		params.ReasoningFormat = "deepseek-legacy"
		params.EnableJinja = true
	}

	// Auto-detect adjacent mmproj if any
	baseNoExt := strings.TrimSuffix(filepath.Base(filePath), filepath.Ext(filePath))
	dir := filepath.Dir(filePath)
	if entries, err := os.ReadDir(dir); err == nil {
		for _, e := range entries {
			name := strings.ToLower(e.Name())
			if strings.HasSuffix(name, ".gguf") && strings.Contains(name, "mmproj") {
				if strings.HasPrefix(strings.ToLower(e.Name()), strings.ToLower(baseNoExt)) || params.MMProjPath == "" {
					params.MMProjPath = filepath.Join(dir, e.Name())
				}
			}
		}
	}

	var summaryParts []string
	summaryParts = append(summaryParts, fmt.Sprintf("%d layers", blockCount))
	if contextLength >= 1024 {
		summaryParts = append(summaryParts, fmt.Sprintf("%dK context (native)", contextLength/1024))
	} else {
		summaryParts = append(summaryParts, fmt.Sprintf("%d context", contextLength))
	}
	if params.HasSampling {
		if params.Temperature != nil {
			summaryParts = append(summaryParts, fmt.Sprintf("temp %.2f", *params.Temperature))
		}
		if params.TopP != nil {
			summaryParts = append(summaryParts, fmt.Sprintf("top_p %.2f", *params.TopP))
		}
	}
	if params.HasThinking {
		summaryParts = append(summaryParts, "reasoning/thinking enabled")
	}
	if params.MMProjPath != "" {
		summaryParts = append(summaryParts, "vision projector linked")
	}
	summaryParts = append(summaryParts, fmt.Sprintf("parsed in %v", elapsed.Round(time.Millisecond)))

	params.Summary = strings.Join(summaryParts, " · ")
	return params, nil
}
