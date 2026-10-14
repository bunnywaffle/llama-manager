# 🦙 llama.cpp Manager

A blazing-fast, lightweight, production-grade desktop application built with the **[MyGo](https://mygo.egoist.dev/)** framework (Go backend + native WebView2 frontend + Vite TypeScript).

Manage, download, update, and run `llama.cpp` backends and `.gguf` AI models with an ultra-clean, modern, anti-slop developer UI.

---

## ⚡ Highlights

- **Ultra-Lightweight & Native Performance**:
  - Standalone portable binary: **~10.3 MB**
  - Memory footprint: **~60 MB RAM** (zero Electron overhead, uses system WebView2).
  - High-performance Go runtime handling process supervision, disk I/O, and hardware telemetry.
- **Hardware Agnostic & Auto-Detection**:
  - Automatically identifies GPUs (NVIDIA GeForce, AMD Radeon via DXGI/WMI/Vulkan, and CPU fallback).
  - Automatically recommends and configures **CUDA 12.4**, **Vulkan** (optimized for AMD Radeon), or **CPU/AVX2**.
- **Server Supervision & Router Mode**:
  - **Single Model Mode**: Point-and-click execution for loaded `.gguf` weights.
  - **Router Mode (`--models-path`)**: Serve an entire directory of models dynamically on demand without restarting the server.
  - Clean child process lifecycle: Start, Stop, and Restart with Windows Job Object guarantees against orphan processes.
  - Real-time streaming console logs with filter, search, and copy.
- **Advanced Parameter Engine**:
  - Collapsible accordion for full parameter control:
    - **Sampling**: Temperature, Top-P, Top-K, Min-P, Repeat Penalty, Repeat Last N, Frequency/Presence Penalty.
    - **Performance & KV Cache**: Flash Attention (`-fa`), KV Cache Type (`f16`, `q8_0`, `q4_0`), Context Shift, Defrag.
    - **Context & RoPE**: Context Size (`-c`), Parallel Slots (`-np`), Batch Size (`-b`), RoPE Frequency Base & Scale.
    - **Engine**: Speculative Decoding / Draft Model, Split Mode, Main GPU index.
- **Branching Chat Tree & Personas**:
  - **Tree-Structured Branching Chat**: Branch off any assistant response, edit past prompts, navigate between siblings (`< 1/2 >`), regenerate answers, and manage conversation threads.
  - **Persona System**: Switch between custom personas (System Prompts) with 1 click or create custom ones on the fly.
  - Live streaming markdown rendering, token-per-second metrics, and stop controls.
- **Model Import & Management**:
  - **Native File Import**: Drag-and-drop or select any `.gguf` from your system with zero-disk-space NTFS hardlinks or direct copy.
  - Automatically detects model quantizations (`Q4_K_M`, `Q8_0`, `Q5_K_M`, etc.).
  - **Curated 1-Click Presets**:
    - *Llama 3.2 3B Instruct*
    - *Qwen 2.5 Coder 7B Instruct*
    - *DeepSeek R1 Distill Qwen 7B*
  - **Hugging Face Explorer**: Enter any Hugging Face repo (e.g. `bartowski/Llama-3.2-3B-Instruct-GGUF`), browse files, and download with live speed (MB/s) and progress resumption.
- **Automated Backend Troubleshooter & Self-Repair**:
  - 6-point automated diagnostics:
    1. Binary existence (`llama-server.exe`)
    2. Architecture compatibility (64-bit AMD64)
    3. GPU runtime DLLs (`cudart64_12.dll` / `vulkan-1.dll`)
    4. Port availability (`127.0.0.1:8080`)
    5. Models directory & permissions
    6. Process permissions & execution rights
  - 1-click **Self-Repair Engine** to reinstall missing binaries or kill zombie port-hogging processes.
- **Comprehensive Settings**:
  - Switch backend engines on the fly (CUDA 12.4, Vulkan, CPU).
  - Custom models directory selector.
  - Custom application data directory selector.

---

## 🚀 Getting Started

### Development Mode (Live Reload)
```powershell
# Install dependencies
npm install

# Run the dev server with instant hot reload
npx mygo-cli dev
```

### Build Portable Executable
```powershell
npx mygo-cli build
```
The compiled standalone executable and Windows NSIS installer will be placed in `build/windows-amd64/`.

---

## 📂 Architecture & Project Structure

```
llama-manager/
├── main.go               # MyGo app initialization & native window setup
├── service.go            # LlamaService RPC methods & typed events
├── hardware.go           # CPU / NVIDIA / AMD GPU hardware introspection
├── release_mgr.go        # GitHub release checker, downloader & zip extractor
├── model_mgr.go          # GGUF scanner, hardlinker, Hugging Face explorer & downloader
├── server_mgr.go         # llama-server.exe process supervisor & log streamer
├── troubleshoot.go       # 6-point backend diagnostics & self-repair engine
├── mygo.config.ts        # MyGo desktop configuration
├── index.html            # Anti-slop neutral dark UI layout
├── src/
│   ├── main.ts           # Reactive frontend controller
│   ├── chat_tree.ts      # Tree-structured branching conversation engine
│   ├── mygo.ts           # Auto-generated TypeScript IPC client
│   └── style.css         # Modern neutral dark theme tokens
```

---

## 📄 License

MIT License. Crafted with ❤️ for local AI enthusiasts.

