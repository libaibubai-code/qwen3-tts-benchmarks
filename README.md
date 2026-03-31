# Qwen3-TTS Benchmarks

Performance benchmarking scripts for Qwen3-TTS models via vLLM-Omni serving.

## 📁 Scripts

### Base Model (Voice Cloning)

**`bench_tts_serve_base.py`** - Benchmark script for Qwen3-TTS-Base model with voice cloning support

- Requires reference audio and transcript
- Supports multiple reference samples
- Measures TTFP, E2E latency, RTF, and throughput
- JSON result export with per-request details

### CustomVoice Model

**`bench_tts_serve.py`** - Benchmark script for Qwen3-TTS-CustomVoice model

- Predefined speaker voices (vivian, ryan, aiden, etc.)
- Optional style instructions
- Same metrics as Base model

## 🚀 Quick Start

### Prerequisites

```bash
pip install aiohttp numpy tqdm
```

### Base Model Benchmark

```bash
# Single reference audio
python bench_tts_serve_base.py \
    --host 127.0.0.1 --port 8000 \
    --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --ref-audio /path/to/reference.wav \
    --ref-text "This is the reference transcript" \
    --num-prompts 50 \
    --max-concurrency 1 4 10

# Different model variant (0.6B)
python bench_tts_serve_base.py \
    --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
    --ref-audio reference.wav \
    --ref-text "Reference transcript" \
    --num-prompts 50

# Multiple reference audios (randomly selected per request)
python bench_tts_serve_base.py \
    --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --ref-audio ref1.wav ref2.wav ref3.wav \
    --ref-text "Transcript 1" "Transcript 2" "Transcript 3" \
    --num-prompts 50

# X-vector only mode (no ICL)
python bench_tts_serve_base.py \
    --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --ref-audio reference.wav \
    --ref-text "Reference transcript" \
    --x-vector-only \
    --num-prompts 50
```

### CustomVoice Model Benchmark

```bash
python bench_tts_serve.py \
    --host 127.0.0.1 --port 8000 \
    --voice vivian \
    --language English \
    --num-prompts 50 \
    --max-concurrency 1 4 10
```

## 📊 Metrics

| Metric | Description |
|--------|-------------|
| **TTFP** | Time to First Packet - latency from request to first audio chunk |
| **E2E** | End-to-End - total time from request to complete response |
| **RTF** | Real-Time Factor - E2E / audio_duration (RTF < 1 means faster than real-time) |
| **Throughput** | Requests per second and audio seconds per wall-clock second |

## 📈 Output Example

```
==================================================
       Serving Benchmark Result (Qwen3-TTS-Base)       
==================================================
Successful requests:                  50        
Failed requests:                      0         
Maximum request concurrency:          10        
Benchmark duration (s):               52.34     
Request throughput (req/s):           0.96      
--------------------------------------------------
Mean E2EL (ms):                       1023.45   
Median E2EL (ms):                     890.23    
P99 E2EL (ms):                        1850.67   
==================================================
Mean AUDIO_TTFP (ms):                 145.67    
Mean AUDIO_RTF:                       0.412     
==================================================
```

Results are saved as JSON files with:
- Aggregated statistics (mean, median, p90, p95, p99)
- Per-request details
- Configuration metadata

## 🔗 Related Projects

- [vLLM-Omni](https://github.com/vllm-project/vllm-omni)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)

## 📝 License

MIT License
