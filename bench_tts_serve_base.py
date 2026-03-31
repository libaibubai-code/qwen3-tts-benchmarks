"""Benchmark client for Qwen3-TTS-Base (Voice Cloning) via /v1/audio/speech endpoint.

Measures TTFP (Time-to-First-Packet), E2E latency, and RTF (Real-Time Factor)
across configurable concurrency levels. Supports voice cloning with reference audio.

Base Task Requirements:
- ref_audio: Reference audio file path, URL, or base64 data URL (required)
- ref_text: Transcript of reference audio (required for ICL mode, optional for x-vector only)
- x_vector_only_mode: Use speaker embedding only without ICL (optional)

Usage:
    # Single reference audio
    python bench_tts_serve_base.py \
        --host 127.0.0.1 --port 8000 \
        --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
        --ref-audio /path/to/reference.wav \
        --ref-text "This is the reference transcript" \
        --num-prompts 50 \
        --max-concurrency 1 4 10

    # Multiple reference audios (randomly selected per request)
    python bench_tts_serve_base.py \
        --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
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
"""

import argparse
import asyncio
import base64
import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm

PROMPTS = [
    # English prompts
    "Hello, welcome to the voice synthesis benchmark test.",
    "She said she would be here by noon, but nobody showed up.",
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "I can't believe how beautiful the sunset looks from up here on the mountain.",
    "Please remember to bring your identification documents to the appointment tomorrow morning.",
    "Have you ever wondered what it would be like to travel through time and visit ancient civilizations?",
    "The restaurant on the corner serves the best pasta I have ever tasted in my entire life.",
    "After the meeting, we should discuss the quarterly results and plan for the next phase.",
    "Learning a new language takes patience, practice, and a genuine curiosity about other cultures.",
    "The train leaves at half past seven, so we need to arrive at the station before then.",
    # Chinese prompts
    "你好，欢迎使用语音合成基准测试。",
    "今天天气真好，适合出去散步。",
    "人工智能正在改变我们的生活方式。",
    "这本书的内容非常有趣，我读得津津有味。",
    "希望你能喜欢这个语音合成模型的表现。",
    "科学技术的发展让我们的生活变得更加便利。",
    "学习一门新语言需要耐心和持续的练习。",
]


@dataclass
class RequestResult:
    """Result of a single TTS request."""
    success: bool = False
    ttfp: float = 0.0  # Time to first audio packet (seconds)
    e2e: float = 0.0  # End-to-end latency (seconds)
    audio_bytes: int = 0  # Total audio bytes received
    audio_duration: float = 0.0  # Audio duration in seconds (estimated from PCM)
    rtf: float = 0.0  # Real-time factor = e2e / audio_duration
    prompt: str = ""
    error: str = ""
    prompt_language: str = ""
    ref_audio_used: str = ""


@dataclass
class BenchmarkResult:
    """Aggregated benchmark results."""
    config_name: str = ""
    concurrency: int = 0
    num_prompts: int = 0
    completed: int = 0
    failed: int = 0
    duration_s: float = 0.0
    # TTFP stats (ms)
    mean_ttfp_ms: float = 0.0
    median_ttfp_ms: float = 0.0
    std_ttfp_ms: float = 0.0
    p90_ttfp_ms: float = 0.0
    p95_ttfp_ms: float = 0.0
    p99_ttfp_ms: float = 0.0
    # E2E stats (ms)
    mean_e2e_ms: float = 0.0
    median_e2e_ms: float = 0.0
    std_e2e_ms: float = 0.0
    p90_e2e_ms: float = 0.0
    p95_e2e_ms: float = 0.0
    p99_e2e_ms: float = 0.0
    # RTF stats
    mean_rtf: float = 0.0
    median_rtf: float = 0.0
    std_rtf: float = 0.0
    p99_rtf: float = 0.0
    # Audio stats
    mean_audio_duration_s: float = 0.0
    total_audio_duration_s: float = 0.0
    audio_throughput: float = 0.0  # audio_duration / wall_time
    request_throughput: float = 0.0  # requests / second
    # Per-request details
    per_request: list = field(default_factory=list)


def pcm_bytes_to_duration(num_bytes: int, sample_rate: int = 24000, sample_width: int = 2) -> float:
    """Convert raw PCM byte count to duration in seconds.
    
    Qwen3-TTS uses 24kHz sample rate with 16-bit PCM (2 bytes per sample).
    """
    num_samples = num_bytes / sample_width
    return num_samples / sample_rate


def detect_prompt_language(prompt: str) -> str:
    """Simple heuristic to detect prompt language."""
    if any('\u4e00' <= c <= '\u9fff' for c in prompt):
        return "zh"
    return "en"


def encode_audio_to_base64(audio_path: str) -> str:
    """Encode a local audio file to base64 data URL.
    
    Args:
        audio_path: Path to audio file or URL
    
    Returns:
        Base64 data URL or original URL if already remote
    """
    if audio_path.startswith(("http://", "https://")):
        return audio_path
    
    if audio_path.startswith("data:"):
        return audio_path
    
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Reference audio file not found: {audio_path}")
    
    ext = os.path.splitext(audio_path)[1].lower()
    mime_map = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
        ".webm": "audio/webm",
    }
    mime_type = mime_map.get(ext, "audio/wav")
    
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{audio_b64}"


async def send_tts_request(
    session: aiohttp.ClientSession,
    api_url: str,
    prompt: str,
    model: str,
    ref_audio_b64: str,
    ref_text: str,
    x_vector_only: bool = False,
    stream: bool = True,
    response_format: str = "pcm",
    pbar: tqdm | None = None,
) -> RequestResult:
    """Send a streaming TTS request for Base task (voice cloning).
    
    Args:
        session: aiohttp session
        api_url: TTS endpoint URL
        prompt: Text to synthesize
        model: Model name or path
        ref_audio_b64: Reference audio as base64 data URL or URL
        ref_text: Reference audio transcript
        x_vector_only: Use x-vector only mode (no ICL)
        stream: Enable streaming response
        response_format: Audio format (pcm, wav, etc.)
        pbar: Progress bar to update
    
    Returns:
        RequestResult with latency metrics
    """
    # Build payload for Qwen3-TTS-Base
    payload = {
        "model": model,
        "input": prompt,
        "task_type": "Base",
        "ref_audio": ref_audio_b64,
        "stream": stream,
        "response_format": response_format,
    }
    
    # Add ref_text if provided (required for ICL mode)
    if ref_text and ref_text.strip():
        payload["ref_text"] = ref_text.strip()
    
    # Add x_vector_only_mode flag
    if x_vector_only:
        payload["x_vector_only_mode"] = True
    
    prompt_lang = detect_prompt_language(prompt)
    result = RequestResult(
        prompt=prompt,
        prompt_language=prompt_lang,
        ref_audio_used=ref_audio_b64[:50] + "..." if len(ref_audio_b64) > 50 else ref_audio_b64
    )
    
    st = time.perf_counter()
    
    try:
        async with session.post(api_url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                result.error = f"HTTP {response.status}: {error_text}"
                result.success = False
                return result
            
            first_chunk = True
            total_bytes = 0
            
            async for chunk in response.content.iter_any():
                if first_chunk and len(chunk) > 0:
                    result.ttfp = time.perf_counter() - st
                    first_chunk = False
                total_bytes += len(chunk)
            
            result.e2e = time.perf_counter() - st
            result.audio_bytes = total_bytes
            result.audio_duration = pcm_bytes_to_duration(total_bytes)
            
            if result.audio_duration > 0:
                result.rtf = result.e2e / result.audio_duration
            result.success = True
    
    except asyncio.TimeoutError:
        result.error = "Request timeout (600s)"
        result.success = False
        result.e2e = time.perf_counter() - st
    
    except Exception as e:
        result.error = str(e)
        result.success = False
        result.e2e = time.perf_counter() - st
    
    if pbar:
        pbar.update(1)
    return result


async def run_benchmark(
    host: str,
    port: int,
    num_prompts: int,
    max_concurrency: int,
    model: str,
    ref_audios: list[str],
    ref_texts: list[str],
    x_vector_only: bool = False,
    num_warmups: int = 3,
) -> BenchmarkResult:
    """Run benchmark at a given concurrency level.
    
    Args:
        host: Server hostname
        port: Server port
        num_prompts: Number of prompts to test
        max_concurrency: Maximum concurrent requests
        model: Model name or path
        ref_audios: List of reference audio paths/URLs
        ref_texts: List of reference transcripts
        x_vector_only: Use x-vector only mode
        num_warmups: Number of warmup requests
    
    Returns:
        BenchmarkResult with aggregated metrics
    """
    api_url = f"http://{host}:{port}/v1/audio/speech"
    
    # Validate reference audio/text pairing
    if len(ref_audios) != len(ref_texts):
        # If only one ref_text provided, use it for all ref_audios
        if len(ref_texts) == 1:
            ref_texts = ref_texts * len(ref_audios)
        else:
            raise ValueError(
                f"Number of ref_audios ({len(ref_audios)}) must match "
                f"number of ref_texts ({len(ref_texts)}) or ref_texts should have exactly 1 entry"
            )
    
    # Encode reference audios to base64
    ref_audios_b64 = [encode_audio_to_base64(path) for path in ref_audios]
    
    # Configure connection pool
    connector = aiohttp.TCPConnector(
        limit=max_concurrency,
        limit_per_host=max_concurrency,
        keepalive_timeout=60,
    )
    session = aiohttp.ClientSession(
        connector=connector,
        timeout=aiohttp.ClientTimeout(total=600),
    )
    
    # Warmup phase
    if num_warmups > 0 and ref_audios_b64:
        print(f"  Warming up with {num_warmups} requests...")
        print(f"  Model: {model}")
        warmup_prompt = PROMPTS[0]
        warmup_ref_audio = ref_audios_b64[0]
        warmup_ref_text = ref_texts[0]
        
        warmup_tasks = []
        for i in range(num_warmups):
            warmup_tasks.append(
                send_tts_request(
                    session, api_url, warmup_prompt, model,
                    warmup_ref_audio, warmup_ref_text, x_vector_only
                )
            )
        await asyncio.gather(*warmup_tasks)
        print("  Warmup done.")
    
    # Build request list with random reference audio selection
    request_prompts = [PROMPTS[i % len(PROMPTS)] for i in range(num_prompts)]
    
    # Run benchmark with concurrency control
    print(f"  Running {num_prompts} requests with concurrency={max_concurrency}...")
    print(f"  Using {len(ref_audios_b64)} reference audio sample(s)")
    print(f"  X-vector only mode: {x_vector_only}")
    
    semaphore = asyncio.Semaphore(max_concurrency)
    pbar = tqdm(total=num_prompts, desc=f"  concurrency={max_concurrency}")
    
    async def limited_request(prompt_idx: int):
        async with semaphore:
            # Randomly select a reference audio for each request
            ref_idx = random.randint(0, len(ref_audios_b64) - 1)
            return await send_tts_request(
                session, api_url, request_prompts[prompt_idx], model,
                ref_audios_b64[ref_idx], ref_texts[ref_idx], x_vector_only,
                pbar=pbar
            )
    
    start_time = time.perf_counter()
    tasks = [asyncio.create_task(limited_request(i)) for i in range(num_prompts)]
    results: list[RequestResult] = await asyncio.gather(*tasks)
    duration = time.perf_counter() - start_time
    pbar.close()
    
    await session.close()
    
    # Compute statistics
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    bench = BenchmarkResult(
        concurrency=max_concurrency,
        num_prompts=num_prompts,
        completed=len(successful),
        failed=len(failed),
        duration_s=duration,
    )
    
    if successful:
        ttfps = [r.ttfp * 1000 for r in successful]  # convert to ms
        e2es = [r.e2e * 1000 for r in successful]
        rtfs = [r.rtf for r in successful]
        audio_durs = [r.audio_duration for r in successful]
        
        # TTFP statistics
        bench.mean_ttfp_ms = float(np.mean(ttfps))
        bench.median_ttfp_ms = float(np.median(ttfps))
        bench.std_ttfp_ms = float(np.std(ttfps))
        bench.p90_ttfp_ms = float(np.percentile(ttfps, 90))
        bench.p95_ttfp_ms = float(np.percentile(ttfps, 95))
        bench.p99_ttfp_ms = float(np.percentile(ttfps, 99))
        
        # E2E latency statistics
        bench.mean_e2e_ms = float(np.mean(e2es))
        bench.median_e2e_ms = float(np.median(e2es))
        bench.std_e2e_ms = float(np.std(e2es))
        bench.p90_e2e_ms = float(np.percentile(e2es, 90))
        bench.p95_e2e_ms = float(np.percentile(e2es, 95))
        bench.p99_e2e_ms = float(np.percentile(e2es, 99))
        
        # RTF statistics
        bench.mean_rtf = float(np.mean(rtfs))
        bench.median_rtf = float(np.median(rtfs))
        bench.std_rtf = float(np.std(rtfs))
        bench.p99_rtf = float(np.percentile(rtfs, 99))
        
        # Audio statistics
        bench.mean_audio_duration_s = float(np.mean(audio_durs))
        bench.total_audio_duration_s = float(np.sum(audio_durs))
        bench.audio_throughput = bench.total_audio_duration_s / duration
        bench.request_throughput = len(successful) / duration
        
        # Per-request details
        bench.per_request = [
            {
                "ttfp_ms": r.ttfp * 1000,
                "e2e_ms": r.e2e * 1000,
                "rtf": r.rtf,
                "audio_duration_s": r.audio_duration,
                "prompt": r.prompt,
                "language": r.prompt_language,
                "ref_audio": r.ref_audio_used,
            }
            for r in successful
        ]
    
    # Print summary
    W = 50
    print("")
    print(f"{'=' * W}")
    print(f"{'Serving Benchmark Result (Qwen3-TTS-Base)':^{W}}")
    print(f"{'=' * W}")
    print(f"{'Successful requests:':<40}{bench.completed:<10}")
    print(f"{'Failed requests:':<40}{bench.failed:<10}")
    print(f"{'Maximum request concurrency:':<40}{max_concurrency:<10}")
    print(f"{'Benchmark duration (s):':<40}{duration:<10.2f}")
    print(f"{'Request throughput (req/s):':<40}{bench.request_throughput:<10.2f}")
    print(f"{'-' * W}")
    print(f"{'End-to-End Latency':^{W}}")
    print(f"{'-' * W}")
    print(f"{'Mean E2EL (ms):':<40}{bench.mean_e2e_ms:<10.2f}")
    print(f"{'Median E2EL (ms):':<40}{bench.median_e2e_ms:<10.2f}")
    print(f"{'P99 E2EL (ms):':<40}{bench.p99_e2e_ms:<10.2f}")
    print(f"{'=' * W}")
    print(f"{'Audio Result':^{W}}")
    print(f"{'=' * W}")
    print(f"{'Total audio duration generated (s):':<40}{bench.total_audio_duration_s:<10.2f}")
    print(f"{'Audio throughput (audio duration/s):':<40}{bench.audio_throughput:<10.2f}")
    print(f"{'-' * W}")
    print(f"{'Time to First Packet':^{W}}")
    print(f"{'-' * W}")
    print(f"{'Mean AUDIO_TTFP (ms):':<40}{bench.mean_ttfp_ms:<10.2f}")
    print(f"{'Median AUDIO_TTFP (ms):':<40}{bench.median_ttfp_ms:<10.2f}")
    print(f"{'P99 AUDIO_TTFP (ms):':<40}{bench.p99_ttfp_ms:<10.2f}")
    print(f"{'-' * W}")
    print(f"{'Real Time Factor':^{W}}")
    print(f"{'-' * W}")
    print(f"{'Mean AUDIO_RTF:':<40}{bench.mean_rtf:<10.3f}")
    print(f"{'Median AUDIO_RTF:':<40}{bench.median_rtf:<10.3f}")
    print(f"{'P99 AUDIO_RTF:':<40}{bench.p99_rtf:<10.3f}")
    print(f"{'=' * W}")
    print("")
    
    # Print language distribution
    if successful:
        lang_counts = {}
        for r in successful:
            lang = r.prompt_language
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        print(f"  Prompt language distribution: {lang_counts}")
    
    # Print errors
    if failed:
        print(f"\n  Failed requests: {len(failed)}")
        for r in failed[:5]:
            print(f"    [ERROR] {r.error[:200]}")
    
    return bench


async def main(args):
    """Main entry point."""
    all_results = []
    
    for concurrency in args.max_concurrency:
        result = await run_benchmark(
            host=args.host,
            port=args.port,
            num_prompts=args.num_prompts,
            max_concurrency=concurrency,
            model=args.model,
            ref_audios=args.ref_audio,
            ref_texts=args.ref_text,
            x_vector_only=args.x_vector_only,
            num_warmups=args.num_warmups,
        )
        result.config_name = args.config_name
        all_results.append(asdict(result))
    
    # Save results
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = result_dir / f"bench_{args.config_name}_{timestamp}.json"
    
    with open(result_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {result_file}")
    
    return all_results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS-Base (Voice Cloning) Benchmark Client",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Server configuration
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server hostname")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    
    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        help="Model name or path (e.g., Qwen/Qwen3-TTS-12Hz-1.7B-Base, Qwen/Qwen3-TTS-12Hz-0.6B-Base, or local path)"
    )
    
    # Reference audio (required for Base task)
    parser.add_argument(
        "--ref-audio",
        type=str,
        nargs="+",
        required=True,
        help="Reference audio file path(s), URL(s), or base64 data URL(s). "
             "Multiple files can be provided for varied voice cloning."
    )
    
    # Reference text (required for ICL mode)
    parser.add_argument(
        "--ref-text",
        type=str,
        nargs="+",
        required=True,
        help="Reference audio transcript(s). Must match number of ref-audio entries, "
             "or provide exactly one transcript to use for all reference audios."
    )
    
    # Benchmark settings
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=50,
        help="Number of prompts per concurrency level"
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        nargs="+",
        default=[1, 4, 10],
        help="Concurrency levels to test"
    )
    parser.add_argument(
        "--num-warmups",
        type=int,
        default=3,
        help="Number of warmup requests"
    )
    
    # Base task options
    parser.add_argument(
        "--x-vector-only",
        action="store_true",
        help="Use x-vector only mode (speaker embedding without ICL)"
    )
    
    # Output configuration
    parser.add_argument(
        "--config-name",
        type=str,
        default="base_voice_clone",
        help="Label for this config (used in filenames)"
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default="results",
        help="Directory to save results"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        raise
