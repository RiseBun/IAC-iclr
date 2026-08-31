#!/usr/bin/env python3
"""Resumable parallel HTTP Range downloader for large public checkpoints."""

from __future__ import annotations

import argparse
import concurrent.futures
import os
from pathlib import Path
import time

import requests


def _fetch(url: str, output: Path, start: int, end: int, retries: int) -> int:
    expected = end - start + 1
    for attempt in range(retries):
        try:
            response = requests.get(
                url,
                headers={"Range": f"bytes={start}-{end}"},
                stream=True,
                timeout=(30, 180),
            )
            response.raise_for_status()
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {start}-{end}/"):
                raise RuntimeError(f"unexpected Content-Range: {content_range!r}")
            offset = start
            with output.open("r+b") as handle:
                handle.seek(offset)
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
                        offset += len(chunk)
            if offset - start != expected:
                raise RuntimeError(f"short range: got {offset - start}, expected {expected}")
            return expected
        except Exception:
            if attempt + 1 == retries:
                raise
            time.sleep(2.0 * (attempt + 1))
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunk-mb", type=int, default=64)
    parser.add_argument("--retries", type=int, default=6)
    args = parser.parse_args()
    if args.size <= 0 or args.workers <= 0 or args.chunk_mb <= 0:
        raise ValueError("size, workers, and chunk-mb must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a+b") as handle:
        handle.truncate(args.size)
    chunk = args.chunk_mb * 1024 * 1024
    ranges = [(start, min(start + chunk, args.size) - 1) for start in range(0, args.size, chunk)]
    completed = 0
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_fetch, args.url, args.output, start, end, args.retries) for start, end in ranges]
        for future in concurrent.futures.as_completed(futures):
            completed += future.result()
            elapsed = max(time.time() - started, 1e-6)
            print(f"completed={completed}/{args.size} ({completed / args.size:.1%}) speed={completed / elapsed / 1024**2:.2f} MiB/s", flush=True)
    if args.output.stat().st_size != args.size:
        raise RuntimeError("output size mismatch")
    print(f"done: {args.output} ({args.size} bytes)")


if __name__ == "__main__":
    main()
