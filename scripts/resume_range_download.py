#!/usr/bin/env python3
"""Resume a preallocated HTTP Range download by filling missing chunks."""
from __future__ import annotations

import argparse
import concurrent.futures
from pathlib import Path
import time

import requests


def chunk_present(path: Path, start: int, end: int) -> bool:
    with path.open("rb") as handle:
        handle.seek(start)
        head = handle.read(min(65536, end - start + 1))
        handle.seek(max(start, end - 65535))
        tail = handle.read(min(65536, end - start + 1))
    return bool(head and tail and any(head) and any(tail))


def fetch(url: str, path: Path, start: int, end: int, retries: int) -> int:
    expected = end - start + 1
    for attempt in range(retries):
        try:
            response = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, stream=True, timeout=(20, 90))
            response.raise_for_status()
            if not response.headers.get("Content-Range", "").startswith(f"bytes {start}-{end}/"):
                raise RuntimeError(f"unexpected Content-Range for {start}-{end}")
            offset = start
            with path.open("r+b") as handle:
                handle.seek(start)
                for block in response.iter_content(chunk_size=1024 * 1024):
                    if block:
                        handle.write(block)
                        offset += len(block)
            if offset != end + 1:
                raise RuntimeError(f"short range {offset - start}/{expected}")
            return expected
        except Exception:
            if attempt + 1 == retries:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-mb", type=int, default=16)
    parser.add_argument("--retries", type=int, default=8)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a+b") as handle:
        handle.truncate(args.size)
    chunk = args.chunk_mb * 1024 * 1024
    ranges = [(start, min(start + chunk, args.size) - 1) for start in range(0, args.size, chunk)]
    missing = [(start, end) for start, end in ranges if not chunk_present(args.output, start, end)]
    print(f"chunks={len(ranges)} missing={len(missing)}", flush=True)
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch, args.url, args.output, start, end, args.retries) for start, end in missing]
        for future in concurrent.futures.as_completed(futures):
            done += future.result()
            print(f"filled={done}/{sum(e-s+1 for s,e in missing)}", flush=True)
    if args.output.stat().st_size != args.size:
        raise RuntimeError("output size mismatch")
    print(f"done: {args.output}", flush=True)


if __name__ == "__main__":
    main()
