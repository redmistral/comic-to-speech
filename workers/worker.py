#!/usr/bin/env python3
"""
AI Worker - Processes OCR and TTS tasks from the queue

This worker connects to Redis and processes tasks enqueued by the interface server.
You can run multiple instances of this worker for parallel processing.

Usage:
    python start_worker.py

Or to run multiple workers:
    python start_worker.py &
    python start_worker.py &
    python start_worker.py &
"""

import os
import platform
import sys

if sys.version_info[:3] != (3, 11, 9):
    print(f"⚠️  WARNING: This application requires Python 3.11.9")
    print(f"   Current version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print(f"   Some features may not work correctly with other versions")
    print()

from redis import Redis
from rq import Worker, Queue

try:
    from rq import Connection
except ImportError:
    # RQ >=2 moved Connection into rq.connections
    from rq.connections import Connection

try:
    from rq.worker import SimpleWorker
except ImportError:
    SimpleWorker = None

import config

# Import tasks module to ensure it's loaded before worker starts
# This helps RQ find the task functions
from workers import tasks

# Connect to Redis
redis_conn = Redis(host=config.REDIS_HOST, port=config.REDIS_PORT, db=config.REDIS_DB)

# List of queues to listen to (in order of priority)
listen = ['default', 'ocr', 'tts']

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🤖 AI WORKER - Comic Processing Worker")
    print("="*70)
    print(f"✓ Connecting to Redis at {config.REDIS_HOST}:{config.REDIS_PORT}...")

    try:
        redis_conn.ping()
        print(f"✓ Connected to Redis successfully")
    except Exception as e:
        print(f"❌ Could not connect to Redis at {config.REDIS_HOST}:{config.REDIS_PORT}")
        print(f"   Error: {e}")
        print(f"   Please ensure Redis is running on the orchestrator server")
        print(f"   Set REDIS_HOST environment variable if using remote Redis")
        sys.exit(1)

    env_worker = os.environ.get("RQ_WORKER_CLASS", "").strip().lower()
    default_to_simple = platform.system() == "Darwin"

    if env_worker not in {"", "worker", "simple"}:
        print(f"⚠️  Unknown RQ_WORKER_CLASS='{env_worker}'. Falling back to default.")
        env_worker = ""

    use_simple_worker = False
    if env_worker == "simple":
        use_simple_worker = True
    elif env_worker == "worker":
        use_simple_worker = False
    else:
        use_simple_worker = default_to_simple

    if use_simple_worker and SimpleWorker is None:
        print("⚠️  SimpleWorker not available in this RQ version. Using standard Worker instead.")
        use_simple_worker = False

    worker_cls = SimpleWorker if use_simple_worker else Worker
    worker_mode = "SimpleWorker (no fork)" if use_simple_worker else "Worker (prefork)"

    print(f"✓ Listening to queues: {', '.join(listen)}")
    print(f"✓ Worker class: {worker_mode}")
    if use_simple_worker:
        print("  ℹ️  SimpleWorker avoids macOS fork safety crashes. Set RQ_WORKER_CLASS=worker to force prefork mode.")
    print(f"✓ Worker is ready to process tasks")
    print("="*70 + "\n")

    # Start the worker
    with Connection(redis_conn):
        worker = worker_cls(list(map(Queue, listen)))
        worker.work()
