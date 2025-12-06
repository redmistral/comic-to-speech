#!/usr/bin/env python3
"""
Central Configuration Module for Comic-to-Speech Application

This module manages all configuration settings for the distributed architecture:
- Redis queue connection (orchestrator between interface and workers)
- API credentials (OpenAI for LLM narration, Google Cloud for Vision/TTS)
- Feature flags (LLM vs OCR mode)
- File storage paths and cleanup settings
- Queue names and job timeout configurations

Architecture:
    Interface Server → Redis Queue → AI Workers

Settings are loaded from environment variables (.env file) with sensible defaults.
Import this module to access configuration across all components.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Redis Configuration (Queue/Orchestrator)
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_DB = int(os.getenv('REDIS_DB', 0))

# Server Configuration
INTERFACE_SERVER_HOST = '0.0.0.0'
INTERFACE_SERVER_PORT = 5001

# Queue Names
QUEUE_DEFAULT = 'default'
QUEUE_OCR = 'ocr'
QUEUE_TTS = 'tts'

# Job Configuration
JOB_TIMEOUT = '10m'  # Maximum time a job can run
JOB_RESULT_TTL = 3600  # How long to keep job results (1 hour)
JOB_FAILURE_TTL = 86400  # How long to keep failed job info (24 hours)

# Worker Configuration
WORKER_COUNT = int(os.getenv('WORKER_COUNT', 2))  # Number of workers to start

# Google Cloud Configuration
GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

# OpenAI Configuration (for ChatGPT-based text extraction)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
USE_LLM_NARRATOR = os.getenv('USE_LLM_NARRATOR', 'true').lower() == 'true'  # Set to true to use ChatGPT instead of Google OCR

# File Storage
AUDIO_DIR = 'audio_files'
TEMP_DIR = 'temp_images'
FILE_CLEANUP_AGE_HOURS = 1  # Remove files older than this

# Logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
