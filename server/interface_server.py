#!/usr/bin/env python3
"""
Interface Server - Web UI and API Gateway for Comic-to-Speech System

This Flask-based server provides the user-facing interface for the distributed
comic processing pipeline. It handles:

Responsibilities:
    - Serve web UI for comic uploads
    - Accept comic image uploads via POST /api/process-comic
    - Enqueue processing jobs to Redis queue
    - Poll job status and return results via GET /api/job-status/<id>
    - Serve generated audio files via GET /api/audio/<id>

Architecture Role:
    This server does NOT perform AI processing directly. It acts as a lightweight
    gateway that enqueues jobs to Redis, where separate AI worker processes
    execute the actual OCR, LLM narration, translation, and TTS tasks.

    User → Interface Server → Redis Queue → AI Workers
                ↑__________________|
                  (Poll job status)

Usage:
    python server/interface_server.py

Then access http://localhost:5001 in your browser.
"""

import sys
import os
from pathlib import Path

if sys.version_info[:3] != (3, 11, 9):
    print(f"⚠️  WARNING: This application requires Python 3.11.9")
    print(f"   Current version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print(f"   Some features may not work correctly with other versions")
    print()

from flask import Flask, request, jsonify, send_file, render_template_string
from flask_cors import CORS
from redis import Redis
from rq import Queue
from rq.job import Job
import uuid
from datetime import datetime, timedelta
import config





app = Flask(__name__)
CORS(app)

def validate_image_upload(file):
    """
    Validate uploaded image:
    - must exist
    - must have allowed extension
    - must be under 10 MB
    """
    if file is None:
        return False, "No file uploaded"

    allowed_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
    filename = file.filename.lower()

    if not any(filename.endswith(ext) for ext in allowed_extensions):
        return False, "Unsupported file type"

    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)

    if size > 10 * 1024 * 1024:
        return False, "File too large"

    return True, None


# Connect to Redis (orchestrator/queue)
try:
    redis_conn = Redis(host=config.REDIS_HOST, port=config.REDIS_PORT, db=config.REDIS_DB)
    redis_conn.ping()
    print(f"✓ Connected to Redis queue at {config.REDIS_HOST}:{config.REDIS_PORT}")
except Exception as e:
    print(f"❌ Could not connect to Redis at {config.REDIS_HOST}:{config.REDIS_PORT}")
    print(f"   Error: {e}")
    print("   Please ensure Redis is running on the orchestrator server")
    redis_conn = None

# Create task queues
# We can have different queues for different priorities or task types
ocr_queue = Queue('ocr', connection=redis_conn) if redis_conn else None
tts_queue = Queue('tts', connection=redis_conn) if redis_conn else None
default_queue = Queue('default', connection=redis_conn) if redis_conn else None

# Directories - use absolute paths to avoid issues with Flask's send_file
AUDIO_DIR = Path("/app/audio_files")
AUDIO_DIR.mkdir(exist_ok=True)
TEMP_DIR = Path("/app/temp_images")
TEMP_DIR.mkdir(exist_ok=True)


def cleanup_old_files():
    """Remove files older than 1 hour"""
    cutoff = datetime.now() - timedelta(hours=1)
    for directory in [AUDIO_DIR, TEMP_DIR]:
        for file in directory.glob("*"):
            if datetime.fromtimestamp(file.stat().st_mtime) < cutoff:
                file.unlink()


# HTML Template (same as before, with minor updates)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Comic Reader - Distributed Architecture</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: #333;
      padding: 20px;
      min-height: 100vh;
    }
    .container {
      max-width: 1200px;
      margin: 30px auto;
      background: white;
      padding: 30px;
      border-radius: 16px;
      box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    }
    h1 {
      color: #667eea;
      margin-bottom: 10px;
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }
    .badge {
      background: linear-gradient(135deg, #667eea, #764ba2);
      color: white;
      padding: 4px 12px;
      border-radius: 20px;
      font-size: 12px;
    }
    .badge.new {
      background: linear-gradient(135deg, #10b981, #059669);
      animation: pulse 2s infinite;
    }
    @keyframes pulse {
      0%, 100% { transform: scale(1); }
      50% { transform: scale(1.05); }
    }
    .subtitle {
      color: #666;
      margin-bottom: 25px;
    }
    .upload-area {
      border: 3px dashed #667eea;
      border-radius: 12px;
      padding: 50px 20px;
      margin: 20px 0;
      cursor: pointer;
      transition: all 0.3s;
      background: #f7fafc;
      text-align: center;
    }
    .upload-area:hover {
      background: #ebf8ff;
      border-color: #764ba2;
      transform: scale(1.01);
    }
    .upload-icon { font-size: 48px; margin-bottom: 15px; }
    .button {
      background: linear-gradient(135deg, #667eea, #764ba2);
      color: white;
      border: none;
      padding: 14px 28px;
      border-radius: 8px;
      cursor: pointer;
      font-size: 16px;
      font-weight: 600;
      transition: all 0.3s;
      box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    .button:hover {
      transform: translateY(-2px);
      box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    .button:disabled {
      background: #ccc;
      cursor: not-allowed;
      box-shadow: none;
    }
    .result-box {
      background: #f7fafc;
      border-radius: 12px;
      padding: 25px;
      margin: 20px 0;
      border-left: 5px solid #667eea;
    }
    .result-box h3 {
      color: #667eea;
      margin-bottom: 15px;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    textarea {
      width: 100%;
      padding: 15px;
      border: 2px solid #e2e8f0;
      border-radius: 8px;
      font-size: 15px;
      font-family: inherit;
      resize: vertical;
      min-height: 150px;
    }
    .stats {
      display: flex;
      gap: 20px;
      flex-wrap: wrap;
      margin: 20px 0;
    }
    .stat-card {
      background: white;
      padding: 15px 20px;
      border-radius: 10px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.1);
      flex: 1;
      min-width: 150px;
    }
    .stat-label {
      color: #666;
      font-size: 13px;
      margin-bottom: 5px;
    }
    .stat-value {
      color: #667eea;
      font-size: 24px;
      font-weight: bold;
    }
    audio {
      width: 100%;
      margin: 15px 0;
    }
    .controls {
      display: flex;
      gap: 15px;
      margin: 20px 0;
      flex-wrap: wrap;
    }
    .voice-select {
      flex: 1;
      min-width: 200px;
      padding: 12px;
      border: 2px solid #e2e8f0;
      border-radius: 8px;
      font-size: 15px;
    }
    .spinner {
      border: 4px solid #f3f3f3;
      border-top: 4px solid #667eea;
      border-radius: 50%;
      width: 40px;
      height: 40px;
      animation: spin 1s linear infinite;
      margin: 20px auto;
    }
    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }
    .checkbox-group {
      display: flex;
      align-items: center;
      gap: 10px;
      margin: 15px 0;
    }
    .status-badge {
      display: inline-block;
      padding: 4px 8px;
      border-radius: 4px;
      font-size: 12px;
      font-weight: bold;
      margin-left: 10px;
    }
    .status-queued { background: #fbbf24; color: #78350f; }
    .status-processing { background: #3b82f6; color: white; }
    .status-completed { background: #10b981; color: white; }
    .status-failed { background: #ef4444; color: white; }
    .upload-preview {
      display: none;
      gap: 16px;
      align-items: center;
      justify-content: center;
      margin-top: 10px;
      color: #2d3748;
      flex-wrap: wrap;
    }
    .upload-preview img {
      max-width: 260px;
      max-height: 260px;
      border-radius: 10px;
      object-fit: contain;
      border: 2px solid #e2e8f0;
      background: white;
      box-shadow: 0 10px 30px rgba(0,0,0,0.08);
    }
    .upload-preview .upload-preview__details {
      text-align: left;
      max-width: 320px;
    }
    .upload-preview__details h3 {
      margin-bottom: 8px;
      color: #1f2937;
    }
    .upload-preview__details p {
      color: #4b5563;
      line-height: 1.4;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>
      🎨 Comic Reader
      <span class="badge">V3.0</span>
      <span class="badge new">⚡ Distributed</span>
    </h1>
    <p class="subtitle">Upload your comic page and let AI read it to you!</p>

    <div class="upload-area" id="uploadArea">
      <div id="uploadPrompt">
        <div class="upload-icon">📤</div>
        <h3>Click to upload or drag & drop</h3>
        <p>Supports: JPG, PNG, GIF, WebP</p>
      </div>
      <div class="upload-preview" id="uploadPreview">
        <img id="previewImage" alt="Comic preview" />
        <div class="upload-preview__details">
          <h3 id="previewName"></h3>
          <p id="previewInfo">Ready to process!</p>
        </div>
      </div>
      <input type="file" id="fileInput" accept="image/*" style="display: none;" />
    </div>

    <div class="checkbox-group">
      <input type="checkbox" id="preprocessToggle" checked />
      <label for="preprocessToggle">🔧 Enable image preprocessing (recommended for better accuracy)</label>
    </div>

    <div class="checkbox-group">
      <input type="checkbox" id="translateToggle" />
      <label for="translateToggle">🌐 Translate to Dutch (Nederlands)</label>
    </div>

    <div class="controls">
      <select class="voice-select" id="voiceSelect">
        <option value="en-US-Neural2-F" data-lang="en-US">🎭 Female Voice 1 (US)</option>
        <option value="en-US-Neural2-C" data-lang="en-US">🎭 Female Voice 2 (US)</option>
        <option value="en-US-Neural2-E" data-lang="en-US">🎭 Female Voice 3 (US)</option>
        <option value="en-US-Neural2-D" data-lang="en-US">👨 Male Voice 1 (US)</option>
        <option value="en-US-Neural2-A" data-lang="en-US">👨 Male Voice 2 (US)</option>
        <option value="en-US-Neural2-I" data-lang="en-US">👦 Child Voice (US)</option>
        <option value="en-GB-Neural2-A" data-lang="en-GB">🇬🇧 British Female</option>
        <option value="en-GB-Neural2-B" data-lang="en-GB">🇬🇧 British Male</option>
        <option value="en-AU-Neural2-A" data-lang="en-AU">🇦🇺 Australian Female</option>
        <option value="en-AU-Neural2-B" data-lang="en-AU">🇦🇺 Australian Male</option>
        <option value="nl-NL-Standard-A" data-lang="nl-NL">🇳🇱 Dutch Female</option>
        <option value="nl-NL-Standard-B" data-lang="nl-NL">🇳🇱 Dutch Male</option>
        <option value="nl-NL-Wavenet-A" data-lang="nl-NL">🇳🇱 Dutch Female (Premium)</option>
        <option value="nl-NL-Wavenet-B" data-lang="nl-NL">🇳🇱 Dutch Male (Premium)</option>
      </select>
      <button class="button" id="processBtn" disabled>🚀 Process Comic</button>
    </div>

    <div id="jobStatus" style="display: none;">
      <div class="result-box">
        <h3>📋 Job Status <span class="status-badge" id="statusBadge">QUEUED</span></h3>
        <p id="statusText">Your task has been queued for processing...</p>
      </div>
    </div>

    <div id="results" style="display: none;">
      <div class="stats">
        <div class="stat-card">
          <div class="stat-label">📊 Panels Detected</div>
          <div class="stat-value" id="panelCount">0</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">💬 Bubbles Found</div>
          <div class="stat-value" id="bubbleCount">0</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">✅ Confidence</div>
          <div class="stat-value" id="confidence">0%</div>
        </div>
      </div>

      <div class="result-box">
        <h3>📝 Extracted Text (Original)</h3>
        <textarea id="extractedText" placeholder="Extracted text will appear here..."></textarea>
      </div>

      <div class="result-box" id="translatedTextBox" style="display: none;">
        <h3>🌐 Translated Text (Dutch)</h3>
        <textarea id="translatedText" placeholder="Translated text will appear here..."></textarea>
      </div>

      <div class="result-box">
        <h3>🔊 Generated Audio</h3>
        <audio id="audioPlayer" controls></audio>
      </div>
    </div>

    <div id="loadingDiv" style="display: none;">
      <div class="spinner"></div>
      <p style="text-align: center; color: #667eea; margin-top: 10px;">
        Processing your comic using distributed workers...
      </p>
    </div>
  </div>

  <script>
    let selectedFile = null;
    let pollInterval = null;

    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const processBtn = document.getElementById('processBtn');
    const voiceSelect = document.getElementById('voiceSelect');
    const preprocessToggle = document.getElementById('preprocessToggle');
    const translateToggle = document.getElementById('translateToggle');
    const resultsDiv = document.getElementById('results');
    const loadingDiv = document.getElementById('loadingDiv');
    const jobStatusDiv = document.getElementById('jobStatus');
    const uploadPrompt = document.getElementById('uploadPrompt');
    const uploadPreview = document.getElementById('uploadPreview');
    const previewImage = document.getElementById('previewImage');
    const previewName = document.getElementById('previewName');

    // Auto-select Dutch voice when translation is enabled
    translateToggle.addEventListener('change', function() {
      if (this.checked) {
        // Switch to Dutch voice
        voiceSelect.value = 'nl-NL-Standard-A';
      } else {
        // Switch back to English voice
        voiceSelect.value = 'en-US-Neural2-F';
      }
    });

    uploadArea.addEventListener('click', () => fileInput.click());

    uploadArea.addEventListener('dragover', (e) => {
      e.preventDefault();
      uploadArea.style.background = '#ebf8ff';
    });

    uploadArea.addEventListener('dragleave', () => {
      uploadArea.style.background = '#f7fafc';
    });

    uploadArea.addEventListener('drop', (e) => {
      e.preventDefault();
      uploadArea.style.background = '#f7fafc';
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith('image/')) {
        handleFileSelect(file);
      }
    });

    fileInput.addEventListener('change', (e) => {
      const file = e.target.files[0];
      if (file) {
        handleFileSelect(file);
      }
    });

    function handleFileSelect(file) {
      selectedFile = file;
      const reader = new FileReader();
      reader.onload = (event) => {
        previewImage.src = event.target.result;
        previewName.textContent = `File selected: ${file.name}`;
        uploadPreview.style.display = 'flex';
        uploadPrompt.style.display = 'none';
      };
      reader.readAsDataURL(file);
      processBtn.disabled = false;
    }

    processBtn.addEventListener('click', async () => {
      if (!selectedFile) return;

      resultsDiv.style.display = 'none';
      jobStatusDiv.style.display = 'block';
      loadingDiv.style.display = 'block';
      processBtn.disabled = true;

      const formData = new FormData();
      formData.append('image', selectedFile);
      formData.append('voice_name', voiceSelect.value);

      // Get language code from selected voice
      const selectedOption = voiceSelect.options[voiceSelect.selectedIndex];
      const languageCode = selectedOption.getAttribute('data-lang');
      formData.append('language_code', languageCode);

      formData.append('preprocess', preprocessToggle.checked);
      formData.append('translate', translateToggle.checked);
      formData.append('target_language', 'nl');  // Dutch

      try {
        // Submit job
        const response = await fetch('/api/process-comic', {
          method: 'POST',
          body: formData
        });

        const data = await response.json();

        if (data.success && data.job_id) {
          // Start polling for job status
          updateStatus('QUEUED', 'Your task has been queued for processing...');
          pollJobStatus(data.job_id);
        } else {
          alert('Error: ' + (data.error || 'Unknown error'));
          loadingDiv.style.display = 'none';
          jobStatusDiv.style.display = 'none';
          processBtn.disabled = false;
        }
      } catch (error) {
        alert('Error submitting job: ' + error.message);
        loadingDiv.style.display = 'none';
        jobStatusDiv.style.display = 'none';
        processBtn.disabled = false;
      }
    });

    function updateStatus(status, message) {
      const statusBadge = document.getElementById('statusBadge');
      const statusText = document.getElementById('statusText');

      statusBadge.textContent = status;
      statusBadge.className = 'status-badge status-' + status.toLowerCase();
      statusText.textContent = message;
    }

    async function pollJobStatus(jobId) {
      pollInterval = setInterval(async () => {
        try {
          const response = await fetch(`/api/job-status/${jobId}`);
          const data = await response.json();

          if (data.status === 'finished') {
            clearInterval(pollInterval);

            if (data.result && data.result.success) {
              const result = data.result;
              document.getElementById('panelCount').textContent = result.panel_count || 0;
              document.getElementById('bubbleCount').textContent = result.bubble_count || 0;
              document.getElementById('confidence').textContent =
                Math.round((result.confidence || 0) * 100) + '%';
              document.getElementById('extractedText').value = result.extracted_text || '';
              document.getElementById('audioPlayer').src = result.audio_url || '';

              // Show translated text if available
              if (result.translated_text) {
                document.getElementById('translatedText').value = result.translated_text;
                document.getElementById('translatedTextBox').style.display = 'block';
              } else {
                document.getElementById('translatedTextBox').style.display = 'none';
              }

              updateStatus('COMPLETED', 'Processing completed successfully!');
              loadingDiv.style.display = 'none';
              resultsDiv.style.display = 'block';
              processBtn.disabled = false;
            } else {
              updateStatus('FAILED', 'Error: ' + (data.result?.error || 'Unknown error'));
              loadingDiv.style.display = 'none';
              processBtn.disabled = false;
            }
          } else if (data.status === 'failed') {
            clearInterval(pollInterval);
            updateStatus('FAILED', 'Job failed: ' + (data.exc_info || 'Unknown error'));
            loadingDiv.style.display = 'none';
            processBtn.disabled = false;
          } else if (data.status === 'started') {
            updateStatus('PROCESSING', 'Worker is processing your comic...');
          } else {
            updateStatus('QUEUED', 'Waiting for available worker...');
          }
        } catch (error) {
          console.error('Error polling job status:', error);
        }
      }, 1000); // Poll every second
    }
  </script>
</body>
</html>
"""


# Routes
@app.route('/')
def index():
    """Serve the frontend"""
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    queue_status = "connected" if redis_conn else "disconnected"
    return jsonify({
        "status": "healthy",
        "queue": queue_status,
        "service": "interface_server"
    })


@app.route('/api/process-comic', methods=['POST'])
def process_comic():
    """
    Enqueue comic processing task
    This does NOT process the task directly - it sends it to a worker
    """
    try:
        if not redis_conn or not default_queue:
            return jsonify({
                "success": False,
                "error": "Queue service not available. Please ensure Redis is running."
            }), 503

        cleanup_old_files()

        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files['image']
        language_code = request.form.get('language_code', 'en-US')
        voice_name = request.form.get('voice_name', 'en-US-Neural2-F')
        preprocess = request.form.get('preprocess', 'true').lower() == 'true'
        translate = request.form.get('translate', 'false').lower() == 'true'
        target_language = request.form.get('target_language', 'nl')

        # Read image bytes
        image_bytes = file.read()

        # Save to temp file for worker access
        temp_id = str(uuid.uuid4())
        temp_path = TEMP_DIR / f"{temp_id}.png"
        with open(temp_path, 'wb') as f:
            f.write(image_bytes)

        # Enqueue job (send to worker)
        print(f"[INTERFACE] Enqueueing job: temp_id={temp_id}, translate={translate}")

        job = default_queue.enqueue(
            'tasks.process_comic_full_pipeline',
            image_bytes=image_bytes,
            language_code=language_code,
            voice_name=voice_name,
            preprocess=preprocess,
            translate=translate,
            target_language=target_language,
            job_timeout='10m'  # 10 minute timeout
        )

        return jsonify({
            "success": True,
            "job_id": job.id,
            "status": "queued",
            "message": "Task has been queued for processing"
        })

    except Exception as e:
        print(f"[INTERFACE] Error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/job-status/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """
    Get status of a queued job
    """
    try:
        if not redis_conn:
            return jsonify({"error": "Queue service not available"}), 503

        job = Job.fetch(job_id, connection=redis_conn)

        response = {
            "job_id": job.id,
            "status": job.get_status(),
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "ended_at": job.ended_at.isoformat() if job.ended_at else None,
        }

        # Include result if job is finished
        if job.is_finished:
            response["result"] = job.result
        elif job.is_failed:
            response["exc_info"] = job.exc_info

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/audio/<audio_id>', methods=['GET'])
def get_audio(audio_id):
    """Serve generated audio file"""
    try:
        audio_path = AUDIO_DIR / f"{audio_id}.mp3"

        if not audio_path.exists():
            return jsonify({"error": "Audio file not found"}), 404

        return send_file(
            audio_path,
            mimetype='audio/mpeg',
            as_attachment=False
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🌐 INTERFACE SERVER - Distributed Comic Reader")
    print("="*70)

    if redis_conn:
        print(f"✓ Connected to Redis queue")
        print(f"✓ Workers will process AI tasks")
    else:
        print(f"⚠️  WARNING: Redis not connected!")
        print(f"   Please start Redis: brew install redis && brew services start redis")

    print(f"\n📍 Interface Server: http://localhost:5001")
    print(f"")
    print(f"Architecture:")
    print(f"  1. This server (Interface) - Handles user requests")
    print(f"  2. Redis (Queue/Orchestrator) - Manages task distribution")
    print(f"  3. Workers (AI Processing) - Process OCR and TTS tasks")
    print("="*70 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5001)
