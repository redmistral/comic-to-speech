#!/usr/bin/env python3
"""
Task Definitions for Distributed Comic-to-Speech Processing

This module defines the AI processing tasks executed by worker processes.
Workers consume jobs from Redis queues and execute these functions.

Available Tasks:
    - process_ocr_task(): Extract text from comic images using LLM or OCR
    - process_translation_task(): Translate text EN→NL using OpenNMT model
    - process_tts_task(): Generate speech audio from text using Google Cloud TTS
    - process_comic_full_pipeline(): Complete end-to-end pipeline (all above)

Task Flow:
    1. Interface server enqueues job to Redis
    2. Worker picks up job from queue
    3. Worker calls task function defined here
    4. Task processes image/text using AI APIs
    5. Result stored in Redis and returned to interface server

Each task is designed to be idempotent and can handle failures gracefully.
Workers import this module to execute the actual AI processing logic.
"""

from google.cloud import texttospeech
import uuid
import logging

# Import the OCR processing logic from the narration module
from narration.vision_ocr import (
    ComicOCR,
    setup_credentials,
    AUDIO_DIR,
    TEMP_DIR,
    get_tts_client
)

# Import translation module
from translation.translator import translate_text, is_translation_available

# Ensure credentials are set up
setup_credentials()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_ocr_task(image_bytes, preprocess=True):
    """
    Task: Extract text from comic image using OCR

    Args:
        image_bytes: Raw image data
        preprocess: Whether to preprocess the image

    Returns:
        dict: OCR results including text, panels, bubbles, confidence
    """
    print(f"[WORKER] Processing OCR task (preprocess={preprocess})")

    try:
        ocr = ComicOCR()
        result = ocr.extract_text(image_bytes, preprocess=preprocess)

        narration_mode = result.get('narration_mode', 'ocr')
        print(f"[WORKER] Text extraction completed using {narration_mode.upper()}: "
              f"{result['panel_count']} panels, "
              f"{result['bubble_count']} bubbles, "
              f"confidence={result['confidence']:.2f}")

        return {
            "success": True,
            "extracted_text": result["text"],
            "panel_count": result["panel_count"],
            "bubble_count": result["bubble_count"],
            "text_blocks": len(result["text_blocks"]),
            "confidence": result["confidence"],
            "narration_mode": narration_mode,
            "tokens_used": result.get("tokens_used")
        }
    except Exception as e:
        print(f"[WORKER] OCR error: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }


def process_translation_task(text, src_lang="en", tgt_lang="nl"):
    """
    Task: Translate text from English to Dutch (or other target language)

    Args:
        text: Text to translate
        src_lang: Source language code (default: "en")
        tgt_lang: Target language code (default: "nl" for Dutch)

    Returns:
        dict: Translation result and metadata
    """
    # Validate input before logging (to avoid len(None) error)
    if not text:
        return {
            "success": False,
            "error": "No text provided for translation"
        }

    logger.info(f"[WORKER] Processing translation task ({len(text)} chars, {src_lang} -> {tgt_lang})")

    try:
        # Check if translation is available
        if not is_translation_available():
            return {
                "success": False,
                "error": "Translation model not available"
            }

        # Translate the text
        translated = translate_text(text, src_lang=src_lang, tgt_lang=tgt_lang)

        logger.info(f"[WORKER] Translation completed: {len(translated)} chars")

        return {
            "success": True,
            "translated_text": translated,
            "original_text": text,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang
        }
    except Exception as e:
        logger.error(f"[WORKER] Translation error: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "original_text": text  # Return original on error
        }


def process_tts_task(text, language_code='en-US', voice_name='en-US-Neural2-F'):
    """
    Task: Generate speech audio from text using TTS

    Args:
        text: Text to convert to speech
        language_code: Language code (e.g., 'en-US', 'nl-NL')
        voice_name: Voice name (e.g., 'en-US-Neural2-F', 'nl-NL-Standard-A')

    Returns:
        dict: Audio file ID and metadata
    """
    # Validate input before logging (to avoid len(None) error)
    if not text:
        return {
            "success": False,
            "error": "No text provided"
        }

    print(f"[WORKER] Processing TTS task ({len(text)} chars, voice={voice_name})")

    try:
        tts_client = get_tts_client()
    except Exception as exc:
        return {
            "success": False,
            "error": f"TTS client not initialized: {exc}"
        }

    try:
        # Generate audio
        synthesis_input = texttospeech.SynthesisInput(text=text)

        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name=voice_name
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=1.0,
            pitch=0.0
        )

        response = tts_client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )

        # Save audio file
        audio_id = str(uuid.uuid4())
        audio_path = AUDIO_DIR / f"{audio_id}.mp3"

        with open(audio_path, 'wb') as out:
            out.write(response.audio_content)

        print(f"[WORKER] TTS completed: audio_id={audio_id}")

        return {
            "success": True,
            "audio_id": audio_id,
            "audio_url": f"/api/audio/{audio_id}",
            "characters_used": len(text)
        }
    except Exception as e:
        print(f"[WORKER] TTS error: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }


def process_comic_full_pipeline(image_bytes, language_code='en-US',
                                voice_name='en-US-Neural2-F', preprocess=True,
                                translate=False, target_language="nl"):
    """
    Task: Complete pipeline - OCR + (Optional Translation) + TTS

    This combines OCR, optional translation, and TTS into a single task.
    Useful for simpler orchestration.

    Args:
        image_bytes: Raw image data
        language_code: Language for TTS (e.g., 'en-US', 'nl-NL')
        voice_name: Voice for TTS (e.g., 'en-US-Neural2-F', 'nl-NL-Standard-A')
        preprocess: Whether to preprocess image
        translate: Whether to translate text before TTS (default: False)
        target_language: Target language for translation (default: "nl" for Dutch)

    Returns:
        dict: Combined results from OCR, translation (if enabled), and TTS
    """
    logger.info(f"[WORKER] Processing full pipeline task (translate={translate})")

    # Step 1: OCR - Extract text from comic
    ocr_result = process_ocr_task(image_bytes, preprocess)

    if not ocr_result["success"]:
        return ocr_result

    extracted_text = ocr_result["extracted_text"]

    if not extracted_text:
        return {
            "success": False,
            "error": "No text extracted from image"
        }

    # Step 2: Translation (optional)
    text_for_tts = extracted_text
    translated_text = None
    translation_error = None

    if translate:
        logger.info(f"[WORKER] Translation enabled, translating to {target_language}")
        translation_result = process_translation_task(
            extracted_text,
            src_lang="en",
            tgt_lang=target_language
        )

        if translation_result["success"]:
            text_for_tts = translation_result["translated_text"]
            translated_text = translation_result["translated_text"]
            logger.info("[WORKER] Translation successful")
        else:
            # If translation fails, continue with original text
            translation_error = translation_result.get("error", "Translation failed")
            logger.warning(f"[WORKER] Translation failed: {translation_error}, using original text")

    # Step 3: TTS - Convert text to speech
    tts_result = process_tts_task(text_for_tts, language_code, voice_name)

    if not tts_result["success"]:
        return tts_result

    # Combine results
    result = {
        "success": True,
        "extracted_text": extracted_text,
        "panel_count": ocr_result["panel_count"],
        "bubble_count": ocr_result["bubble_count"],
        "text_blocks": ocr_result["text_blocks"],
        "confidence": ocr_result["confidence"],
        "narration_mode": ocr_result.get("narration_mode", "ocr"),
        "tokens_used": ocr_result.get("tokens_used"),
        "audio_id": tts_result["audio_id"],
        "audio_url": tts_result["audio_url"],
        "characters_used": tts_result["characters_used"]
    }

    # Add translation info if translation was performed
    if translate:
        result["translation_enabled"] = True
        result["target_language"] = target_language
        if translated_text:
            result["translated_text"] = translated_text
        if translation_error:
            result["translation_error"] = translation_error

    return result
