"""
LLM-based Comic Narrator using OpenAI GPT-4 Vision API
Generates audiobook-style narration from comic images while preserving dialogue
"""

import os
import base64
import logging
from typing import Dict, List, Optional, Any
from openai import OpenAI
from io import BytesIO
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComicNarrator:
    """
    Uses OpenAI GPT-4 Vision to generate audiobook-style narration from comic panels.
    Preserves original dialogue while adding narrative context.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        """
        Initialize the Comic Narrator.

        Args:
            api_key: OpenAI API key. If None, will try to get from environment.
            model: OpenAI model to use (default: gpt-4o which has vision capabilities)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable.")

        self.model = model
        self.client = OpenAI(api_key=self.api_key)
        logger.info(f"ComicNarrator initialized with model: {model}")

    def _encode_image_to_base64(self, image_bytes: bytes) -> str:
        """
        Encode image bytes to base64 string for OpenAI API.

        Args:
            image_bytes: Raw image bytes

        Returns:
            Base64 encoded string
        """
        return base64.b64encode(image_bytes).decode('utf-8')

    def _create_narration_prompt(self, panel_number: Optional[int] = None, total_panels: Optional[int] = None) -> str:
        """
        Create the system prompt for audiobook-style narration.

        Args:
            panel_number: Current panel number (1-indexed)
            total_panels: Total number of panels

        Returns:
            Formatted prompt string
        """
        panel_context = ""
        if panel_number and total_panels:
            panel_context = f"This is panel {panel_number} of {total_panels}. "

        prompt = f"""You are a professional audiobook narrator for comic books. Your task is to create an engaging, dramatic narration that brings this comic panel to life.

{panel_context}Your narration should:

1. **PRESERVE EXACT DIALOGUE**: Extract and include the actual text from speech bubbles word-for-word in quotation marks.
2. **ADD NARRATIVE CONTEXT**: Describe the scene, setting, character actions, and basic expressions.
3. **AUDIOBOOK STYLE**: Write as if reading aloud - engaging, dramatic, immersive.
4. **BASIC VISUAL ANALYSIS**: Describe what's clearly visible (characters, actions, scene composition) without deep interpretation.
5. **FLOW AND PACING**: Create smooth transitions between narrative and dialogue.

Format your response as a flowing narrative suitable for text-to-speech, like this example:

"The hero stands at the edge of a crumbling building, wind whipping through their cape. They look down at the city below with determination in their eyes. 'I won't let them down,' they say firmly. In the distance, smoke rises from the chaos."

IMPORTANT:
- Include ALL visible text from speech bubbles as direct quotes
- Use narrative descriptions to set scenes and describe actions
- Keep narration concise but vivid
- Make it sound natural when spoken aloud
- If there are multiple characters speaking, indicate who is speaking through narrative context

Now, analyze this comic panel and create the audiobook narration:"""

        return prompt

    def narrate_panel(
        self,
        image_bytes: bytes,
        panel_number: Optional[int] = None,
        total_panels: Optional[int] = None,
        max_tokens: int = 500
    ) -> Dict[str, Any]:
        """
        Generate audiobook-style narration for a single comic panel.

        Args:
            image_bytes: Raw bytes of the panel image
            panel_number: Current panel number (1-indexed)
            total_panels: Total number of panels in the comic
            max_tokens: Maximum tokens for the response

        Returns:
            Dict containing:
                - narration: The generated narrative text
                - success: Boolean indicating success
                - error: Error message if failed
                - raw_response: Full API response
        """
        try:
            # Encode image to base64
            base64_image = self._encode_image_to_base64(image_bytes)

            # Create the prompt
            system_prompt = self._create_narration_prompt(panel_number, total_panels)

            # Call OpenAI API
            logger.info(f"Generating narration for panel {panel_number or 'unknown'}")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": system_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=max_tokens,
                temperature=0.7  # Slight creativity for engaging narration
            )

            # Extract narration
            narration = response.choices[0].message.content.strip()

            logger.info(f"Successfully generated narration ({len(narration)} chars)")

            return {
                "narration": narration,
                "success": True,
                "error": None,
                "raw_response": response,
                "tokens_used": response.usage.total_tokens if response.usage else None
            }

        except Exception as e:
            logger.error(f"Error generating narration: {str(e)}")
            return {
                "narration": "",
                "success": False,
                "error": str(e),
                "raw_response": None,
                "tokens_used": None
            }

    def narrate_comic(
        self,
        panel_images: List[bytes],
        combine_narration: bool = True
    ) -> Dict[str, Any]:
        """
        Generate audiobook-style narration for multiple comic panels.

        Args:
            panel_images: List of panel images as bytes
            combine_narration: If True, combine all panel narrations into one text

        Returns:
            Dict containing:
                - full_narration: Combined narration text (if combine_narration=True)
                - panels: List of individual panel narrations
                - success: Boolean indicating overall success
                - total_tokens: Total tokens used across all panels
        """
        total_panels = len(panel_images)
        panel_results = []
        total_tokens = 0

        logger.info(f"Generating narration for {total_panels} panels")

        for i, panel_bytes in enumerate(panel_images, 1):
            result = self.narrate_panel(
                panel_bytes,
                panel_number=i,
                total_panels=total_panels
            )
            panel_results.append(result)

            if result.get('tokens_used'):
                total_tokens += result['tokens_used']

        # Check if all panels succeeded
        all_success = all(r['success'] for r in panel_results)

        # Combine narrations if requested
        full_narration = ""
        if combine_narration:
            narrations = [r['narration'] for r in panel_results if r['success']]
            full_narration = "\n\n".join(narrations)

        return {
            "full_narration": full_narration,
            "panels": panel_results,
            "success": all_success,
            "total_tokens": total_tokens,
            "panel_count": total_panels
        }

    def narrate_single_image(
        self,
        image_bytes: bytes,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """
        Generate narration for a single comic image (may contain multiple panels).
        The LLM will analyze the entire image and create a cohesive narration.

        Args:
            image_bytes: Raw bytes of the full comic image
            max_tokens: Maximum tokens for the response

        Returns:
            Dict containing narration and metadata
        """
        try:
            base64_image = self._encode_image_to_base64(image_bytes)

            # Modified prompt for full comic page - optimized for TTS output
            prompt = """You are an accessibility assistant transforming a comic book page into a vivid audiobook scene.

You are looking at a single page from a comic book.
Your job is to produce a cinematic narration, as if describing a scene from a high-quality audiobook or radio drama.

Follow these rules:

1) **Read all text exactly as it appears** - Include ALL dialogue from speech balloons, narration boxes, signs, and sound effects word-for-word in quotation marks.

2) **Identify speakers clearly** - When possible, identify who is speaking (e.g., 'The grandmother asks:', 'She responds:', use character descriptions if names aren't visible).

3) **Describe visual action cinematically** using dynamic, sensory language:
   - Describe character movement, positioning, and interactions
   - Mention facial expressions, body language, and emotional states
   - Describe the environment, lighting, and atmosphere
   - Note visual details that add context to the dialogue

4) **Follow the natural comic reading order** - Top-left to bottom-right, panel by panel. Create smooth transitions between panels.

5) **Blend dialogue and visuals into smooth narration** - Don't separate them. Flow naturally from description to dialogue and back.

6) **Avoid formatting symbols** - No bullet points, asterisks, markdown. Use flowing prose only.

7) **Write for speech synthesis** - Use proper punctuation for natural pauses. Write numbers as words. Use clear, conversational language that sounds natural when spoken aloud.

**Tone & Style:**
- Use the voice of a professional audiobook narrator
- Keep it immersive and vivid, but clear and accessible
- Prioritize clarity for listeners
- Vary sentence structure for natural rhythm
- Use transitional phrases between panels

**IMPORTANT OUTPUT FORMAT:**
Write one continuous cinematic narration describing the page as if it is a movie scene unfolding in real time. Make it engaging and natural for text-to-speech conversion.

EXAMPLE OUTPUT:
"In a cozy living room, a grandmother sits on a sofa beside her granddaughter. The older woman holds a decorative fan, her eyes bright with curiosity as she turns to the young woman. 'So what's the United States like?' she asks. Her granddaughter smiles warmly, relaxing into the cushions. 'Well, it's very different from China, Grandma,' she replies. The grandmother leans forward eagerly. 'Tell me about their festivals!' she says with excitement. 'For example, they have Thanksgiving,' the granddaughter begins to explain. The grandmother tilts her head. 'What do they do?' she asks. 'In short, family getting together, eating lots of food,' the granddaughter says with a knowing smile. The grandmother's face lights up with recognition and delight. 'Oh, it's like the Spring Festival! We are the same!' she exclaims happily. The granddaughter grins at her grandmother's enthusiasm. 'You win,' she says warmly."

Now analyze this comic page and create the audiobook narration:"""

            logger.info("Generating narration for full comic image")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )

            narration = response.choices[0].message.content.strip()

            logger.info(f"Successfully generated full narration ({len(narration)} chars)")

            return {
                "text": narration,  # Using 'text' key for compatibility with existing pipeline
                "narration": narration,
                "success": True,
                "error": None,
                "tokens_used": response.usage.total_tokens if response.usage else None,
                "confidence": 1.0  # LLM-based, assumed high confidence
            }

        except Exception as e:
            logger.error(f"Error generating full narration: {str(e)}")
            return {
                "text": "",
                "narration": "",
                "success": False,
                "error": str(e),
                "tokens_used": None,
                "confidence": 0.0
            }


# Singleton instance for reuse
_narrator_instance: Optional[ComicNarrator] = None


def get_comic_narrator(api_key: Optional[str] = None) -> ComicNarrator:
    """
    Get or create a singleton ComicNarrator instance.

    Args:
        api_key: OpenAI API key (optional)

    Returns:
        ComicNarrator instance
    """
    global _narrator_instance

    if _narrator_instance is None:
        _narrator_instance = ComicNarrator(api_key=api_key)

    return _narrator_instance




def narrate_panel(image_bytes, panel_number=None, total_panels=None, max_tokens=500):
    """
    Compatibility wrapper for unit tests.
    Calls ComicNarrator().narrate_panel(...)
    """
    narrator = get_comic_narrator()
    return narrator.narrate_panel(
        image_bytes=image_bytes,
        panel_number=panel_number,
        total_panels=total_panels,
        max_tokens=max_tokens
    )

