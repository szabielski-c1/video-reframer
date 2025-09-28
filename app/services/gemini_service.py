import google.generativeai as genai
import base64
import json
import asyncio
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image
import io
import logging
import time

from app.config import settings

logger = logging.getLogger(__name__)

class GeminiService:
    """Service for interacting with Google Gemini API for video analysis"""

    def __init__(self):
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(settings.GEMINI_MODEL)
        self.vision_model = genai.GenerativeModel(settings.GEMINI_MODEL)

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum seconds between requests

        # Retry configuration
        self.max_retries = 3
        self.retry_delay = 2.0

    async def analyze_frames(self, frames: List[np.ndarray], prompt: str) -> List[Dict]:
        """Analyze multiple frames with Gemini Vision"""

        if not frames:
            return []

        try:
            # Convert frames to base64
            frame_data = []
            for i, frame in enumerate(frames):
                img_base64 = await self.numpy_to_base64(frame)
                frame_data.append({
                    'index': i,
                    'data': img_base64
                })

            # Build multimodal prompt
            response = await self.call_gemini_with_retry(frame_data, prompt)

            # Parse JSON response
            results = self.parse_analysis_response(response, len(frames))

            return results

        except Exception as e:
            logger.error(f"Gemini frame analysis error: {str(e)}")
            # Return fallback results
            return [self.create_fallback_result() for _ in frames]

    async def classify_scene(self, frames: List[np.ndarray]) -> Dict:
        """Classify the overall scene type from frames"""

        prompt = """
        Analyze these video frames to classify the content type and scene characteristics.

        Return a JSON object with:
        {
            "type": "interview|vlog|sports|presentation|music|documentary|auto",
            "description": "Brief description of what's happening",
            "key_elements": ["list", "of", "important", "visual", "elements"],
            "subject_count": number_of_main_subjects,
            "motion_level": "low|medium|high",
            "recommended_strategy": "Suggested reframing approach",
            "confidence": 0.0-1.0
        }

        Consider:
        - Number and types of people/subjects
        - Movement patterns and activity level
        - Setting and environment
        - Production style and quality
        - Text or graphics present
        """

        try:
            # Use first 5 frames for classification
            sample_frames = frames[:5]
            frame_data = []

            for frame in sample_frames:
                img_base64 = await self.numpy_to_base64(frame)
                frame_data.append({'data': img_base64})

            response = await self.call_gemini_with_retry(frame_data, prompt, is_classification=True)

            # Parse classification response
            classification = self.parse_classification_response(response)

            logger.info(f"Scene classified as: {classification.get('type', 'unknown')}")
            return classification

        except Exception as e:
            logger.error(f"Scene classification error: {str(e)}")
            return {
                'type': 'auto',
                'description': 'Classification failed',
                'key_elements': [],
                'subject_count': 1,
                'motion_level': 'medium',
                'recommended_strategy': 'Default center tracking',
                'confidence': 0.1
            }

    async def detect_speakers(self, frames: List[np.ndarray], audio_peaks: Optional[List[float]] = None) -> List[str]:
        """Detect who is speaking in frames based on visual cues"""

        prompt = """
        Identify who appears to be speaking in these video frames.

        Look for:
        - Mouth movement and lip motion
        - Body language indicating speech
        - Facial expressions while talking
        - Hand gestures accompanying speech
        - Eye contact with camera/audience

        Return a JSON array with subject IDs who are actively speaking:
        ["subject_id_1", "subject_id_2"]

        If no clear speakers detected, return empty array: []
        """

        if audio_peaks:
            prompt += f"\n\nAudio activity levels: {audio_peaks}"
            prompt += "\nUse audio levels to help identify speaking patterns."

        try:
            frame_data = []
            for frame in frames:
                img_base64 = await self.numpy_to_base64(frame)
                frame_data.append({'data': img_base64})

            response = await self.call_gemini_with_retry(frame_data, prompt)

            # Parse speaker response
            speakers = self.parse_speaker_response(response)
            return speakers

        except Exception as e:
            logger.error(f"Speaker detection error: {str(e)}")
            return []

    async def predict_motion(self, frames: List[np.ndarray]) -> Dict:
        """Predict future motion from sequential frames"""

        prompt = """
        Analyze the motion patterns in these sequential video frames.
        Predict where subjects and objects will move in the next 1-2 seconds.

        Return JSON:
        {
            "motion_vectors": {
                "subject_id": {"dx": float, "dy": float, "speed": float},
                ...
            },
            "predicted_positions": {
                "subject_id": {"x": float, "y": float, "confidence": float},
                ...
            },
            "scene_dynamics": "static|slow|moderate|fast|chaotic",
            "tracking_difficulty": 0.0-1.0,
            "confidence": 0.0-1.0
        }

        Focus on:
        - Direction and speed of movement
        - Predictable vs unpredictable motion
        - Subject entry/exit patterns
        - Camera movement vs subject movement
        """

        try:
            frame_data = []
            for frame in frames:
                img_base64 = await self.numpy_to_base64(frame)
                frame_data.append({'data': img_base64})

            response = await self.call_gemini_with_retry(frame_data, prompt)

            # Parse motion prediction
            motion_data = self.parse_motion_response(response)
            return motion_data

        except Exception as e:
            logger.error(f"Motion prediction error: {str(e)}")
            return {
                'motion_vectors': {},
                'predicted_positions': {},
                'scene_dynamics': 'moderate',
                'tracking_difficulty': 0.5,
                'confidence': 0.1
            }

    async def call_gemini_with_retry(
        self,
        frame_data: List[Dict],
        prompt: str,
        is_classification: bool = False
    ) -> str:
        """Call Gemini API with retry logic and rate limiting"""

        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)

        for attempt in range(self.max_retries):
            try:
                # Build the request parts
                parts = []

                # Add images
                for frame_info in frame_data:
                    parts.append({
                        'inline_data': {
                            'mime_type': 'image/jpeg',
                            'data': frame_info['data']
                        }
                    })

                # Add text prompt
                parts.append(prompt)

                # Call Gemini
                generation_config = {
                    'temperature': 0.2 if is_classification else 0.3,
                    'top_p': 0.95,
                    'max_output_tokens': 2048 if not is_classification else 1024
                }

                response = await asyncio.to_thread(
                    self.vision_model.generate_content,
                    parts,
                    generation_config=generation_config
                )

                self.last_request_time = time.time()

                if response and response.text:
                    return response.text
                else:
                    raise Exception("Empty response from Gemini")

            except Exception as e:
                logger.warning(f"Gemini API attempt {attempt + 1} failed: {str(e)}")

                if attempt == self.max_retries - 1:
                    raise e

                # Exponential backoff
                await asyncio.sleep(self.retry_delay * (2 ** attempt))

        raise Exception("All Gemini API retry attempts failed")

    async def numpy_to_base64(self, frame: np.ndarray) -> str:
        """Convert numpy array to base64 JPEG"""

        # Convert numpy array to PIL Image
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)

        # Handle different color channels
        if len(frame.shape) == 3:
            if frame.shape[2] == 3:
                img = Image.fromarray(frame, 'RGB')
            elif frame.shape[2] == 4:
                img = Image.fromarray(frame, 'RGBA')
            else:
                raise ValueError(f"Unsupported number of channels: {frame.shape[2]}")
        else:
            img = Image.fromarray(frame, 'L')  # Grayscale

        # Convert to JPEG bytes
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=85, optimize=True)
        img_bytes = buffer.getvalue()

        # Encode to base64
        return base64.b64encode(img_bytes).decode('utf-8')

    def parse_analysis_response(self, response_text: str, expected_count: int) -> List[Dict]:
        """Parse Gemini analysis response into structured data"""

        try:
            # Try to find JSON in the response
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                results = json.loads(json_str)

                # Validate results
                if isinstance(results, list) and len(results) <= expected_count:
                    return results
                else:
                    logger.warning(f"Unexpected result format: {len(results)} vs {expected_count}")

            # Try single object format
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                single_result = json.loads(json_str)
                return [single_result] if expected_count == 1 else [single_result] * expected_count

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            logger.error(f"Response text: {response_text[:500]}...")

        # Fallback
        return [self.create_fallback_result() for _ in range(expected_count)]

    def parse_classification_response(self, response_text: str) -> Dict:
        """Parse scene classification response"""

        try:
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                return json.loads(json_str)

        except json.JSONDecodeError as e:
            logger.error(f"Classification parsing error: {e}")

        # Fallback classification
        return {
            'type': 'auto',
            'description': 'Auto-detected content',
            'key_elements': ['unknown'],
            'subject_count': 1,
            'motion_level': 'medium',
            'recommended_strategy': 'Center tracking',
            'confidence': 0.3
        }

    def parse_speaker_response(self, response_text: str) -> List[str]:
        """Parse speaker detection response"""

        try:
            # Look for JSON array
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                speakers = json.loads(json_str)

                if isinstance(speakers, list):
                    return [str(s) for s in speakers]

        except json.JSONDecodeError as e:
            logger.error(f"Speaker parsing error: {e}")

        return []

    def parse_motion_response(self, response_text: str) -> Dict:
        """Parse motion prediction response"""

        try:
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                return json.loads(json_str)

        except json.JSONDecodeError as e:
            logger.error(f"Motion parsing error: {e}")

        return {
            'motion_vectors': {},
            'predicted_positions': {},
            'scene_dynamics': 'moderate',
            'tracking_difficulty': 0.5,
            'confidence': 0.1
        }

    def create_fallback_result(self) -> Dict:
        """Create fallback analysis result when Gemini fails"""

        return {
            'subjects': [{
                'id': 'fallback_subject',
                'type': 'object',
                'bbox': [0.5, 0.5, 0.3, 0.3],
                'confidence': 0.1,
                'is_speaking': False,
                'is_moving': False,
                'importance_score': 0.5,
                'name_or_description': 'Unknown subject'
            }],
            'primary_subject': 'fallback_subject',
            'suggested_center': [0.5, 0.5],
            'action_description': 'Analysis failed - using fallback',
            'text_regions': [],
            'motion_intensity': 0.3,
            'should_cut_here': False,
            'hold_duration_suggestion': 2.0,
            'confidence': 0.1
        }