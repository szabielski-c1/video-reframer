import google.generativeai as genai
import json
import asyncio
from typing import List, Dict, Any, Optional
import os
import logging
import time

from app.config import settings

logger = logging.getLogger(__name__)

class GeminiService:
    """Service for interacting with Google Gemini API for video analysis"""

    def __init__(self):
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(settings.GEMINI_MODEL)

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # 1 second between requests

        # Retry configuration
        self.max_retries = 3
        self.retry_delay = 5.0  # 5 seconds initial delay

        # Video upload tracking
        self.uploaded_files = []  # Track uploaded files for cleanup

    async def analyze_video_for_reframing(self, video_path: str, request_settings: Dict, metadata: Dict) -> Dict:
        """Analyze entire video and provide shot-by-shot reframing instructions"""

        video_file = None
        try:
            # Upload video to Gemini
            logger.info(f"Uploading video to Gemini: {os.path.basename(video_path)}")
            video_file = await self.upload_video_to_gemini(video_path)

            # Create comprehensive reframing analysis prompt
            prompt = self.create_reframing_analysis_prompt(request_settings, metadata)

            # Analyze video with Gemini
            response = await self.call_gemini_with_video(video_file, prompt)

            # Parse response into shot-based reframing data
            reframing_data = self.parse_reframing_response(response, metadata)

            logger.info(f"Successfully analyzed video with {len(reframing_data.get('shots', []))} shots")
            return reframing_data

        except Exception as e:
            logger.error(f"Video reframing analysis failed: {str(e)}")
            # Return fallback with single shot covering entire video
            duration = metadata.get('duration', 60)
            return {
                'shots': [{
                    'start_time': 0.0,
                    'end_time': duration,
                    'duration': duration,
                    'crop_center': [0.5, 0.5],  # Center crop
                    'crop_strategy': 'static_center',
                    'keyframes': [
                        {'timestamp': 0.0, 'center': [0.5, 0.5], 'zoom': 1.0},
                        {'timestamp': duration, 'center': [0.5, 0.5], 'zoom': 1.0}
                    ],
                    'confidence': 0.1
                }],
                'overall_strategy': 'center_crop_fallback',
                'confidence': 0.1
            }

        finally:
            # Clean up uploaded file
            if video_file:
                await self.cleanup_video_file(video_file)

    async def detect_shots(self, video_path: str, metadata: Dict) -> List[Dict]:
        """Detect shot boundaries and scene changes in the video"""

        video_file = None
        try:
            # Upload video to Gemini for shot detection
            video_file = await self.upload_video_to_gemini(video_path)

            duration = metadata.get('duration', 0)

            prompt = f"""
            Analyze this video to detect shot boundaries and scene changes with precise timestamps.

            Video duration: {duration} seconds

            Identify:
            1. Shot boundaries (cuts, transitions, scene changes)
            2. Camera movements (pan, zoom, tilt)
            3. Subject changes (new people entering/leaving frame)
            4. Content changes (different locations, activities)

            Return a JSON array of shots:
            [
                {{
                    "start_time": start_seconds,
                    "end_time": end_seconds,
                    "duration": duration_seconds,
                    "shot_type": "static|pan|zoom|tilt|handheld|cut",
                    "scene_description": "what's happening in this shot",
                    "primary_subjects": ["list", "of", "main", "subjects"],
                    "location_or_setting": "description of location/background",
                    "movement_intensity": 0.0_to_1.0,
                    "shot_stability": 0.0_to_1.0,
                    "recommended_crop_strategy": "center|follow_subject|wide_view|dynamic",
                    "confidence": 0.0_to_1.0
                }}
            ]

            Requirements:
            - Detect ALL shot boundaries, even quick cuts
            - Provide precise timestamps to 0.1 second accuracy
            - Ensure shots cover the entire video duration (0 to {duration} seconds)
            - No gaps or overlaps between shots
            - Minimum shot duration: 0.5 seconds
            - Focus on visual changes that would affect reframing strategy
            """

            response = await self.call_gemini_with_video(video_file, prompt)
            shots = self.parse_shot_detection_response(response, duration)

            logger.info(f"Detected {len(shots)} shots in {duration}s video")
            return shots

        except Exception as e:
            logger.error(f"Shot detection error: {str(e)}")
            # Return fallback: single shot covering entire video
            return [{
                'start_time': 0.0,
                'end_time': duration,
                'duration': duration,
                'shot_type': 'static',
                'scene_description': 'Shot detection failed',
                'primary_subjects': ['unknown'],
                'location_or_setting': 'unknown',
                'movement_intensity': 0.3,
                'shot_stability': 0.5,
                'recommended_crop_strategy': 'center',
                'confidence': 0.1
            }]

        finally:
            if video_file:
                await self.cleanup_video_file(video_file)

    async def classify_scene(self, video_path: str, request_settings: Dict) -> Dict:
        """Classify the scene type from the video"""

        video_file = None
        try:
            # Upload video to Gemini for classification
            video_file = await self.upload_video_to_gemini(video_path)

            prompt = """
            Analyze this video to classify the content type and scene characteristics.

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
            - Number and types of people/subjects throughout the video
            - Movement patterns and activity level
            - Setting and environment
            - Production style and quality
            - Text or graphics present
            - Overall video flow and pacing
            """

            response = await self.call_gemini_with_video(video_file, prompt)
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

        finally:
            if video_file:
                await self.cleanup_video_file(video_file)

    async def upload_video_to_gemini(self, video_path: str) -> any:
        """Upload video file to Gemini for analysis"""

        max_retries = self.max_retries
        retry_delay = self.retry_delay

        for attempt in range(max_retries):
            try:
                file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
                logger.info(f"Uploading video ({file_size_mb:.1f}MB) to Gemini (attempt {attempt + 1})")

                # Upload video file
                video_file = await asyncio.to_thread(
                    genai.upload_file,
                    path=video_path,
                    display_name=os.path.basename(video_path)
                )

                # Wait for processing
                while video_file.state.name == "PROCESSING":
                    logger.info("⏳ Waiting for video to be processed by Gemini...")
                    await asyncio.sleep(10)
                    video_file = await asyncio.to_thread(genai.get_file, name=video_file.name)

                if video_file.state.name == "FAILED":
                    raise ValueError("Video processing failed on Gemini side")

                logger.info(f"✓ Video uploaded successfully: {video_file.name}")
                self.uploaded_files.append(video_file)
                return video_file

            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Upload attempt {attempt + 1} failed: {error_msg}")

                if "[Errno 49]" in error_msg and attempt < max_retries - 1:
                    logger.info(f"Network error detected, retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue

                if attempt == max_retries - 1:
                    raise e

                await asyncio.sleep(retry_delay)

        raise Exception("All video upload attempts failed")

    async def cleanup_video_file(self, video_file) -> None:
        """Clean up uploaded video file from Gemini"""

        try:
            await asyncio.to_thread(genai.delete_file, name=video_file.name)
            logger.info(f"🗑️ Deleted video file from Gemini: {video_file.name}")

            # Remove from tracking list
            if video_file in self.uploaded_files:
                self.uploaded_files.remove(video_file)

        except Exception as e:
            logger.warning(f"Failed to delete video file {video_file.name}: {e}")

    async def call_gemini_with_video(self, video_file, prompt: str) -> str:
        """Call Gemini API with uploaded video file"""

        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)

        for attempt in range(self.max_retries):
            try:
                # Generation config optimized for video analysis
                generation_config = {
                    'temperature': 0.3,
                    'top_p': 0.95,
                    'max_output_tokens': 8192  # Larger token limit for video analysis
                }

                # Use BLOCK_ONLY_HIGH for safety settings
                safety_settings = [
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_ONLY_HIGH"
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_ONLY_HIGH"
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_ONLY_HIGH"
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_ONLY_HIGH"
                    }
                ]

                # Call Gemini with video and prompt
                response = await asyncio.to_thread(
                    self.model.generate_content,
                    [prompt, video_file],
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                    request_options={"timeout": 1800}  # 30 minute timeout for video processing
                )

                self.last_request_time = time.time()

                # Check response validity
                if not response:
                    raise Exception("No response from Gemini")

                # Check finish_reason for safety blocks
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'finish_reason'):
                        finish_reason = candidate.finish_reason

                        if finish_reason == 2:  # SAFETY
                            safety_ratings = getattr(candidate, 'safety_ratings', [])
                            raise Exception(f"Content blocked by safety filters (finish_reason: SAFETY). Ratings: {safety_ratings}")
                        elif finish_reason == 3:  # RECITATION
                            raise Exception("Content blocked due to recitation (finish_reason: RECITATION)")
                        elif finish_reason not in [1, None]:  # Not STOP or None
                            raise Exception(f"Abnormal finish_reason: {finish_reason}")

                # Check response text
                if response.text and len(response.text.strip()) > 0:
                    logger.info(f"✓ Gemini video analysis completed ({len(response.text)} chars)")
                    return response.text
                else:
                    raise Exception("Empty response text from Gemini")

            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Gemini video API attempt {attempt + 1} failed: {error_msg}")

                if attempt == self.max_retries - 1:
                    raise e

                # Exponential backoff
                await asyncio.sleep(self.retry_delay * (2 ** attempt))

        raise Exception("All Gemini video API retry attempts failed")

    def create_reframing_analysis_prompt(self, request_settings: Dict, metadata: Dict) -> str:
        """Create prompt for shot-by-shot reframing analysis"""

        duration = metadata.get('duration', 0)
        width = metadata.get('width', 1920)
        height = metadata.get('height', 1080)

        prompt = f"""
        Analyze this video for intelligent reframing from 16:9 horizontal to 9:16 vertical format.

        Video specs:
        - Duration: {duration} seconds
        - Resolution: {width}x{height}
        - Target output: 9:16 vertical video (56.25% of original width)

        Provide a complete shot-by-shot reframing plan:

        {{
            "shots": [
                {{
                    "start_time": seconds,
                    "end_time": seconds,
                    "duration": seconds,
                    "shot_description": "what's happening in this shot",
                    "crop_strategy": "static_center|follow_subject|pan_left_to_right|zoom_in|zoom_out|track_speaker",
                    "primary_subjects": ["main subjects to keep in frame"],
                    "crop_center": [x_0_to_1, y_0_to_1],  // Center point for the 9:16 crop
                    "keyframes": [
                        {{
                            "timestamp": seconds_into_shot,
                            "center": [x_0_to_1, y_0_to_1],  // Crop center at this time
                            "zoom": 1.0_to_2.0,             // Zoom level (1.0 = fit height, 2.0 = 2x zoom)
                            "description": "why this position"
                        }}
                    ],
                    "transition_to_next": "cut|smooth_pan",
                    "confidence": 0.0_to_1.0
                }}
            ],
            "overall_strategy": "description of reframing approach",
            "total_shots": number,
            "confidence": 0.0_to_1.0
        }}

        Requirements:
        1. **Shot Boundaries**: Detect ALL shot changes, cuts, and scene transitions
        2. **Precise Timestamps**: Provide exact start/end times for each shot
        3. **Crop Centers**: For each shot, determine optimal center point for 9:16 crop
        4. **Movement Tracking**: For shots with camera/subject movement, provide keyframes
        5. **Subject Priority**: Keep faces, speakers, and important elements in frame
        6. **Text Preservation**: Keep on-screen text and graphics visible when possible
        7. **Transition Handling**:
           - "cut": Keep original cut/jump (most common for music videos)
           - "smooth_pan": Only when crop position changes dramatically within same scene

        Shot Strategies:
        - **static_center**: Fixed center crop (for stable shots)
        - **follow_subject**: Track a moving person/object
        - **pan_left_to_right**: Gradual horizontal movement
        - **zoom_in/zoom_out**: Gradual zoom changes
        - **track_speaker**: Follow whoever is speaking

        Keyframes:
        - Provide keyframes for any shot longer than 3 seconds
        - Include start, end, and intermediate points for movement
        - Each keyframe specifies exact crop center and zoom level

        Cover the entire {duration} seconds with no gaps between shots.
        """

        return prompt

    def parse_reframing_response(self, response_text: str, metadata: Dict) -> Dict:
        """Parse Gemini reframing response into shot-based reframing plan"""

        duration = metadata.get('duration', 0)

        try:
            # Extract JSON object from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                reframing_data = json.loads(json_str)

                if isinstance(reframing_data, dict) and 'shots' in reframing_data:
                    shots = reframing_data['shots']

                    if isinstance(shots, list) and len(shots) > 0:
                        # Validate and clean up shots
                        validated_shots = []
                        last_end_time = 0.0

                        for i, shot in enumerate(shots):
                            if isinstance(shot, dict) and 'start_time' in shot and 'end_time' in shot:
                                # Ensure no gaps between shots
                                if i > 0:
                                    shot['start_time'] = max(shot['start_time'], last_end_time)

                                # Ensure shot doesn't exceed video duration
                                shot['end_time'] = min(shot['end_time'], duration)

                                # Ensure minimum duration
                                if shot['end_time'] - shot['start_time'] >= 0.5:
                                    shot['duration'] = shot['end_time'] - shot['start_time']

                                    # Ensure required fields exist
                                    if 'crop_center' not in shot:
                                        shot['crop_center'] = [0.5, 0.5]
                                    if 'keyframes' not in shot:
                                        shot['keyframes'] = [
                                            {'timestamp': shot['start_time'], 'center': shot['crop_center'], 'zoom': 1.0},
                                            {'timestamp': shot['end_time'], 'center': shot['crop_center'], 'zoom': 1.0}
                                        ]

                                    validated_shots.append(shot)
                                    last_end_time = shot['end_time']

                        # Ensure we cover the full video duration
                        if validated_shots and validated_shots[-1]['end_time'] < duration:
                            # Extend last shot to cover remaining time
                            validated_shots[-1]['end_time'] = duration
                            validated_shots[-1]['duration'] = duration - validated_shots[-1]['start_time']
                            # Update last keyframe timestamp
                            if validated_shots[-1]['keyframes']:
                                validated_shots[-1]['keyframes'][-1]['timestamp'] = duration

                        if len(validated_shots) > 0:
                            logger.info(f"Parsed {len(validated_shots)} shots from reframing response")
                            return {
                                'shots': validated_shots,
                                'overall_strategy': reframing_data.get('overall_strategy', 'parsed_from_gemini'),
                                'confidence': reframing_data.get('confidence', 0.8)
                            }

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in reframing response: {e}")
            logger.error(f"Response text sample: {response_text[:1000]}...")
        except Exception as e:
            logger.error(f"Error parsing reframing response: {e}")

        # Fallback: single shot covering entire video
        logger.warning(f"Using fallback reframing for {duration}s video")
        return {
            'shots': [{
                'start_time': 0.0,
                'end_time': duration,
                'duration': duration,
                'shot_description': 'Fallback single shot',
                'crop_strategy': 'static_center',
                'crop_center': [0.5, 0.5],
                'keyframes': [
                    {'timestamp': 0.0, 'center': [0.5, 0.5], 'zoom': 1.0},
                    {'timestamp': duration, 'center': [0.5, 0.5], 'zoom': 1.0}
                ],
                'confidence': 0.1
            }],
            'overall_strategy': 'center_crop_fallback',
            'confidence': 0.1
        }

    def parse_classification_response(self, response_text: str) -> Dict:
        """Parse scene classification response"""

        try:
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                return json.loads(json_str)

        except json.JSONDecodeError as e:
            # Count JSON parsing errors as failures too
            self.consecutive_failures += 1
            logger.error(f"Classification parsing error: {e} (total failures: {self.consecutive_failures})")

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

    def repair_truncated_json(self, json_str: str) -> Optional[str]:
        """Repair truncated or malformed JSON from Gemini responses"""

        try:
            # Try to parse as-is first
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError:
            pass

        # Common fixes for truncated JSON
        fixed = json_str.strip()

        # Fix unclosed arrays
        if fixed.startswith('[') and not fixed.endswith(']'):
            # Count opening vs closing brackets
            open_brackets = fixed.count('[')
            close_brackets = fixed.count(']')
            if open_brackets > close_brackets:
                fixed += ']' * (open_brackets - close_brackets)

        # Fix unclosed objects
        if fixed.startswith('{') and not fixed.endswith('}'):
            open_braces = fixed.count('{')
            close_braces = fixed.count('}')
            if open_braces > close_braces:
                fixed += '}' * (open_braces - close_braces)

        # Fix trailing commas
        fixed = fixed.replace(',]', ']').replace(',}', '}')

        try:
            json.loads(fixed)
            logger.info("Successfully repaired truncated JSON")
            return fixed
        except json.JSONDecodeError:
            logger.warning("Could not repair truncated JSON")
            return None

    def parse_shot_detection_response(self, response_text: str, duration: float) -> List[Dict]:
        """Parse shot detection response into shot boundaries"""

        try:
            # Extract JSON array from response
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                shots = json.loads(json_str)

                if isinstance(shots, list) and len(shots) > 0:
                    # Validate and clean shots
                    validated_shots = []
                    last_end_time = 0.0

                    for i, shot in enumerate(shots):
                        if isinstance(shot, dict) and 'start_time' in shot and 'end_time' in shot:
                            # Ensure no gaps between shots
                            if i > 0:
                                shot['start_time'] = max(shot['start_time'], last_end_time)

                            # Ensure shot doesn't exceed video duration
                            shot['end_time'] = min(shot['end_time'], duration)

                            # Ensure minimum duration
                            if shot['end_time'] - shot['start_time'] >= 0.5:
                                shot['duration'] = shot['end_time'] - shot['start_time']
                                validated_shots.append(shot)
                                last_end_time = shot['end_time']

                    # Ensure we cover the full video duration
                    if validated_shots and validated_shots[-1]['end_time'] < duration:
                        # Extend last shot to cover remaining time
                        validated_shots[-1]['end_time'] = duration
                        validated_shots[-1]['duration'] = duration - validated_shots[-1]['start_time']

                    if len(validated_shots) > 0:
                        logger.info(f"Validated {len(validated_shots)} shots from Gemini response")
                        return validated_shots

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in shot detection: {e}")
        except Exception as e:
            logger.error(f"Error parsing shot detection response: {e}")

        # Fallback: create reasonable shot segments
        logger.warning(f"Using fallback shot detection for {duration}s video")
        shot_duration = min(30.0, duration / 3)  # 30s or 1/3 of video, whichever is smaller
        shots = []

        current_time = 0.0
        shot_id = 1
        while current_time < duration:
            end_time = min(current_time + shot_duration, duration)
            shots.append({
                'start_time': current_time,
                'end_time': end_time,
                'duration': end_time - current_time,
                'shot_type': 'static',
                'scene_description': f'Shot {shot_id}',
                'primary_subjects': ['unknown'],
                'location_or_setting': 'unknown',
                'movement_intensity': 0.3,
                'shot_stability': 0.5,
                'recommended_crop_strategy': 'center',
                'confidence': 0.2
            })
            current_time = end_time
            shot_id += 1

        return shots

    async def cleanup_all_files(self) -> None:
        """Clean up all uploaded video files"""

        for video_file in self.uploaded_files.copy():
            await self.cleanup_video_file(video_file)

    async def health_check(self) -> Dict:
        """Test Gemini API connectivity and functionality"""

        start_time = time.time()

        try:
            # Simple text-only health check
            test_prompt = "Respond with only this JSON: {\"status\": \"ok\", \"test\": \"passed\"}"

            # Use basic model call without video
            generation_config = {
                'temperature': 0.1,
                'top_p': 0.95,
                'max_output_tokens': 128
            }

            response = await asyncio.to_thread(
                self.model.generate_content,
                test_prompt,
                generation_config=generation_config
            )

            response_time = time.time() - start_time

            # Validate response
            if response and response.text:
                try:
                    json_start = response.text.find('{')
                    json_end = response.text.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json.loads(response.text[json_start:json_end])
                        json_valid = True
                    else:
                        json_valid = False
                except:
                    json_valid = False

                return {
                    'status': 'healthy' if json_valid else 'degraded',
                    'response_time_ms': round(response_time * 1000, 2),
                    'model': settings.GEMINI_MODEL,
                    'json_parsing': 'working' if json_valid else 'issues_detected',
                    'video_upload': 'supported',
                    'last_checked': time.time()
                }
            else:
                raise Exception("No response from Gemini")

        except Exception as e:
            response_time = time.time() - start_time
            error_str = str(e)

            # Categorize error
            if "not found" in error_str:
                status = 'model_not_found'
            elif "quota" in error_str.lower() or "limit" in error_str.lower():
                status = 'quota_exceeded'
            elif "finish_reason" in error_str:
                status = 'safety_blocked'
            else:
                status = 'connection_failed'

            return {
                'status': status,
                'error': error_str,
                'response_time_ms': round(response_time * 1000, 2),
                'model': settings.GEMINI_MODEL,
                'video_upload': 'unknown',
                'last_checked': time.time()
            }

    def create_fallback_result(self, timestamp: float = 0.0) -> Dict:
        """Create fallback analysis result when Gemini fails"""

        return {
            'timestamp': timestamp,
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