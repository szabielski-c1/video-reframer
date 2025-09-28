import logging
import json
from typing import List, Dict, Optional
import numpy as np

from app.models import FrameAnalysis, SubjectInfo, VideoStyle, ReframeRequest

logger = logging.getLogger(__name__)

class FrameAnalyzer:
    """Analyzes video frames using Gemini AI for intelligent reframing decisions"""

    def __init__(self, gemini_service):
        self.gemini = gemini_service

    async def classify_scene(self, frames: List[np.ndarray], request: ReframeRequest) -> Dict:
        """Classify the overall scene type from initial frames"""

        prompt = self.build_scene_classification_prompt(request)
        scene_info = await self.gemini.classify_scene(frames)

        # Override with user preference if specified
        if request.settings.style != VideoStyle.AUTO:
            scene_info['type'] = request.settings.style.value

        logger.info(f"Scene classified as: {scene_info['type']} with confidence {scene_info.get('confidence', 0)}")
        return scene_info

    async def analyze_batch(
        self,
        frames: List[np.ndarray],
        timestamps: List[float],
        scene_info: Dict,
        request: ReframeRequest,
        previous_analyses: Optional[List[FrameAnalysis]] = None
    ) -> List[FrameAnalysis]:
        """Analyze a batch of frames with context"""

        prompt = self.build_frame_analysis_prompt(
            timestamps,
            scene_info,
            request,
            previous_analyses
        )

        # Get Gemini analysis
        raw_analyses = await self.gemini.analyze_frames(frames, prompt)

        # Parse and validate results
        frame_analyses = []
        for i, raw_data in enumerate(raw_analyses):
            try:
                analysis = self.parse_gemini_analysis(
                    raw_data,
                    timestamps[i] if i < len(timestamps) else 0.0,
                    scene_info.get('type', 'auto')
                )
                frame_analyses.append(analysis)
            except Exception as e:
                logger.error(f"Failed to parse analysis for frame {i}: {e}")
                # Create fallback analysis
                frame_analyses.append(self.create_fallback_analysis(timestamps[i] if i < len(timestamps) else 0.0))

        return frame_analyses

    def build_scene_classification_prompt(self, request: ReframeRequest) -> str:
        """Build prompt for scene classification"""

        base_prompt = """
        Analyze these video frames to determine the overall content type and style.
        This will help guide the reframing strategy for converting from 16:9 to 9:16 format.

        Consider:
        - Number and type of subjects (people, objects, text)
        - Movement patterns and dynamics
        - Setting and environment
        - Production style
        - Audio cues if available

        Classify into one of these categories:
        - interview: Conversation between people
        - vlog: Single person talking to camera
        - sports: Athletic activity or competition
        - presentation: Educational or business content
        - documentary: Narrative or informational content
        - music: Musical performance
        - auto: Mixed or unclear content
        """

        if request.gemini_prompts.custom_focus:
            base_prompt += f"\n\nAdditional context: {request.gemini_prompts.custom_focus}"

        return base_prompt

    def build_frame_analysis_prompt(
        self,
        timestamps: List[float],
        scene_info: Dict,
        request: ReframeRequest,
        previous_analyses: Optional[List[FrameAnalysis]] = None
    ) -> str:
        """Build comprehensive prompt for frame analysis"""

        base_prompt = f"""
        Analyze these video frames for intelligent reframing from 16:9 to 9:16 format.

        Scene Type: {scene_info.get('type', 'unknown')}
        Scene Description: {scene_info.get('description', '')}

        For EACH frame at timestamps {timestamps}, provide a JSON response with:
        {{
            "subjects": [
                {{
                    "id": "unique_identifier",
                    "type": "person|object|text|animal",
                    "bbox": [x_center, y_center, width, height],
                    "confidence": 0.0-1.0,
                    "is_speaking": boolean,
                    "is_moving": boolean,
                    "importance_score": 0.0-1.0,
                    "name_or_description": "description"
                }}
            ],
            "primary_subject": "subject_id or null",
            "suggested_center": [x, y],
            "action_description": "What's happening",
            "text_regions": [{{"bbox": [x, y, w, h], "importance": "high|medium|low"}}],
            "motion_intensity": 0.0-1.0,
            "should_cut_here": boolean,
            "hold_duration_suggestion": float,
            "confidence": 0.0-1.0
        }}

        CRITICAL RULES FOR 9:16 REFRAMING:
        1. The vertical frame MUST capture the most important content
        2. Prioritize: Speaker > Active person > Main subject > Background
        3. For multiple subjects, choose based on importance and action
        4. Maintain smooth transitions between frames
        5. Suggest cuts only when panning would be too jarring
        6. Consider text readability in vertical format
        7. Account for edge padding to avoid awkward crops

        Scene-Specific Strategy for {scene_info.get('type', 'unknown')}:
        {self.get_scene_specific_instructions(scene_info.get('type', 'auto'))}
        """

        # Add custom instructions
        if request.gemini_prompts.custom_focus:
            base_prompt += f"\n\nCUSTOM FOCUS: {request.gemini_prompts.custom_focus}"

        if request.gemini_prompts.priority_subjects:
            base_prompt += f"\n\nPRIORITY SUBJECTS: {', '.join(request.gemini_prompts.priority_subjects)}"

        if request.gemini_prompts.exclude_areas:
            base_prompt += f"\n\nEXCLUDE AREAS: {', '.join(request.gemini_prompts.exclude_areas)}"

        # Add temporal context
        if previous_analyses:
            recent_primary = previous_analyses[-1].primary_subject if previous_analyses else None
            base_prompt += f"\n\nPREVIOUS CONTEXT: Last primary subject was '{recent_primary}'. Maintain continuity unless scene changes significantly."

        return base_prompt

    def get_scene_specific_instructions(self, scene_type: str) -> str:
        """Get scene-specific framing instructions"""

        instructions = {
            "interview": """
            - Hold steady on current speaker (3-5 seconds minimum)
            - Quick cut to listener for reactions
            - Avoid panning between distant speakers
            - Include hand gestures when relevant
            """,

            "vlog": """
            - Keep vlogger's face centered and stable
            - Adjust for hand gestures and props
            - Follow eyeline to referenced objects
            - Maintain intimate framing
            """,

            "sports": """
            - Track the ball/puck as highest priority
            - Use predictive framing for action direction
            - Quick cuts acceptable for fast action
            - Include celebrating players after scores
            """,

            "presentation": """
            - Balance speaker and visual content
            - Prioritize readable text/slides
            - Follow pointing gestures to content
            - Hold on important information
            """,

            "music": """
            - Follow primary performer
            - Cut on beat changes when natural
            - Include instruments during solos
            - Capture audience reactions if shown
            """,

            "documentary": """
            - Stable, thoughtful framing
            - Longer holds (4-6 seconds)
            - Follow narrative focus
            - Preserve compositional beauty
            """,

            "auto": """
            - Identify and follow main subject
            - Smooth, professional movements
            - Avoid jarring transitions
            - Keep important elements in frame
            """
        }

        return instructions.get(scene_type, instructions["auto"])

    def parse_gemini_analysis(self, raw_data: Dict, timestamp: float, scene_type: str) -> FrameAnalysis:
        """Parse Gemini response into FrameAnalysis object"""

        # Parse subjects
        subjects = []
        for subj_data in raw_data.get('subjects', []):
            try:
                subject = SubjectInfo(
                    id=subj_data.get('id', f"subject_{len(subjects)}"),
                    type=subj_data.get('type', 'object'),
                    bbox=subj_data.get('bbox', [0.5, 0.5, 0.2, 0.2]),
                    confidence=subj_data.get('confidence', 0.5),
                    is_speaking=subj_data.get('is_speaking', False),
                    is_moving=subj_data.get('is_moving', False),
                    importance_score=subj_data.get('importance_score', 0.5),
                    name_or_description=subj_data.get('name_or_description', 'Unknown')
                )
                subjects.append(subject)
            except Exception as e:
                logger.warning(f"Failed to parse subject: {e}")

        # Create frame analysis
        return FrameAnalysis(
            timestamp=timestamp,
            subjects=subjects,
            primary_subject=raw_data.get('primary_subject'),
            scene_type=VideoStyle(scene_type) if scene_type in [e.value for e in VideoStyle] else VideoStyle.AUTO,
            motion_vectors=raw_data.get('motion_vectors', {}),
            text_regions=raw_data.get('text_regions', []),
            suggested_center=tuple(raw_data.get('suggested_center', [0.5, 0.5])),
            confidence=raw_data.get('confidence', 0.5),
            action_description=raw_data.get('action_description', ''),
            audio_peak=raw_data.get('audio_peak'),
            should_cut_here=raw_data.get('should_cut_here', False),
            hold_duration_suggestion=raw_data.get('hold_duration_suggestion', 2.0)
        )

    def create_fallback_analysis(self, timestamp: float) -> FrameAnalysis:
        """Create a fallback analysis when Gemini fails"""

        return FrameAnalysis(
            timestamp=timestamp,
            subjects=[],
            primary_subject=None,
            scene_type=VideoStyle.AUTO,
            motion_vectors={},
            text_regions=[],
            suggested_center=(0.5, 0.5),
            confidence=0.1,
            action_description="Analysis failed - using center crop",
            audio_peak=None,
            should_cut_here=False,
            hold_duration_suggestion=2.0
        )

    def extract_recent_subjects(self, analyses: List[FrameAnalysis]) -> str:
        """Extract summary of recent subjects for context"""

        subject_ids = set()
        for analysis in analyses[-5:]:  # Last 5 frames
            for subject in analysis.subjects:
                subject_ids.add(f"{subject.id}:{subject.name_or_description}")

        return ", ".join(list(subject_ids)[:5])  # Limit to 5 subjects