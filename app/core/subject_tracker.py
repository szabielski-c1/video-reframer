import logging
from typing import List, Dict, Optional
import numpy as np

from app.models import FrameAnalysis, SubjectInfo

logger = logging.getLogger(__name__)

class SubjectTracker:
    """Tracks subjects across frames for consistent identification"""

    def __init__(self):
        self.subject_history = {}
        self.id_counter = 0

    def track_subjects(self, analyses: List[FrameAnalysis]) -> List[FrameAnalysis]:
        """Track subjects across all frames for consistency"""

        for i, analysis in enumerate(analyses):
            # Update subject IDs based on tracking
            analysis = self.update_subject_ids(analysis, i)

            # Update subject statistics
            self.update_subject_stats(analysis)

        # Apply consistency corrections
        analyses = self.apply_temporal_consistency(analyses)

        return analyses

    def update_subject_ids(self, analysis: FrameAnalysis, frame_idx: int) -> FrameAnalysis:
        """Update subject IDs based on tracking history"""

        if not analysis.subjects:
            return analysis

        # Match subjects to existing tracked subjects
        matched_subjects = []
        for subject in analysis.subjects:
            matched_id = self.find_matching_subject(subject, analysis.timestamp)

            if matched_id:
                # Update existing subject
                subject.id = matched_id
                self.subject_history[matched_id]['last_seen'] = analysis.timestamp
                self.subject_history[matched_id]['positions'].append({
                    'timestamp': analysis.timestamp,
                    'bbox': subject.bbox,
                    'confidence': subject.confidence
                })
            else:
                # New subject
                new_id = f"subject_{self.id_counter}"
                self.id_counter += 1
                subject.id = new_id

                self.subject_history[new_id] = {
                    'type': subject.type,
                    'description': subject.name_or_description,
                    'first_seen': analysis.timestamp,
                    'last_seen': analysis.timestamp,
                    'positions': [{
                        'timestamp': analysis.timestamp,
                        'bbox': subject.bbox,
                        'confidence': subject.confidence
                    }],
                    'total_frames': 1,
                    'speaking_frames': 1 if subject.is_speaking else 0,
                    'importance_sum': subject.importance_score
                }

            matched_subjects.append(subject)

        analysis.subjects = matched_subjects

        # Update primary subject if needed
        if analysis.primary_subject:
            analysis.primary_subject = self.validate_primary_subject(analysis)

        return analysis

    def find_matching_subject(self, subject: SubjectInfo, timestamp: float) -> Optional[str]:
        """Find matching subject from history using IoU and features"""

        best_match = None
        best_score = 0.5  # Minimum matching threshold

        for subject_id, history in self.subject_history.items():
            # Skip if not seen recently (> 2 seconds)
            if timestamp - history['last_seen'] > 2.0:
                continue

            # Skip if type doesn't match
            if history['type'] != subject.type:
                continue

            # Calculate matching score
            score = self.calculate_matching_score(subject, history)

            if score > best_score:
                best_score = score
                best_match = subject_id

        return best_match

    def calculate_matching_score(self, subject: SubjectInfo, history: Dict) -> float:
        """Calculate matching score between subject and historical track"""

        score = 0.0

        # Get last known position
        if history['positions']:
            last_pos = history['positions'][-1]

            # Calculate IoU (Intersection over Union)
            iou = self.calculate_iou(subject.bbox, last_pos['bbox'])
            score += iou * 0.5

            # Distance-based score
            dist = self.calculate_distance(subject.bbox, last_pos['bbox'])
            dist_score = max(0, 1 - dist / 0.5)  # Normalize distance
            score += dist_score * 0.3

        # Description similarity
        if subject.name_or_description and history['description']:
            desc_score = self.calculate_description_similarity(
                subject.name_or_description,
                history['description']
            )
            score += desc_score * 0.2

        return score

    def calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union for bounding boxes"""

        # Convert center format to corner format
        x1_min = bbox1[0] - bbox1[2] / 2
        y1_min = bbox1[1] - bbox1[3] / 2
        x1_max = bbox1[0] + bbox1[2] / 2
        y1_max = bbox1[1] + bbox1[3] / 2

        x2_min = bbox2[0] - bbox2[2] / 2
        y2_min = bbox2[1] - bbox2[3] / 2
        x2_max = bbox2[0] + bbox2[2] / 2
        y2_max = bbox2[1] + bbox2[3] / 2

        # Calculate intersection
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)

        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
            return 0.0

        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)

        # Calculate union
        area1 = bbox1[2] * bbox1[3]
        area2 = bbox2[2] * bbox2[3]
        union_area = area1 + area2 - inter_area

        if union_area == 0:
            return 0.0

        return inter_area / union_area

    def calculate_distance(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Euclidean distance between bbox centers"""

        return np.sqrt((bbox1[0] - bbox2[0])**2 + (bbox1[1] - bbox2[1])**2)

    def calculate_description_similarity(self, desc1: str, desc2: str) -> float:
        """Simple description similarity based on common words"""

        words1 = set(desc1.lower().split())
        words2 = set(desc2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def update_subject_stats(self, analysis: FrameAnalysis):
        """Update statistics for tracked subjects"""

        for subject in analysis.subjects:
            if subject.id in self.subject_history:
                history = self.subject_history[subject.id]
                history['total_frames'] += 1
                if subject.is_speaking:
                    history['speaking_frames'] += 1
                history['importance_sum'] += subject.importance_score

    def validate_primary_subject(self, analysis: FrameAnalysis) -> Optional[str]:
        """Validate and potentially correct primary subject selection"""

        if not analysis.primary_subject:
            return None

        # Check if primary subject exists in current frame
        subject_ids = [s.id for s in analysis.subjects]
        if analysis.primary_subject not in subject_ids:
            # Find best replacement
            if analysis.subjects:
                # Choose subject with highest importance
                best_subject = max(analysis.subjects, key=lambda s: s.importance_score)
                return best_subject.id
            return None

        return analysis.primary_subject

    def apply_temporal_consistency(self, analyses: List[FrameAnalysis]) -> List[FrameAnalysis]:
        """Apply temporal consistency corrections"""

        # Smooth primary subject transitions
        for i in range(1, len(analyses) - 1):
            prev_primary = analyses[i-1].primary_subject
            curr_primary = analyses[i].primary_subject
            next_primary = analyses[i+1].primary_subject

            # If current is different but prev and next are same, might be a glitch
            if curr_primary != prev_primary and prev_primary == next_primary:
                # Check if it's a brief glitch
                if self.is_subject_present(prev_primary, analyses[i]):
                    logger.debug(f"Correcting primary subject glitch at frame {i}")
                    analyses[i].primary_subject = prev_primary

        return analyses

    def is_subject_present(self, subject_id: str, analysis: FrameAnalysis) -> bool:
        """Check if subject is present in frame"""

        return any(s.id == subject_id for s in analysis.subjects)

    def get_subject_statistics(self) -> Dict:
        """Get statistics for all tracked subjects"""

        stats = {}
        for subject_id, history in self.subject_history.items():
            total_time = history['last_seen'] - history['first_seen']
            avg_importance = history['importance_sum'] / history['total_frames'] if history['total_frames'] > 0 else 0

            stats[subject_id] = {
                'type': history['type'],
                'description': history['description'],
                'screen_time': total_time,
                'frame_count': history['total_frames'],
                'speaking_ratio': history['speaking_frames'] / history['total_frames'] if history['total_frames'] > 0 else 0,
                'average_importance': avg_importance
            }

        return stats