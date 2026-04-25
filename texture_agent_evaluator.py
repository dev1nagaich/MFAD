"""
texture_agent_evaluator.py — Comprehensive Evaluation & Benchmarking
════════════════════════════════════════════════════════════════════════════════

Evaluation suite for TextureAgent across multiple datasets with independent
face detection (no preprocessing agent dependency).

Datasets Supported:
  • FF++ (FaceForensics++): 1000 videos (raw & compressed)
  • Celeb-DF v2: 590 real + 5,639 fake videos
  • DFDC (Deepfake Detection Challenge): 19,154 videos
  • WildDeepfake: 707 videos (in-the-wild)
  • All dataset folders in ./dataset/

Metrics Computed:
  • AUC-ROC, AUC-PR, Accuracy, Precision, Recall, F1, Threshold-optimal Youden
  • Per-dataset, per-method cross-validation
  • Confusion matrices, calibration curves
  • Per-zone forensic statistics

Face Detection:
  • MediaPipe FaceMesh (default) — fast, accurate
  • Fallback to dlib if needed
  • No dependency on preprocessing_agent
"""

from __future__ import annotations

import os
import sys
import json
import logging
import argparse
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass, asdict, field
from collections import defaultdict

# ─ Image Processing ───────────────────────────────────────────────────────────
import cv2
from PIL import Image

# ─ Face Detection ──────────────────────────────────────────────────────────────
import mediapipe as mp

try:
    import dlib
    HAS_DLIB = True
except ImportError:
    HAS_DLIB = False

# ─ Metrics ────────────────────────────────────────────────────────────────────
from sklearn.metrics import (
    roc_auc_score, auc, roc_curve, precision_recall_curve, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score
)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ─ Project imports ─────────────────────────────────────────────────────────────
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from agents.texture_agent import TextureAgent, BoundingBox, TextureAnalysisResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
log = logging.getLogger("texture_evaluator")

# ═════════════════════════════════════════════════════════════════════════════
# FACE DETECTION ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class FaceDetector:
    """Independent face detection using MediaPipe or OpenCV."""
    
    def __init__(self, backend: str = "mediapipe"):
        """
        Args:
            backend: "mediapipe", "opencv", or "dlib"
        """
        self.backend = backend
        
        try:
            if backend == "mediapipe":
                self._init_mediapipe()
            elif backend == "opencv":
                self._init_opencv()
            else:
                self._init_dlib()
        except Exception as e:
            log.warning(f"Failed to init {backend}: {e}. Falling back to OpenCV.")
            self._init_opencv()
        
        log.info(f"FaceDetector initialized with {self.backend}")
    
    def _init_mediapipe(self):
        """Initialize MediaPipe face detection (v0.10+)."""
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            
            base_options = python.BaseOptions(model_asset_path=None)
            options = vision.FaceDetectorOptions(base_options=base_options)
            self.face_detector = vision.FaceDetector.create_from_options(options)
            self.backend = "mediapipe"
        except Exception as e:
            log.warning(f"MediaPipe Tasks API failed: {e}. Using face_mesh fallback.")
            self._init_mediapipe_mesh()
    
    def _init_mediapipe_mesh(self):
        """Initialize MediaPipe FaceMesh (older API fallback)."""
        try:
            mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=10,
                refine_landmarks=False,
                min_detection_confidence=0.5
            )
            self.backend = "mediapipe_mesh"
        except Exception as e:
            raise RuntimeError(f"MediaPipe initialization failed: {e}")
    
    def _init_opencv(self):
        """Initialize OpenCV Haar cascade face detector."""
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_detector = cv2.CascadeClassifier(cascade_path)
        self.backend = "opencv"
    
    def _init_dlib(self):
        """Initialize dlib face detector."""
        if not HAS_DLIB:
            raise ImportError(
                "dlib not installed. Install with: pip install dlib\n"
                "Or use MediaPipe backend (default): FaceDetector(backend='mediapipe')"
            )
        self.face_detector = dlib.get_frontal_face_detector()
    
    def detect(self, image: np.ndarray | Image.Image) -> List[BoundingBox]:
        """
        Detect faces in image.
        
        Args:
            image: OpenCV BGR array or PIL Image
        
        Returns:
            List[BoundingBox] for each detected face
        """
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        try:
            if self.backend == "mediapipe":
                return self._detect_mediapipe(image)
            elif self.backend == "mediapipe_mesh":
                return self._detect_mediapipe_mesh(image)
            elif self.backend == "opencv":
                return self._detect_opencv(image)
            else:
                return self._detect_dlib(image)
        except Exception as e:
            log.warning(f"Face detection with {self.backend} failed: {e}. Trying OpenCV.")
            try:
                return self._detect_opencv(image)
            except Exception as e2:
                log.error(f"All face detection methods failed: {e2}")
                return []
    
    def _detect_mediapipe(self, image_bgr: np.ndarray) -> List[BoundingBox]:
        """MediaPipe Tasks API face detection (v0.10+)."""
        import numpy as np
        from mediapipe.framework.formats import detection_pb2
        
        h, w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        detection_result = self.face_detector.detect(mp_image)
        
        bboxes = []
        for detection in detection_result.detections:
            bbox_data = detection.bounding_box
            
            x1 = int(bbox_data.origin_x)
            y1 = int(bbox_data.origin_y)
            x2 = int(bbox_data.origin_x + bbox_data.width)
            y2 = int(bbox_data.origin_y + bbox_data.height)
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 > x1 and y2 > y1:
                bboxes.append(BoundingBox(x1, y1, x2, y2))
        
        return bboxes
    
    def _detect_mediapipe_mesh(self, image_bgr: np.ndarray) -> List[BoundingBox]:
        """MediaPipe FaceMesh fallback."""
        h, w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        results = self.face_mesh.process(image_rgb)
        
        bboxes = []
        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                # Convert landmarks to bbox
                xs = [lm.x * w for lm in landmarks.landmark]
                ys = [lm.y * h for lm in landmarks.landmark]
                
                x1, x2 = int(min(xs)), int(max(xs))
                y1, y2 = int(min(ys)), int(max(ys))
                
                # Add padding
                pad_x = int((x2 - x1) * 0.1)
                pad_y = int((y2 - y1) * 0.1)
                
                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - pad_y)
                x2 = min(w, x2 + pad_x)
                y2 = min(h, y2 + pad_y)
                
                if x2 > x1 and y2 > y1:
                    bboxes.append(BoundingBox(x1, y1, x2, y2))
        
        return bboxes
    
    def _detect_opencv(self, image_bgr: np.ndarray) -> List[BoundingBox]:
        """OpenCV Haar cascade detection."""
        h, w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20),
            maxSize=(w, h)
        )
        
        bboxes = []
        for (x, y, fw, fh) in faces:
            x1, y1 = x, y
            x2, y2 = x + fw, y + fh
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 > x1 and y2 > y1:
                bboxes.append(BoundingBox(x1, y1, x2, y2))
        
        return bboxes
    
    def _detect_dlib(self, image_bgr: np.ndarray) -> List[BoundingBox]:
        """dlib face detection."""
        h, w = image_bgr.shape[:2]
        
        # dlib expects BGR
        dets = self.face_detector(image_bgr, 1)
        
        bboxes = []
        for det in dets:
            x1, y1 = det.left(), det.top()
            x2, y2 = det.right(), det.bottom()
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 > x1 and y2 > y1:
                bboxes.append(BoundingBox(x1, y1, x2, y2))
        
        return bboxes


# ═════════════════════════════════════════════════════════════════════════════
# EVALUATION DATA STRUCTURES
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class EvaluationSample:
    """Single evaluation sample."""
    image_path: str
    label: int  # 0=real, 1=fake
    dataset: str
    detection_method: str = "unknown"
    face_bbox: Optional[BoundingBox] = None
    texture_result: Optional[TextureAnalysisResult] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        d = {
            "image_path": self.image_path,
            "label": self.label,
            "dataset": self.dataset,
            "detection_method": self.detection_method,
            "error": self.error,
        }
        
        if self.face_bbox:
            d["face_bbox"] = [self.face_bbox.x1, self.face_bbox.y1,
                               self.face_bbox.x2, self.face_bbox.y2]
        
        if self.texture_result:
            d["texture_result"] = self.texture_result.to_dict()
        
        return d


@dataclass
class EvaluationMetrics:
    """Evaluation metrics container."""
    dataset: str
    total_samples: int = 0
    real_samples: int = 0
    fake_samples: int = 0
    
    # Detection rate
    faces_detected: int = 0
    detection_rate: float = 0.0
    
    # Classification metrics
    auc_roc: float = 0.0
    auc_pr: float = 0.0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    
    # Threshold metrics
    optimal_threshold: float = 0.5
    youden_index: float = 0.0
    
    # Per-class metrics
    tn: int = 0
    fp: int = 0
    fn: int = 0
    tp: int = 0
    
    # Zone statistics
    zone_stats: Dict[str, Dict] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def __str__(self) -> str:
        return f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Texture Agent Evaluation Report — {self.dataset}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Dataset Statistics:
  Total samples: {self.total_samples}
  Real videos: {self.real_samples}
  Fake videos: {self.fake_samples}
  Ratio: {100*self.fake_samples/max(self.total_samples, 1):.1f}% fake

🔍 Detection Performance:
  Faces detected: {self.faces_detected} / {self.total_samples} ({100*self.detection_rate:.1f}%)

📈 Classification Metrics:
  AUC-ROC: {self.auc_roc:.4f}
  AUC-PR:  {self.auc_pr:.4f}
  Accuracy: {self.accuracy:.4f}
  Precision: {self.precision:.4f}
  Recall: {self.recall:.4f}
  F1 Score: {self.f1:.4f}

⚙️ Threshold Optimization:
  Optimal threshold: {self.optimal_threshold:.3f}
  Youden Index: {self.youden_index:.4f}

📋 Confusion Matrix:
  True Negatives (TN):  {self.tn}
  False Positives (FP): {self.fp}
  False Negatives (FN): {self.fn}
  True Positives (TP):  {self.tp}
"""


# ═════════════════════════════════════════════════════════════════════════════
# DATASET LOADERS
# ═════════════════════════════════════════════════════════════════════════════

class DatasetLoader:
    """Load evaluation datasets."""
    
    def __init__(self, dataset_dir: str = "./dataset"):
        self.dataset_dir = Path(dataset_dir)
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    def load_dataset(
        self,
        dataset_name: str,
        split: str = "test",
        limit: Optional[int] = None
    ) -> List[EvaluationSample]:
        """
        Load dataset samples.
        
        Args:
            dataset_name: "FF++", "Celeb-DF", "DFDC", "WildDeepfake", or custom
            split: "train", "test", "val"
            limit: Max samples to load (for testing)
        
        Returns:
            List[EvaluationSample]
        """
        log.info(f"Loading {dataset_name}/{split}...")
        
        samples = []
        
        if dataset_name.lower() in ["ff++", "faceforensics"]:
            samples = self._load_ffpp(split, limit)
        elif dataset_name.lower() in ["celeb-df", "celebdf"]:
            samples = self._load_celebdf(split, limit)
        elif dataset_name.lower() == "dfdc":
            samples = self._load_dfdc(split, limit)
        elif dataset_name.lower() == "wilddeepfake":
            samples = self._load_wilddeepfake(split, limit)
        else:
            # Generic folder-based loading
            samples = self._load_generic_folder(dataset_name, limit)
        
        log.info(f"Loaded {len(samples)} samples from {dataset_name}")
        return samples
    
    def _load_ffpp(self, split: str = "test", limit: Optional[int] = None) -> List[EvaluationSample]:
        """Load FaceForensics++."""
        samples = []
        
        # FF++ has: manipulated_sequences, original_sequences
        # For simplicity, treat all in test/ as test set
        
        dataset_path = self.dataset_dir / "FF++" / split
        if not dataset_path.exists():
            log.warning(f"FF++ path not found: {dataset_path}")
            return samples
        
        # Real samples
        real_dir = self.dataset_dir / "original" / split
        if real_dir.exists():
            for img_file in sorted(real_dir.glob("**/*.jpg"))[:limit]:
                samples.append(EvaluationSample(
                    image_path=str(img_file),
                    label=0,
                    dataset="FF++"
                ))
        
        # Fake samples (from any manipulation folder)
        for method_dir in dataset_path.glob("*/"):
            if method_dir.is_dir():
                for img_file in sorted(method_dir.glob("**/*.jpg"))[:limit]:
                    samples.append(EvaluationSample(
                        image_path=str(img_file),
                        label=1,
                        dataset="FF++"
                    ))
        
        return samples[:limit]
    
    def _load_celebdf(self, split: str = "test", limit: Optional[int] = None) -> List[EvaluationSample]:
        """Load Celeb-DF v2."""
        samples = []
        
        # Structure: Celeb-DF/Celeb-real, Celeb-DF/YouTube-real, Celeb-DF/Celeb-synthesis
        
        dataset_path = self.dataset_dir / "celeba"
        if not dataset_path.exists():
            log.warning(f"Celeb-DF path not found: {dataset_path}")
            return samples
        
        # Real videos
        real_dirs = [dataset_path / split / "Celeb-real",
                     dataset_path / split / "YouTube-real"]
        
        for real_dir in real_dirs:
            if real_dir.exists():
                for img_file in sorted(real_dir.glob("**/*.jpg"))[:limit]:
                    samples.append(EvaluationSample(
                        image_path=str(img_file),
                        label=0,
                        dataset="Celeb-DF"
                    ))
        
        # Fake samples
        fake_dir = dataset_path / split / "Celeb-synthesis"
        if fake_dir.exists():
            for img_file in sorted(fake_dir.glob("**/*.jpg"))[:limit]:
                samples.append(EvaluationSample(
                    image_path=str(img_file),
                    label=1,
                    dataset="Celeb-DF"
                ))
        
        return samples[:limit]
    
    def _load_dfdc(self, split: str = "test", limit: Optional[int] = None) -> List[EvaluationSample]:
        """Load DFDC (Deepfake Detection Challenge)."""
        samples = []
        
        dataset_path = self.dataset_dir / "deepdetect25" / split
        if not dataset_path.exists():
            log.warning(f"DFDC path not found: {dataset_path}")
            return samples
        
        # DFDC: folder structure is video_folders/frames
        for video_dir in sorted(dataset_path.glob("*"))[:limit]:
            if video_dir.is_dir():
                # Infer label from filename or metadata
                # For now, load all as unknown and rely on external labeling
                for img_file in sorted(video_dir.glob("*.jpg"))[:10]:  # 10 frames per video
                    label = 1 if "fake" in str(video_dir).lower() else 0
                    samples.append(EvaluationSample(
                        image_path=str(img_file),
                        label=label,
                        dataset="DFDC"
                    ))
        
        return samples[:limit]
    
    def _load_wilddeepfake(self, split: str = "test", limit: Optional[int] = None) -> List[EvaluationSample]:
        """Load WildDeepfake dataset."""
        samples = []
        
        dataset_path = self.dataset_dir / "wilddeepfake" / split
        if not dataset_path.exists():
            log.warning(f"WildDeepfake path not found: {dataset_path}")
            return samples
        
        # Real
        real_dir = dataset_path / "real"
        if real_dir.exists():
            for img_file in sorted(real_dir.glob("**/*.jpg"))[:limit]:
                samples.append(EvaluationSample(
                    image_path=str(img_file),
                    label=0,
                    dataset="WildDeepfake"
                ))
        
        # Fake
        fake_dir = dataset_path / "fake"
        if fake_dir.exists():
            for img_file in sorted(fake_dir.glob("**/*.jpg"))[:limit]:
                samples.append(EvaluationSample(
                    image_path=str(img_file),
                    label=1,
                    dataset="WildDeepfake"
                ))
        
        return samples[:limit]
    
    def _load_generic_folder(self, folder_path: str, limit: Optional[int] = None) -> List[EvaluationSample]:
        """Load from generic folder structure (real/ and fake/ subfolders)."""
        samples = []
        
        root = self.dataset_dir / folder_path if not Path(folder_path).is_absolute() else Path(folder_path)
        
        if not root.exists():
            log.warning(f"Dataset path not found: {root}")
            return samples
        
        # Real samples
        real_dir = root / "real" if (root / "real").exists() else root / "test" / "real"
        if real_dir.exists():
            for img_file in sorted(real_dir.glob("**/*"))[:limit]:
                if img_file.suffix.lower() in self.supported_formats:
                    samples.append(EvaluationSample(
                        image_path=str(img_file),
                        label=0,
                        dataset=folder_path
                    ))
        
        # Fake samples
        fake_dir = root / "fake" if (root / "fake").exists() else root / "test" / "fake"
        if fake_dir.exists():
            for img_file in sorted(fake_dir.glob("**/*"))[:limit]:
                if img_file.suffix.lower() in self.supported_formats:
                    samples.append(EvaluationSample(
                        image_path=str(img_file),
                        label=1,
                        dataset=folder_path
                    ))
        
        return samples[:limit]


# ═════════════════════════════════════════════════════════════════════════════
# TEXTURE AGENT EVALUATOR
# ═════════════════════════════════════════════════════════════════════════════

class TextureAgentEvaluator:
    """Evaluate TextureAgent across datasets."""
    
    def __init__(
        self,
        face_detector_backend: str = "mediapipe",
        output_dir: str = "./evaluation_results"
    ):
        self.agent = TextureAgent()
        self.face_detector = FaceDetector(backend=face_detector_backend)
        self.dataset_loader = DatasetLoader()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        log.info(f"TextureAgentEvaluator initialized. Output: {self.output_dir}")
    
    def evaluate_dataset(
        self,
        dataset_name: str,
        split: str = "test",
        limit: Optional[int] = None,
        save_details: bool = True
    ) -> EvaluationMetrics:
        """
        Evaluate TextureAgent on a dataset.
        
        Args:
            dataset_name: Dataset identifier
            split: "train", "test", or "val"
            limit: Max samples to evaluate
            save_details: Save per-sample results
        
        Returns:
            EvaluationMetrics
        """
        log.info(f"Starting evaluation of {dataset_name}/{split}...")
        
        # Load samples
        samples = self.dataset_loader.load_dataset(dataset_name, split, limit)
        
        if not samples:
            log.warning(f"No samples loaded for {dataset_name}")
            return EvaluationMetrics(dataset=dataset_name)
        
        # Process each sample
        predictions = []
        ground_truth = []
        detailed_results = []
        zone_stats = defaultdict(lambda: defaultdict(list))
        
        for idx, sample in enumerate(samples):
            log.info(f"[{idx+1}/{len(samples)}] Processing {sample.image_path}...")
            
            try:
                # Load image
                img = Image.open(sample.image_path).convert('RGB')
                img_array = np.array(img)
                
                # Detect faces
                bboxes = self.face_detector.detect(img_array)
                
                if not bboxes:
                    sample.error = "No faces detected"
                    log.warning(f"No faces detected in {sample.image_path}")
                    continue
                
                # Use largest face
                sample.face_bbox = max(bboxes, key=lambda b: b.area())
                sample.detection_method = self.face_detector.backend
                
                # Analyze texture
                result = self.agent.analyze(img, sample.face_bbox)
                sample.texture_result = result
                
                # Collect predictions
                predictions.append(result.texture_fake_probability)
                ground_truth.append(sample.label)
                
                # Zone statistics
                for zone_name, zone_result in result.zone_results.items():
                    zone_stats[zone_name]['emd'].append(zone_result.emd_score)
                    zone_stats[zone_name]['lbp'].append(zone_result.lbp_uniformity)
                    zone_stats[zone_name]['npr'].append(zone_result.npr_residual)
                    zone_stats[zone_name]['risk'].append(zone_result.risk_level)
                
                detailed_results.append(sample)
            
            except Exception as e:
                log.error(f"Error processing {sample.image_path}: {e}")
                sample.error = str(e)
                traceback.print_exc()
                continue
        
        # Compute metrics
        if not predictions:
            log.error(f"No successful predictions for {dataset_name}")
            return EvaluationMetrics(dataset=dataset_name)
        
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        
        # ─ Classification metrics ───────────────────────────────────────────
        auc_roc = roc_auc_score(ground_truth, predictions)
        auc_pr = average_precision_score(ground_truth, predictions)
        
        # Optimal threshold (Youden)
        fpr, tpr, thresholds = roc_curve(ground_truth, predictions)
        youden = tpr - fpr
        optimal_idx = np.argmax(youden)
        optimal_threshold = thresholds[optimal_idx]
        
        # Predictions at optimal threshold
        pred_binary = (predictions >= optimal_threshold).astype(int)
        
        accuracy = accuracy_score(ground_truth, pred_binary)
        precision = precision_score(ground_truth, pred_binary, zero_division=0)
        recall = recall_score(ground_truth, pred_binary, zero_division=0)
        f1 = f1_score(ground_truth, pred_binary, zero_division=0)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(ground_truth, pred_binary, labels=[0, 1]).ravel()
        
        # ─ Zone statistics ─────────────────────────────────────────────────
        zone_summary = {}
        for zone_name, metrics in zone_stats.items():
            zone_summary[zone_name] = {
                'avg_emd': float(np.mean(metrics['emd'])) if metrics['emd'] else 0,
                'avg_lbp': float(np.mean(metrics['lbp'])) if metrics['lbp'] else 0.5,
                'avg_npr': float(np.mean(metrics['npr'])) if metrics['npr'] else 0,
                'critical_count': sum(1 for r in metrics['risk'] if r == 'critical'),
                'elevated_count': sum(1 for r in metrics['risk'] if r == 'elevated'),
            }
        
        # ─ Create metrics object ────────────────────────────────────────────
        metrics = EvaluationMetrics(
            dataset=dataset_name,
            total_samples=len(samples),
            real_samples=np.sum(ground_truth == 0),
            fake_samples=np.sum(ground_truth == 1),
            faces_detected=len(predictions),
            detection_rate=len(predictions) / len(samples),
            auc_roc=auc_roc,
            auc_pr=auc_pr,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            optimal_threshold=optimal_threshold,
            youden_index=youden[optimal_idx],
            tn=int(tn),
            fp=int(fp),
            fn=int(fn),
            tp=int(tp),
            zone_stats=zone_summary
        )
        
        # ─ Save results ─────────────────────────────────────────────────────
        if save_details:
            self._save_results(metrics, detailed_results, fpr, tpr, predictions, ground_truth)
        
        log.info(f"Evaluation complete for {dataset_name}")
        print(metrics)
        
        return metrics
    
    def _save_results(
        self,
        metrics: EvaluationMetrics,
        samples: List[EvaluationSample],
        fpr: np.ndarray,
        tpr: np.ndarray,
        predictions: np.ndarray,
        ground_truth: np.ndarray
    ):
        """Save evaluation results and visualizations."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = metrics.dataset.replace("/", "_").replace("\\", "_")
        output_subdir = self.output_dir / f"{dataset_name}_{timestamp}"
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        metrics_file = output_subdir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)
        log.info(f"Metrics saved: {metrics_file}")
        
        # Save detailed results
        details_file = output_subdir / "detailed_results.json"
        with open(details_file, 'w') as f:
            json.dump([s.to_dict() for s in samples], f, indent=2)
        log.info(f"Detailed results saved: {details_file}")
        
        # Plot ROC curve (if matplotlib available)
        if HAS_MATPLOTLIB:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.plot(fpr, tpr, label=f'ROC (AUC={metrics.auc_roc:.4f})')
            ax.plot([0, 1], [0, 1], 'k--', label='Random')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC Curve — {metrics.dataset}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            roc_file = output_subdir / "roc_curve.png"
            fig.savefig(roc_file, dpi=150, bbox_inches='tight')
            plt.close(fig)
            log.info(f"ROC curve saved: {roc_file}")
            
            # Plot confusion matrix
            cm = np.array([[metrics.tn, metrics.fp], [metrics.fn, metrics.tp]])
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True Label')
            ax.set_title(f'Confusion Matrix — {metrics.dataset}')
            
            cm_file = output_subdir / "confusion_matrix.png"
            fig.savefig(cm_file, dpi=150, bbox_inches='tight')
            plt.close(fig)
            log.info(f"Confusion matrix saved: {cm_file}")
        else:
            log.warning("matplotlib not installed — skipping visualizations")
        
        # Summary report
        report_file = output_subdir / "report.txt"
        with open(report_file, 'w') as f:
            f.write(str(metrics))
            f.write("\n\n📍 Zone Statistics:\n")
            for zone_name, stats in metrics.zone_stats.items():
                f.write(f"\n{zone_name}:\n")
                for key, value in stats.items():
                    f.write(f"  {key}: {value}\n")
        
        log.info(f"Report saved: {report_file}")
    
    def evaluate_all_datasets(
        self,
        dataset_names: List[str] = None,
        limit_per_dataset: Optional[int] = None
    ) -> Dict[str, EvaluationMetrics]:
        """Evaluate multiple datasets."""
        
        if dataset_names is None:
            dataset_names = [
                "FF++",
                "Celeb-DF",
                "DFDC",
            ]
        
        all_metrics = {}
        
        for dataset_name in dataset_names:
            try:
                metrics = self.evaluate_dataset(
                    dataset_name,
                    split="test",
                    limit=limit_per_dataset,
                    save_details=True
                )
                all_metrics[dataset_name] = metrics
            except Exception as e:
                log.error(f"Failed to evaluate {dataset_name}: {e}")
                traceback.print_exc()
        
        # Save summary
        summary_file = self.output_dir / "evaluation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(
                {name: m.to_dict() for name, m in all_metrics.items()},
                f,
                indent=2
            )
        
        log.info(f"Evaluation summary saved: {summary_file}")
        
        return all_metrics


# ═════════════════════════════════════════════════════════════════════════════
# CLI INTERFACE
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate TextureAgent on deepfake detection datasets"
    )
    parser.add_argument(
        "--dataset",
        default="FF++",
        help="Dataset name (FF++, Celeb-DF, DFDC, WildDeepfake, or path)"
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "test", "val"],
        help="Dataset split"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples per dataset"
    )
    parser.add_argument(
        "--face-detector",
        default="mediapipe",
        choices=["mediapipe", "dlib"],
        help="Face detection backend"
    )
    parser.add_argument(
        "--output-dir",
        default="./evaluation_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help="Evaluate on all available datasets"
    )
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = TextureAgentEvaluator(
        face_detector_backend=args.face_detector,
        output_dir=args.output_dir
    )
    
    # Run evaluation
    if args.all_datasets:
        metrics_dict = evaluator.evaluate_all_datasets(limit_per_dataset=args.limit)
        print("\n" + "="*80)
        print("SUMMARY ACROSS ALL DATASETS")
        print("="*80)
        for dataset_name, metrics in metrics_dict.items():
            print(f"{dataset_name}: AUC={metrics.auc_roc:.4f}, F1={metrics.f1:.4f}")
    else:
        metrics = evaluator.evaluate_dataset(
            args.dataset,
            split=args.split,
            limit=args.limit,
            save_details=True
        )


if __name__ == "__main__":
    main()
