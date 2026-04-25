"""
test_texture_agent.py — Unit Tests for TextureAgent

Pytest tests covering:
  • Basic initialization
  • Pydantic output schema validation
  • ML classifier training and inference
  • Feature extraction
  • Single image analysis
  • Error handling

Run with: pytest test_texture_agent.py -v
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from PIL import Image

from agents.texture_agent import (
    TextureAgent, BoundingBox, TextureAnalysisResult, ZoneScore
)
from texture_agent_evaluator import FaceDetector


class TestBoundingBox:
    """Test BoundingBox utilities."""
    
    def test_bbox_dimensions(self):
        bbox = BoundingBox(x1=10, y1=20, x2=110, y2=220)
        assert bbox.width() == 100
        assert bbox.height() == 200
        assert bbox.area() == 20000
    
    def test_bbox_expand(self):
        bbox = BoundingBox(x1=100, y1=100, x2=200, y2=200)
        expanded = bbox.expand(0.1)
        assert expanded.x1 < bbox.x1
        assert expanded.y1 < bbox.y1
        assert expanded.x2 > bbox.x2
        assert expanded.y2 > bbox.y2


class TestTextureAnalysisResult:
    """Test Pydantic output schema."""
    
    def test_result_creation(self):
        """Test creating TextureAnalysisResult."""
        zone = ZoneScore(
            zone_name="forehead",
            emd_score=0.1,
            lbp_uniformity=0.85,
            npr_residual=0.08,
            texture_variance=0.025,
            boundary_seam=None,
            color_delta_e=7.5,
            risk_level="normal"
        )
        
        result = TextureAnalysisResult(
            texture_fake_probability=0.35,
            is_fake=False,
            anomaly_score=0.35,
            jaw_emd=0.10,
            neck_emd=0.05,
            cheek_emd=0.08,
            lbp_uniformity=0.85,
            seam_detected=False,
            multi_scale_consistency=0.75,
            zone_results={"forehead": zone},
            zone_scores={"forehead": 0.1},
            gram_distances={},
            analyst_note="Test result",
            processing_notes=[]
        )
        
        assert result.texture_fake_probability == 0.35
        assert result.is_fake is False
        assert result.anomaly_score == 0.35
        assert len(result.zone_results) == 1
    
    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        zone = ZoneScore(
            zone_name="forehead",
            emd_score=0.1,
            lbp_uniformity=0.85,
            npr_residual=0.08,
            texture_variance=0.025,
            boundary_seam=None,
            color_delta_e=7.5,
            risk_level="normal"
        )
        
        result = TextureAnalysisResult(
            texture_fake_probability=0.35,
            is_fake=False,
            anomaly_score=0.35,
            jaw_emd=0.10,
            neck_emd=0.05,
            cheek_emd=0.08,
            lbp_uniformity=0.85,
            seam_detected=False,
            multi_scale_consistency=0.75,
            zone_results={"forehead": zone},
            zone_scores={"forehead": 0.1},
            gram_distances={},
            analyst_note="Test result",
            processing_notes=[]
        )
        
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert "texture_fake_probability" in result_dict
        assert "zone_results" in result_dict
        assert isinstance(result_dict["zone_results"], dict)


class TestTextureAgentInitialization:
    """Test TextureAgent initialization."""
    
    def test_init_cpu(self):
        agent = TextureAgent(device="cpu")
        assert agent.device == "cpu"
        assert agent.classifier is None
        assert agent.thresholds is not None
    
    def test_init_with_nonexistent_classifier(self):
        """Should initialize even if classifier path doesn't exist."""
        agent = TextureAgent(device="cpu", classifier_path="/nonexistent/path.pkl")
        assert agent.device == "cpu"
        assert agent.classifier is None


class TestFeatureExtraction:
    """Test ML feature extraction."""
    
    def test_extract_features(self):
        """Test feature vector extraction."""
        agent = TextureAgent()
        
        emd_scores = {
            'jaw_avg': 0.10,
            'neck_avg': 0.05,
            'cheek_L_avg': 0.08,
            'nose_avg': 0.07,
            'perioral_avg': 0.09,
        }
        
        npr_residuals = {
            'jaw_npr': 0.10,
            'neck_npr': 0.08,
            'nose_npr': 0.09,
        }
        
        zone_lbp = {
            'overall_uniformity': 0.85,
            'forehead_uniformity': 0.80,
            'nose_uniformity': 0.90,
            'perioral_uniformity': 0.75,
        }
        
        seam_score = 0.05
        color_deltas = {'forehead': 8.0, 'nose': 7.5, 'jaw': 9.0}
        gabor_vars = {'forehead': 0.025, 'nose': 0.028, 'jaw': 0.022}
        
        features = agent._extract_features(
            emd_scores, npr_residuals, zone_lbp,
            seam_score, color_deltas, gabor_vars
        )
        
        assert features.shape == (1, 14)
        assert np.all(np.isfinite(features))


class TestMLClassifier:
    """Test ML classifier training and inference."""
    
    def test_train_classifier(self):
        """Test training a simple classifier."""
        agent = TextureAgent()
        
        # Generate synthetic training data
        np.random.seed(42)
        n_samples = 100
        
        # Real images (label=0): lower EMD, higher LBP uniformity
        X_real = np.random.normal(
            loc=[0.06, 0.05, 0.85, 0.80, 0.08, 0.08, 0.026, 6.0, 0.05, 0.04, 0.88, 0.85, 0.08, 2.0],
            scale=0.02,
            size=(n_samples // 2, 14)
        )
        y_real = np.zeros(n_samples // 2)
        
        # Fake images (label=1): higher EMD, lower LBP uniformity
        X_fake = np.random.normal(
            loc=[0.18, 0.15, 0.65, 0.60, 0.18, 0.16, 0.010, 4.0, 0.20, 0.14, 0.62, 0.58, 0.19, 0.5],
            scale=0.02,
            size=(n_samples // 2, 14)
        )
        y_fake = np.ones(n_samples // 2)
        
        X = np.vstack([X_real, X_fake])
        y = np.hstack([y_real, y_fake])
        
        # Train
        metrics = agent.train_classifier(X, y, test_size=0.2)
        
        assert 'auc_roc' in metrics
        assert 'f1' in metrics
        assert 'accuracy' in metrics
        assert 0 <= metrics['auc_roc'] <= 1
        assert 0 <= metrics['accuracy'] <= 1
        assert agent.classifier is not None
    
    def test_save_and_load_classifier(self):
        """Test saving and loading classifier."""
        agent1 = TextureAgent()
        
        # Train classifier
        np.random.seed(42)
        X = np.random.randn(50, 14)
        y = np.random.randint(0, 2, 50)
        agent1.train_classifier(X, y, test_size=0.2)
        
        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_classifier.pkl"
            agent1.save_classifier(str(model_path))
            
            # Load in new agent
            agent2 = TextureAgent(classifier_path=str(model_path))
            assert agent2.classifier is not None
            assert agent2.scaler is not None
    
    def test_predict_with_classifier(self):
        """Test predictions with trained classifier."""
        agent = TextureAgent()
        
        # Train
        np.random.seed(42)
        X = np.random.randn(50, 14)
        y = np.random.randint(0, 2, 50)
        agent.train_classifier(X, y, test_size=0.2)
        
        # Predict
        test_feature = np.random.randn(1, 14)
        proba = agent._predict_with_classifier(test_feature)
        
        assert isinstance(proba, float)
        assert 0 <= proba <= 1


class TestTextureAgentAnalysis:
    """Test texture analysis on synthetic images."""
    
    def test_analyze_synthetic_image(self):
        """Test analyzing a synthetic face image."""
        agent = TextureAgent()
        
        # Create synthetic face image (256x256 RGB)
        np.random.seed(42)
        img_array = np.random.randint(100, 200, (256, 256, 3), dtype=np.uint8)
        
        # Add some structure (Gaussian blur)
        from scipy.ndimage import gaussian_filter
        for c in range(3):
            img_array[:, :, c] = gaussian_filter(img_array[:, :, c], sigma=5).astype(np.uint8)
        
        img = Image.fromarray(img_array, 'RGB')
        bbox = BoundingBox(x1=30, y1=30, x2=226, y2=226)
        
        # Analyze
        result = agent.analyze(img, bbox)
        
        # Verify output
        assert isinstance(result, TextureAnalysisResult)
        assert 0 <= result.texture_fake_probability <= 1
        assert isinstance(result.is_fake, bool)
        assert len(result.zone_results) == 7  # 7 zones
        assert all(zone in result.zone_results for zone in [
            'forehead', 'nose', 'cheek_L', 'cheek_R', 'perioral', 'jaw', 'neck'
        ])
    
    def test_analyze_with_real_test_image(self):
        """Test with real test image if available."""
        test_image_path = Path("test_images/fake/fake_041595.jpg")
        
        if not test_image_path.exists():
            pytest.skip("Test image not found")
        
        agent = TextureAgent()
        detector = FaceDetector(backend="opencv")
        
        # Load and detect
        import cv2
        image = cv2.imread(str(test_image_path))
        bboxes = detector.detect(image)
        
        if not bboxes:
            pytest.skip("No faces detected in test image")
        
        # Analyze
        pil_image = Image.open(str(test_image_path)).convert('RGB')
        result = agent.analyze(pil_image, bboxes[0])
        
        assert isinstance(result, TextureAnalysisResult)
        assert 0 <= result.texture_fake_probability <= 1
        assert len(result.zone_results) == 7


class TestErrorHandling:
    """Test error handling."""
    
    def test_invalid_bbox(self):
        """Test with invalid bounding box."""
        agent = TextureAgent()
        
        img = Image.new('RGB', (256, 256), color='red')
        bbox = BoundingBox(x1=200, y1=200, x2=210, y2=210)  # Too small
        
        result = agent.analyze(img, bbox)
        
        # Should return empty result
        assert result is not None
    
    def test_classifier_not_trained_fallback(self):
        """Test that weighted fusion works if classifier not trained."""
        agent = TextureAgent()
        assert agent.classifier is None
        
        # Create synthetic data
        np.random.seed(42)
        img_array = np.random.randint(100, 200, (256, 256, 3), dtype=np.uint8)
        from scipy.ndimage import gaussian_filter
        for c in range(3):
            img_array[:, :, c] = gaussian_filter(img_array[:, :, c], sigma=5).astype(np.uint8)
        
        img = Image.fromarray(img_array, 'RGB')
        bbox = BoundingBox(x1=30, y1=30, x2=226, y2=226)
        
        # Should still work with weighted fusion
        result = agent.analyze(img, bbox)
        assert isinstance(result, TextureAnalysisResult)
        assert 0 <= result.texture_fake_probability <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
