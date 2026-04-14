"""
agents/texture_classifier.py — ML-based Texture Anomaly Classifier

Learns decision boundaries for deepfake detection from labeled texture features.
Replaces hardcoded thresholds with a trained Logistic Regression model.

Usage:
    # Train the model (one-time setup with labeled data)
    classifier = TextureClassifier()
    classifier.train(feature_vectors, labels)  # labels: 0=real, 1=fake
    classifier.save("models/texture_model.pkl")
    
    # Use the trained model
    classifier = TextureClassifier.load("models/texture_model.pkl")
    anomaly_score = classifier.predict(features)
"""

import os
import pickle
import warnings
from typing import Tuple, List, Optional

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. Falling back to threshold-based detection.")


class TextureClassifier:
    """
    ML classifier that learns to predict deepfake probability from texture features.
    
    Features used:
      - jaw_emd: cheek-to-jaw boundary distance
      - neck_emd: jaw-to-neck boundary distance
      - cheek_emd: left-right cheek symmetry distance
      - lbp_uniformity: skin texture uniformity
      - mean_zone_score: average per-zone anomaly
      - multi_scale_consistency: scale consistency variance
    """
    
    def __init__(self):
        """Initialize the classifier with no trained model."""
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.feature_names = [
            "jaw_emd",
            "neck_emd", 
            "cheek_emd",
            "lbp_uniformity",
            "mean_zone_score",
            "multi_scale_consistency"
        ]
    
    def extract_features(self, texture_output: dict) -> np.ndarray:
        """
        Extract a feature vector from TextureOutput dict.
        
        Args:
            texture_output: dict with keys from TextureOutput.model_dump()
            
        Returns:
            1D array of shape (6,) with normalized features
        """
        zone_scores = texture_output.get("zone_scores", {})
        mean_zone = np.mean(list(zone_scores.values())) if zone_scores else 0.5
        
        features = np.array([
            texture_output.get("jaw_emd", 0.0),
            texture_output.get("neck_emd", 0.0),
            texture_output.get("cheek_emd", 0.0),
            texture_output.get("lbp_uniformity", 0.5),
            mean_zone,
            texture_output.get("multi_scale_consistency", 0.5),
        ], dtype=np.float32)
        
        return features
    
    def train(self, feature_vectors: np.ndarray, labels: np.ndarray,
              test_split: float = 0.2) -> dict:
        """
        Train the classifier on labeled feature vectors.
        
        Args:
            feature_vectors: (N, 6) array of texture features
            labels: (N,) array where 0=real, 1=fake
            test_split: fraction of data to use for validation
            
        Returns:
            dict with training metrics: {"accuracy", "precision", "recall"}
        """
        if not _SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for training. Install with: pip install scikit-learn")
        
        # Shuffle and split
        shuffled_idx = np.random.permutation(len(feature_vectors))
        split_idx = int(len(feature_vectors) * (1 - test_split))
        
        train_idx = shuffled_idx[:split_idx]
        test_idx = shuffled_idx[split_idx:]
        
        X_train = feature_vectors[train_idx]
        y_train = labels[train_idx]
        X_test = feature_vectors[test_idx]
        y_test = labels[test_idx]
        
        # Standardize features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train logistic regression
        self.model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'  # Handle class imbalance
        )
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Evaluate
        train_acc = self.model.score(X_train_scaled, y_train)
        test_acc = self.model.score(X_test_scaled, y_test)
        
        y_pred = self.model.predict(X_test_scaled)
        
        # Metrics
        tp = np.sum((y_pred == 1) & (y_test == 1))
        fp = np.sum((y_pred == 1) & (y_test == 0))
        fn = np.sum((y_pred == 0) & (y_test == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        metrics = {
            "train_accuracy": float(train_acc),
            "test_accuracy": float(test_acc),
            "precision": float(precision),
            "recall": float(recall),
            "n_train": len(X_train),
            "n_test": len(X_test),
        }
        
        return metrics
    
    def predict(self, texture_output: dict) -> float:
        """
        Predict anomaly score (deepfake probability).
        
        Args:
            texture_output: dict from TextureOutput.model_dump()
            
        Returns:
            float in [0.0, 1.0] — probability of being a deepfake
        """
        if not self.is_trained:
            raise ValueError(
                "Model not trained. Call .train() first or load a saved model with .load()"
            )
        
        features = self.extract_features(texture_output)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Get probability of class 1 (fake)
        prob_fake = float(self.model.predict_proba(features_scaled)[0, 1])
        
        return prob_fake
    
    def predict_batch(self, texture_outputs: List[dict]) -> np.ndarray:
        """
        Predict on multiple texture outputs.
        
        Args:
            texture_outputs: list of dicts from TextureOutput.model_dump()
            
        Returns:
            (N,) array of anomaly scores
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        features = np.array([self.extract_features(t) for t in texture_outputs])
        features_scaled = self.scaler.transform(features)
        
        probs = self.model.predict_proba(features_scaled)
        return probs[:, 1]  # Probability of fake
    
    def save(self, filepath: str) -> None:
        """Save trained model and scaler to disk."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        
        state = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(state, f)
        
        print(f"✓ Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> "TextureClassifier":
        """Load a trained model from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, "rb") as f:
            state = pickle.load(f)
        
        classifier = cls()
        classifier.model = state["model"]
        classifier.scaler = state["scaler"]
        classifier.feature_names = state["feature_names"]
        classifier.is_trained = True
        
        print(f"✓ Model loaded from {filepath}")
        return classifier
    
    @staticmethod
    def create_synthetic_training_data(n_real: int = 100, n_fake: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic training data for model initialization.
        
        In production, use real labeled data from deepfake datasets.
        This is for demo/testing purposes.
        
        Returns:
            (feature_vectors, labels) where labels: 0=real, 1=fake
        """
        features_list = []
        labels_list = []
        
        # Synthetic REAL faces
        # Real faces: low boundaries, high uniformity, low zone variance
        for _ in range(n_real):
            jaw_emd = np.random.normal(0.10, 0.05)
            neck_emd = np.random.normal(0.08, 0.04)
            cheek_emd = np.random.normal(0.12, 0.06)
            lbp_unif = np.random.normal(0.80, 0.10)
            zone_score = np.random.normal(0.15, 0.08)
            scale_consist = np.random.normal(0.10, 0.05)
            
            features_list.append([
                np.clip(jaw_emd, 0, 1),
                np.clip(neck_emd, 0, 1),
                np.clip(cheek_emd, 0, 1),
                np.clip(lbp_unif, 0, 1),
                np.clip(zone_score, 0, 1),
                np.clip(scale_consist, 0, 1),
            ])
            labels_list.append(0)
        
        # Synthetic FAKE faces
        # Deepfakes: high boundaries (seams), lower uniformity, high zone variance
        for _ in range(n_fake):
            jaw_emd = np.random.normal(0.55, 0.15)
            neck_emd = np.random.normal(0.50, 0.15)
            cheek_emd = np.random.normal(0.35, 0.12)
            lbp_unif = np.random.normal(0.50, 0.15)
            zone_score = np.random.normal(0.55, 0.12)
            scale_consist = np.random.normal(0.60, 0.15)
            
            features_list.append([
                np.clip(jaw_emd, 0, 1),
                np.clip(neck_emd, 0, 1),
                np.clip(cheek_emd, 0, 1),
                np.clip(lbp_unif, 0, 1),
                np.clip(zone_score, 0, 1),
                np.clip(scale_consist, 0, 1),
            ])
            labels_list.append(1)
        
        return np.array(features_list, dtype=np.float32), np.array(labels_list, dtype=np.int32)


if __name__ == "__main__":
    """Demo: train on synthetic data and test prediction."""
    
    print("=" * 70)
    print("Texture Classifier Demo")
    print("=" * 70)
    print()
    
    # Create classifier
    clf = TextureClassifier()
    print(f"✓ Classifier created")
    print(f"  Features: {clf.feature_names}")
    print()
    
    # Generate synthetic training data
    print("Generating synthetic training data...")
    X_train, y_train = TextureClassifier.create_synthetic_training_data(n_real=100, n_fake=100)
    print(f"  X: {X_train.shape}, y: {y_train.shape}")
    print(f"  Real samples: {(y_train == 0).sum()}, Fake samples: {(y_train == 1).sum()}")
    print()
    
    # Train
    print("Training model...")
    metrics = clf.train(X_train, y_train, test_split=0.2)
    print(f"  Train accuracy: {metrics['train_accuracy']:.3f}")
    print(f"  Test accuracy:  {metrics['test_accuracy']:.3f}")
    print(f"  Precision:      {metrics['precision']:.3f}")
    print(f"  Recall:         {metrics['recall']:.3f}")
    print()
    
    # Save
    model_path = "models/texture_classifier.pkl"
    clf.save(model_path)
    print()
    
    # Load and test
    print("Loading model and testing...")
    clf_loaded = TextureClassifier.load(model_path)
    
    # Test on sample texture output
    sample_texture = {
        "jaw_emd": 0.08,
        "neck_emd": 0.10,
        "cheek_emd": 0.12,
        "lbp_uniformity": 0.82,
        "zone_scores": {"forehead": 0.1, "cheek_L": 0.12, "cheek_R": 0.14, "jaw": 0.18, "nose": 0.08},
        "multi_scale_consistency": 0.08,
    }
    
    pred = clf_loaded.predict(sample_texture)
    print(f"  Sample real face prediction: {pred:.4f} (expect ~0.0-0.2)")
    print()
    
    print("=" * 70)
    print("✓ Demo complete")
    print("=" * 70)
