#!/usr/bin/env python3
"""
Demo: Train and test the ML-based texture classifier.

This shows:
  1. Create synthetic training data
  2. Train a Logistic Regression model
  3. Save the model
  4. Use it to make predictions
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from agents.texture_classifier import TextureClassifier


def main():
    print("=" * 70)
    print("ML Texture Classifier Demo")
    print("=" * 70)
    print()
    
    # Step 1: Create synthetic training data
    print("Step 1: Creating synthetic training data...")
    print("  (In production, use real labeled deepfake datasets)")
    print()
    X_train, y_train = TextureClassifier.create_synthetic_training_data(
        n_real=200, n_fake=200
    )
    print(f"✓ Generated {len(X_train)} samples")
    print(f"  - Real faces:  {(y_train == 0).sum()}")
    print(f"  - Fake faces:  {(y_train == 1).sum()}")
    print()
    
    # Step 2: Train the classifier
    print("Step 2: Training Logistic Regression classifier...")
    print()
    clf = TextureClassifier()
    metrics = clf.train(X_train, y_train, test_split=0.2)
    
    print("Training Results:")
    print(f"  Train accuracy: {metrics['train_accuracy']:.4f}")
    print(f"  Test accuracy:  {metrics['test_accuracy']:.4f}")
    print(f"  Precision:      {metrics['precision']:.4f}")
    print(f"  Recall:         {metrics['recall']:.4f}")
    print()
    
    # Step 3: Save the model
    print("Step 3: Saving model...")
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "texture_classifier.pkl")
    
    clf.save(model_path)
    print()
    
    # Step 4: Load and test predictions
    print("Step 4: Loading model and making predictions...")
    print()
    
    clf_loaded = TextureClassifier.load(model_path)
    
    # Test on various examples
    test_cases = [
        ("Authentic face", {
            "jaw_emd": 0.08,
            "neck_emd": 0.10,
            "cheek_emd": 0.12,
            "lbp_uniformity": 0.82,
            "zone_scores": {
                "forehead": 0.10,
                "cheek_L": 0.12,
                "cheek_R": 0.13,
                "jaw": 0.15,
                "nose": 0.08
            },
            "multi_scale_consistency": 0.08,
        }),
        ("Deepfake face", {
            "jaw_emd": 0.52,
            "neck_emd": 0.48,
            "cheek_emd": 0.38,
            "lbp_uniformity": 0.52,
            "zone_scores": {
                "forehead": 0.40,
                "cheek_L": 0.50,
                "cheek_R": 0.52,
                "jaw": 0.70,
                "nose": 0.38
            },
            "multi_scale_consistency": 0.62,
        }),
        ("Ambiguous", {
            "jaw_emd": 0.28,
            "neck_emd": 0.25,
            "cheek_emd": 0.22,
            "lbp_uniformity": 0.65,
            "zone_scores": {
                "forehead": 0.25,
                "cheek_L": 0.28,
                "cheek_R": 0.30,
                "jaw": 0.40,
                "nose": 0.20
            },
            "multi_scale_consistency": 0.35,
        }),
    ]
    
    print("Sample Predictions:")
    print()
    for name, texture_dict in test_cases:
        pred = clf_loaded.predict(texture_dict)
        decision = "🔴 DEEPFAKE" if pred > 0.70 else "🟢 AUTHENTIC" if pred < 0.35 else "🟡 UNCERTAIN"
        print(f"  {name:20s}: {pred:.4f}  {decision}")
    
    print()
    print("=" * 70)
    print("✓ Demo complete!")
    print()
    print("How to use in production:")
    print("  1. Collect labeled real & fake face data")
    print("  2. Run: python train_texture_classifier.py /path/real /path/fake")
    print("  3. The model saves to models/texture_classifier.pkl")
    print("  4. The texture agent automatically loads and uses it")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"Error: {e}")
        print("\nPlease install required dependencies:")
        print("  pip install scikit-learn numpy scikit-image opencv-python pydantic")
        sys.exit(1)
