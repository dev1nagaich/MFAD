#!/usr/bin/env python3
"""
Train the texture ML classifier on labeled deepfake data.

Usage:
    python train_texture_classifier.py /path/to/real/images /path/to/fake/images
    
    Or with default synthetic data:
    python train_texture_classifier.py
"""

import sys
import os
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from agents.texture import run_texture_agent
from agents.texture_classifier import TextureClassifier


def collect_features_from_folder(folder_path: str, label: int) -> tuple:
    """
    Extract texture features from all images in a folder.
    
    Args:
        folder_path: directory with images
        label: 0 for real, 1 for fake
        
    Returns:
        (feature_vectors, labels) arrays
    """
    features_list = []
    labels_list = []
    
    folder = Path(folder_path)
    image_files = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
    
    if not image_files:
        print(f"⚠ No images found in {folder_path}")
        return np.array([]), np.array([])
    
    print(f"\nProcessing {len(image_files)} images from {folder_path}")
    
    for i, img_path in enumerate(image_files):
        try:
            # Dummy bbox — adjust based on your preprocessing
            bbox = [50, 50, img_path.stat().st_size % 400 + 200, 
                   img_path.stat().st_size % 400 + 200]
            
            result = run_texture_agent(str(img_path), bbox)
            texture_dict = result.model_dump()
            
            # Extract features using classifier
            clf = TextureClassifier()
            features = clf.extract_features(texture_dict)
            features_list.append(features)
            labels_list.append(label)
            
            if (i + 1) % 10 == 0:
                print(f"  ✓ {i + 1}/{len(image_files)} processed")
                
        except Exception as e:
            print(f"  ✗ Error processing {img_path.name}: {e}")
            continue
    
    return np.array(features_list, dtype=np.float32), np.array(labels_list, dtype=np.int32)


def main():
    print("=" * 70)
    print("Texture Classifier Training")
    print("=" * 70)
    print()
    
    # Check if custom paths provided
    if len(sys.argv) > 2:
        real_folder = sys.argv[1]
        fake_folder = sys.argv[2]
        
        print(f"Real folder: {real_folder}")
        print(f"Fake folder: {fake_folder}")
        print()
        
        # Collect features
        X_real, y_real = collect_features_from_folder(real_folder, label=0)
        X_fake, y_fake = collect_features_from_folder(fake_folder, label=1)
        
        if len(X_real) == 0 or len(X_fake) == 0:
            print("✗ No features collected. Check your input folders.")
            sys.exit(1)
        
        X_train = np.vstack([X_real, X_fake])
        y_train = np.hstack([y_real, y_fake])
        
        print(f"\n✓ Collected {len(X_train)} samples total")
        print(f"  Real: {(y_train == 0).sum()}")
        print(f"  Fake: {(y_train == 1).sum()}")
        
    else:
        # Use synthetic data
        print("No custom data provided. Generating synthetic training data...")
        X_train, y_train = TextureClassifier.create_synthetic_training_data(
            n_real=100, n_fake=100
        )
        print(f"✓ Generated {len(X_train)} synthetic samples")
        print(f"  Real: {(y_train == 0).sum()}")
        print(f"  Fake: {(y_train == 1).sum()}")
    
    print()
    
    # Train
    clf = TextureClassifier()
    print("Training classifier...")
    metrics = clf.train(X_train, y_train, test_split=0.2)
    
    print()
    print("Training Results:")
    print(f"  Train accuracy: {metrics['train_accuracy']:.4f}")
    print(f"  Test accuracy:  {metrics['test_accuracy']:.4f}")
    print(f"  Precision:      {metrics['precision']:.4f}")
    print(f"  Recall:         {metrics['recall']:.4f}")
    print(f"  Samples: {metrics['n_train']} train, {metrics['n_test']} test")
    print()
    
    # Save
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "texture_classifier.pkl")
    
    clf.save(model_path)
    print()
    print("=" * 70)
    print(f"✓ Model training complete!")
    print(f"  Saved to: {model_path}")
    print(f"\nThe texture agent will now use this model for predictions.")
    print("=" * 70)


if __name__ == "__main__":
    main()
