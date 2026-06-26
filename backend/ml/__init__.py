# ML module for the anemia screening pipeline.
"""
HemaScan ML pipeline package.

Sub-modules:
- config:        Paths, class lists, anemia-positive mappings.
- quality:       Blur / brightness quality checks on uploaded images.
- preprocessing: Resize, normalize, and Test-Time-Augmentation generators.
- model_loader:  Lazy-load MobileNetV2 .h5 weights for eye and nail.
- inference:     Run a model with TTA + return calibrated probabilities.
- gradcam:       Real Grad-CAM using tf.GradientTape on the last conv layer.
- fusion:        Noisy-OR + optional logistic-regression meta-learner.
- pipeline:      Top-level entry-point used by FastAPI server.
"""
