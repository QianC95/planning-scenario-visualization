# Data availability and reproduction limits

The paper's Data Availability Statement offers metadata and evaluation scripts on reasonable request and says that third-party licensing prevents redistribution of street-level images. This repository makes the metadata and evaluation code directly accessible. It does not contain raw panoramas, before/after crops, condition maps derived from them, or the generated images used in the reported metrics.

The committed results CSVs transcribe Tables 3 and 4. They are **published results**, not output of a rerun from this package. Recomputing the tables requires authorized access to the exact 2,015 test image pairs and corresponding generated outputs for all 11 settings. Recreating those outputs also requires the model implementations, weights or services, and run configurations. The metadata and evaluation code support that work but cannot substitute for the restricted inputs.

The images and condition maps must stay outside the repository. Run the evaluator by passing their local directories as command arguments. The code does not fetch Street View imagery or call commercial image-generation services.
