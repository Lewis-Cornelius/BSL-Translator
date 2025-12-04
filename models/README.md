# Model Download Instructions

This directory contains trained ML models for BSL gesture recognition.

## Download the Model

The trained model file (`bsl_classifier.pkl`) is too large for git (64MB).

**Download it from:**
- [Google Drive](your-link-here)
- [Dropbox](your-link-here)

Or use the download script:
```bash
python scripts/download_model.py
```

## Expected Files

After downloading, this directory should contain:
- `bsl_classifier.pkl` - Main BSL gesture classification model (64MB)
- `.gitkeep` - Placeholder to keep directory in git

## Model Details

- **Type:** Random Forest Classifier
- **Features:** 84 (42 hand landmarks × 2 coordinates)
- **Classes:** 26 (A-Z BSL letters)
- **Accuracy:** ~95% on test set
- **Framework:** scikit-learn
