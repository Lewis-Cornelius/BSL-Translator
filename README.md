# BSL Translator - Real-Time British Sign Language Recognition

> Computer vision system translating BSL fingerspelling to text using MediaPipe + scikit-learn

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-28%20passing-brightgreen.svg)](.github/workflows/ci.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 🎯 Overview

Real-time BSL (British Sign Language) fingerspelling translator that:
- Detects hand landmarks using **MediaPipe**
- Classifies BSL letters with a custom **Random Forest model**
- Autocorrects to valid English words using **NLTK**
- Supports both **Raspberry Pi deployment** and **standalone demo mode**

**Built to demonstrate:** Real-time CV pipeline, ML deployment, cloud integration, and embedded systems.

---

## ⚡ Quick Demo (2 Minutes)

No hardware needed! Test the system on your laptop:

```bash
git clone https://github.com/Lewis-Cornelius/BSL-Translator
cd BSL-Translator
pip install -r requirements-demo.txt
python src/demo_standalone.py
```

A webcam window will open with real-time BSL translation. Make BSL letters (A-Z) and watch the translation appear!

---

## 🏗️ Architecture

```
┌──────────┐      ┌──────────┐      ┌───────────┐      ┌──────────┐
│ Camera   │─────▶│MediaPipe │─────▶│Classifier │─────▶│   NLTK   │
│  (CV2)   │frames│Hand Det. │coords│(sklearn)  │letters│Autocorr. │
└──────────┘      └──────────┘      └───────────┘      └────┬─────┘
                                                              │
                                  ┌───────────────────────────▼─────┐
                                  │  Output: "HELLO" → Hello, etc.  │
                                  └─────────────────────────────────┘
```

**Two deployment modes:**
1. **Standalone:** Local webcam → OpenCV window (quick demo)
2. **Production:** Raspberry Pi → Firebase → Web dashboard

See [Architecture Details](docs/training-guide.md)

---

## 📊 Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Hand Detection** | MediaPipe | 21-point hand landmark extraction |
| **Gesture Classifier** | scikit-learn RandomForest | Letter prediction from landmarks |
| **Autocorrect** | NLTK + difflib | Word completion |
| **Streaming** | Firebase Realtime DB | Pi → Cloud data transfer |
| **Web Server** | Express.js + Node | Dashboard backend |
| **Frontend** | HTML/CSS/JS | User interface |

---

## 🚀 Installation

### Quick Demo (Minimal Dependencies)

```bash
# Clone repository
git clone https://github.com/Lewis-Cornelius/BSL-Translator
cd BSL-Translator

# Install minimal dependencies  
pip install -r requirements-demo.txt

# Run standalone demo
python src/demo_standalone.py
```

### Full Installation (Raspberry Pi Mode)

```bash
# Install all dependencies
pip install -r requirements.txt

# Install Node.js dependencies
cd src/web_server
npm install

# Add Firebase credentials
# Download your Firebase Admin SDK JSON and place it in src/web_server/
# File should be named: bsltranslator-93f00-firebase-adminsdk-fbsvc-55978db132.json

# Start server
node server.js
```

> ⚠️ **Firebase credentials are sensitive!** Never commit them. Already excluded in `.gitignore`.

---

## 📁 Project Structure

```
bsl-translator/
├── src/
│   ├── bsl_translator/          # Main Python package
│   │   ├── core/                # Reusable ML/CV modules
│   │   ├── streaming/           # Firebase integration
│   │   └── utils/               # Logging, utilities
│   ├── demo_standalone.py       # Quick demo (no hardware)
│   ├── raspberry_pi_main.py     # Production Pi mode
│   └── web_server/              # Node.js backend
├── tests/                       # Unit tests (pytest)
├── train/                       # ML training scripts
├── models/                      # Model files
└── docs/                        # Documentation
```

---

## 🎓 Training Your Own Model

Want to train on custom gestures?

```bash
# 1. Collect training images
python train/collect_images.py

# 2. Create dataset
python train/create_dataset.py

# 3. Train classifier
python train/train_classifier.py

# Model saved to models/bsl_classifier.pkl
```

See [Training Guide](docs/training-guide.md) for details.

---

## 🧪 Testing

```bash
# Run all tests
pip install -r requirements-dev.txt
pytest

# Run with coverage
pytest --cov=src/bsl_translator

# Lint code
flake8 src/
black --check src/
```

**Current test coverage:** 28 tests across core modules

---

## 🎮 Usage

### Standalone Mode (Demo)

```python
# Just run it!
python src/demo_standalone.py

# Controls:
# - Show BSL letters to camera (A-Z)
# - Press Q or ESC to quit
```

### Raspberry Pi Mode (Production)

```bash
# Start with stream ID
python src/raspberry_pi_main.py <stream_id>

# Or let it read from active_stream.txt
python src/raspberry_pi_main.py
```

---

## 🔧 Configuration

Edit `src/bsl_translator/core/config.py` to customize:

```python
@dataclass
class Config:
    camera: CameraConfig          # Camera settings
    mediapipe: MediaPipeConfig    # Hand detection
    gesture: GestureConfig        # Recognition thresholds
    model: ModelConfig            # ML model paths
```

---

## ⚠️ Limitations

- **Fingerspelling only:** Recognizes BSL letters (A-Z), not full sign language
- **Lighting sensitive:** Requires good lighting for hand detection
- **Single signer:** Optimized for one person at a time
- **English words:** Autocorrect uses English dictionary

---

## 🛣️ Roadmap

- [ ] Support for BSL signs (beyond fingerspelling)
- [ ] Multi-hand tracking improvements
- [ ] Mobile app (Flutter/React Native)
- [ ] Dockerized deployment
- [ ] Performance metrics dashboard

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📧 Contact

**Lewis Cornelius**  
GitHub: [@Lewis-Cornelius](https://github.com/Lewis-Cornelius)

---

## 🙏 Acknowledgments

- **MediaPipe** - Hand landmark detection
- **scikit-learn** - Machine learning framework
- **NLTK** - Natural language processing
- **Firebase** - Cloud infrastructure

---

## 📸 Screenshots

<!-- TODO: Add screenshots -->

### Standalone Demo
<!-- ![Standalone Demo](docs/screenshots/demo.png) -->

### Web Dashboard
<!-- ![Web Dashboard](docs/screenshots/dashboard.png) -->

---

**⭐ If you found this project helpful, please consider giving it a star!**
