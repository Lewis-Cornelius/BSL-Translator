# BSL Translator

[![Python](https://img.shields.io/badge/Python-3.11.9-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Node.js](https://img.shields.io/badge/Node.js-Required-brightgreen.svg)](https://nodejs.org/)

A real-time British Sign Language (BSL) translator that uses computer vision and machine learning to recognize hand gestures and convert them to text. The system uses MediaPipe for hand landmark detection and a custom-trained ML model for gesture classification.

![Logo](https://github.com/Lewis-Cornelius/BSL-Translator/blob/main/project/server/website/logo.png)

## ⚡ Quick Start - Try It Now!

**Want to test BSL translation on your computer in under 2 minutes?**

```bash
# 1. Clone the repository
git clone https://github.com/Lewis-Cornelius/BSL-Translator
cd BSL-Translator

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Run the standalone webcam mode
python project/server/bsl_webcam_standalone.py
```

That's it! A webcam window will open showing:
- 📹 **Live video** with hand landmark detection
- 🔤 **Real-time letter recognition** from BSL gestures
- 💬 **Automatic word completion** and sentence formation

**No server, no Firebase, no Raspberry Pi needed!**

---

## Features

✨ **Real-time Translation** - Instant BSL gesture recognition using your camera  
🤖 **Machine Learning** - Custom-trained model for accurate gesture classification  
🔄 **Smart Auto-correction** - Improves word prediction accuracy  
📱 **Raspberry Pi Support** - Deploy on embedded hardware with camera module  
🌐 **Web Interface** - User-friendly web application with authentication  
🔥 **Firebase Integration** - Cloud-based data synchronization and storage

## Installation

### Installation on a laptop/pc

**Prerequisites:**
- Python 3.11.9
- Node.js and npm
- Git

**Steps:**

```bash
git clone https://github.com/Lewis-Cornelius/BSL-Translator
cd BSL-Translator
```

Open the project in VS Code or your preferred editor and run these commands in the terminal:

```
python -m venv myenv
myenv\Scripts\activate
pip install -r requirements.txt
cd project
cd server
npm install
npm install firebase
npm install express body-parser cors
```

You then need to add your Firebase Admin SDK JSON file.

#### Steps to Download Your Firebase Admin SDK JSON File
1. Go to [Firebase Console](https://console.firebase.google.com/) and log in
2. Select your project
3. Click on the ⚙️ gear icon → **Project settings**
4. Navigate to the **Service accounts** tab
5. Click **Generate new private key**
6. Save the downloaded JSON file as `bsltranslator-93f00-firebase-adminsdk-fbsvc-55978db132.json`
7. Place it in `BSL-Translator/project/server`

> **⚠️ IMPORTANT:** This file contains sensitive credentials. Never commit it to version control. It's already excluded in `.gitignore`.

To make sure everything is installed, run:

```
python fix.py
```

If there are any missing dependencies, type `y` to fix them. OpenCV sometimes has issues, so you may need to run `pip install opencv-python`. Similarly, scikit-learn can be problematic, so try `pip install scikit-learn` if needed.

To start running the server:
```
node server.js
```

### Installation on the Raspberry Pi

First, set up the Pi as shown below:
- Camera plugged into the Raspberry Pi camera pins
- Red LED button connected to port D2
- Grove-LCD RGB Backlight connected to port I2C

![Raspberry Pi Setup](https://github.com/Lewis-Cornelius/BSL-Translator/assets/61e809d4-83b1-43b0-ab5d-4606c24fc7a9)

The only files you need from the repository are in the **`/RaspberryPi`** folder:
- `live_stream_pi_code.py`
- `on_off_button_pi_code.py`
- `requirements.txt`

```
pip install -r requirements.txt
```

Open and run `on_off_button_pi_code.py`

**NOTE: PLEASE PUT THE `index.html` AND `logo.png` INTO A FOLDER CALLED `templates`**

## How Everything Works

### Server.js
The website backend runs using Express.js to handle communication between the BSL interpreter Python scripts and the front-end website. It manages API endpoints that read and update the translation data and provide error messages when problems occur.

### BSL_bridge_integration.py
This Python script implements real-time British Sign Language gesture recognition using OpenCV, MediaPipe, and a pre-trained ML model. It detects hand landmarks (21 points per hand) and predicts hand gestures, converting them to text through these steps:
1. Hand detection using MediaPipe
2. Landmark extraction and classification
3. Gesture stabilization to prevent duplicate predictions
4. Auto-correction to improve word prediction

### Firebase_stream_handler.py
This script handles the conversion of base64-encoded images back to actual images for display on the website and processing by the BSL translator. It manages two-way communication between the client computer and Firebase.

### Raspberry Pi Implementation
The Raspberry Pi runs two main scripts:
- `on_off_button_pi_code.py`: Controls the LCD and camera based on button presses
- `live_stream_pi_code.py`: Captures and streams video frames to Firebase

Configuration parameters:
```python
FRAME_RATE = 3  # Do not exceed 5 FPS due to storage limitations
RESOLUTION = (320, 240)  # Lower resolution for better performance
QUALITY = 20  # JPEG compression quality
MAX_STREAM_TIME = 300  # Maximum streaming time (5 minutes)
```

### Website Interface
The web application includes:
- Authentication pages (login, signup)
- Main translator interface (webpage.html)
- Stream connection and management
- Real-time translation display

### BSL Letter Chart
![BSL Fingerspelling Chart](https://imgs.search.brave.com/DKomfn_cPKzVi7KigGeY5d0Jdn0WK72m8gxgMzOFH6M/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9hY2Nl/c3Nic2wuY29tL3dw/LWNvbnRlbnQvdXBs/b2Fkcy8yMDIyLzEx/LzIuYWNjZXNzYnNs/LUZpbmdlcnNwZWxs/aW5nLXJpZ2h0LWhh/bmQtMS5qcGc)

## Training Your Own Model

To train your own model:

1. **Collect Images** - Run `collect_images.py` to capture hand gestures
   - Configure `number_of_classes` for the number of letters
   - Update `data_dict` path to your local directory

2. **Process Data** - Run `create_dataset.py` to prepare the training data

3. **Train Model** - Run `train_classifier.py` to create the `model.p` file

4. **Test Model** - Run `inference_classifier.py` to validate the model
   - Update `labels_dict` to match your classes (e.g., `{0: 'A', 1: 'B', 2: 'L'}`)

## System Architecture

The data flows from the Raspberry Pi to Firebase, then to a PC/Laptop where the BSL is translated to text. The processed data is sent to:

1. The University's Cloud Webserver (backend for website and mobile interface)
2. Locally hosted on JS server


## Firebase Integration

The system uses Firebase for real-time data storage and synchronization:

![Firebase Structure](https://github.com/Lewis-Cornelius/BSL-Translator/assets/f8e41ac4-51ef-4f1f-825d-01f8e867b1da)

## Web Interface

The current web interface looks like this:

![Web Interface](https://github.com/Lewis-Cornelius/BSL-Translator/assets/9ce45291-fea6-4419-8a29-9f95c9cedc46)


## Video Tutorial for Reference

[![Watch the tutorial on YouTube](https://img.youtube.com/vi/MJCSjXepaAM/0.jpg)](https://www.youtube.com/watch?v=MJCSjXepaAM)

## Future Development

- 🎯 Improve AI model accuracy and recognition speed
- 🚀 Optimize connection speeds and reduce latency

- MediaPipe for hand landmark detection
- Firebase for real-time data synchronization
- The BSL community for gesture references

---

**Made with ❤️ for accessibility**
