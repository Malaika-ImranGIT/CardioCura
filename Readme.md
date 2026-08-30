# CardioCura 🫀
### AI-Powered Multilingual Heart Disease Prediction System

> Developed with support of **KOICA (Korea International Cooperation Agency)** and **Women Development Organization**  
> The Women University, Multan — BS Computer Science Final Year Project

---

## 📋 Overview

CardioCura is an AI-powered web application designed to make cardiac healthcare accessible to rural Pakistani patients — especially women — who face barriers of **language**, **cost**, and **distance** from medical specialists.

The system provides:
- **Symptom-based disease prediction** through natural language conversation
- **Clinical data analysis** using Machine Learning with WHO clinical rules
- **ECG image classification** using Deep Learning (MobileNet)
- **Multilingual support** in English, Urdu, Punjabi, and Saraiki

---

## ✨ Key Features

| Feature | Details |
|---|---|
| 🗣️ Multilingual Chatbot | English, Urdu, Punjabi, Saraiki |
| 🤖 ML Heart Risk Prediction | Random Forest — 88% accuracy |
| 📊 Clinical Rule Engine | Based on WHO / ACC-AHA guidelines |
| 🫀 ECG Image Analysis | MobileNet — 89% overall accuracy |
| 🎤 Voice Input | Web Speech API — 4 languages |
| 📱 PWA Support | Installable on Android without app store |
| 🔒 Privacy First | No patient data stored |

---

## 🏗️ System Architecture

```
User Input (Text / Voice / ECG Image)
            ↓
    Google Translate API
            ↓
    Feature Extraction 
            ↓
┌───────────────────────────────────┐
│  Layer 1: Symptom Prediction      │
│  Layer 2: ML Model   │
│  Layer 3: DL Model  │
└───────────────────────────────────┘
            ↓
    Risk Assessment Result
            ↓
    Translated Response → User
```

---

## 📊 Model Performance

### ML Model (Random Forest)
| Metric | Score |
|---|---|
| Overall Accuracy | **88%** |
| Recall — Low Risk | **97%** |
| Recall — High Risk | **79%** |
| F1 Score | **88%** |

> Previous researchers on same dataset: 60–65% accuracy

### ECG Classification (MobileNet)
| Class | Accuracy |
|---|---|
| Normal | 84% |
| Abnormal | 96% |
| Myocardial Infarction | **100%** |
| Post MI History | 77% |
| **Overall** | **89%** |

---

## 🛠️ Tech Stack

**Backend:**
```
Python 3.x | Flask | Scikit-learn | XGBoost | TensorFlow | Keras
SMOTE | Joblib | Pandas | NumPy | OpenCV | deep-translator
```

**Frontend:**
```
HTML5 | CSS3 | Bootstrap 4 | jQuery | Font Awesome | JavaScript
```

**Deployment:**
```
ngrok (HTTPS tunnel) | Progressive Web App (PWA) | Service Worker
```

---

## 📁 Project Structure

```
CardioCura/
│
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── models/
│   ├── xgboost_heart_risk_model.pkl    # Trained ML model
│   └── ecg_model.h5                    # Trained ECG DL model
│
├── static/
│   ├── images/                     # All website images
│   ├── css/
│   │   ├── mstyle.css              # Main website CSS
│   │   └── style.css               # Medibot CSS
│   ├── js/
│   │   └── main.js                 # JavaScript with speech to text
│   ├── manifest.json               # PWA manifest
│   └── sw.js                       # Service worker
│
├── templates/
│   ├── index.html                  # Home page
│   ├── about.html                  # About page
│   ├── doctor.html                 # Doctor profiles
│   ├── patient.html                # Patient reviews
│   ├── chat.html                   # Medibot chatbot
│   ├── contact.html                # Contact form
│   └── blog.html                   # Health blog
│
├── dataset/
│   └── heart_attack_prediction_dataset.csv
│
└── contact.csv                     # Contact form submissions
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.9 or above
- pip
- Git

### Step 1 — Clone the Repository
```bash
git clone https://github.com/your-username/cardiocura.git
cd cardiocura
```

### Step 2 — Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Run the Application
```bash
python app.py
```

### Step 5 — Open in Browser
```
http://127.0.0.1:5000
```

---




## 📂 Dataset & Files Setup

These files are not included in this repository due to size.
Download them from the links below and place in correct folders.

### CSV Datasets — place in `data/` folder
| File | Download |


### ECG Image Dataset — extract into `data/balanced_augmented_data/`
| File | Download |
|---|---|

After extracting you should have:

data/

└── balanced_augmented_data/

├── abnormal_heartbeat_ecg_images/

├── myocardial_infarction_ecg_images/

├── normal_ecg_images/

└── post_mi_history_ecg_images/

### ML Model
Trains automatically when you run `python app.py`
No download needed.

### ECG Deep Learning Model
| File | Download |


Place in: `models/final_heart_disease_model.h5`





## 📱 Mobile Access (PWA)

To access on mobile phone:

**Step 1** — Install ngrok from https://ngrok.com

**Step 2** — Run Flask:
```bash
python app.py
```

**Step 3** — Run ngrok in a second terminal:
```bash
ngrok http 5000
```

**Step 4** — Open the `https://` link on Android Chrome

**Step 5** — Tap 3 dots → Add to Home Screen → Install

---

## 🌐 Supported Languages

| Language | Input | Voice | Output |
|---|---|---|---|
| English | ✅ | ✅ | ✅ |
| Urdu (اردو) | ✅ | ✅ | ✅ |
| Punjabi (پنجابی) | ✅ | ✅ | ✅ |
| Saraiki (سرائیکی) | ✅ | ✅ | ✅ |

---

## 🫀 ECG Test Images

For testing the ECG classification feature use images from:
- [PhysioNet PTB-XL Dataset](https://physionet.org/content/ptb-xl/1.0.3/)
- [Kaggle ECG Dataset](https://www.kaggle.com/datasets/khyeh0719/ptb-xl-dataset)
- [Wikipedia Normal ECG](https://upload.wikimedia.org/wikipedia/commons/9/9e/SinusRhythmLabels.svg)

---

## 🔬 Dataset

- **Source:** [Kaggle — Sourav Banerjee Heart Attack Prediction Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/heart-attack-prediction-dataset)
- **Rows:** 8,763 patients
- **Features:** 21 clinical features
- **Balancing:** SMOTE applied
- **Challenge:** Synthetic dataset with max correlation 0.027

---

## 📋 21 Clinical Features

```
Demographic:     Age, Sex, BMI
Vitals:          Systolic BP, Diastolic BP, Heart Rate,
                 Cholesterol, Triglycerides
Lifestyle:       Smoking, Alcohol, Exercise Hours/Week,
                 Sedentary Hours/Day, Physical Activity Days,
                 Diet, Sleep Hours/Day, Stress Level
Medical History: Diabetes, Family History, Obesity,
                 Previous Heart Problems, Medication Use
```

---

## ⚕️ WHO Clinical Rules

The system overrides ML output when:
- Systolic BP ≥ 180 or Diastolic BP ≥ 120 (Hypertensive Crisis)
- Systolic BP ≥ 150 + Age ≥ 60 (Elderly Hypertension)
- Triglycerides ≥ 400 mg/dL (Severely Elevated)
- Cholesterol ≥ 300 (Severely Elevated)
- Previous Heart Problems detected
- 3 or more combined risk factors

---

## ⚠️ Disclaimer

> CardioCura is designed for **educational and screening purposes only**.  
> It is **not a substitute** for professional medical advice, diagnosis, or treatment.  
> Always consult a qualified cardiologist for any cardiac concerns.

---

## 🙏 Acknowledgements

Special thanks to:

- **The Women University, Multan** — Institute of Computer Science and Information Technology

---

## 👩‍💻 Developer

**Malaika Imran**  
BS Computer Science  
The Women University, Multan  
2022 – 2026

---

## 📄 License

This project is developed as a Final Year Project at The Women University, Multan.  
© 2026 CardioCura. All rights reserved.
