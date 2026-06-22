from pyexpat import model
from flask import Flask, redirect, render_template, request, session, jsonify
import pandas as pd
import re
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import StackingClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
# from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, recall_score, classification_report, confusion_matrix
import tensorflow as tf
keras = tf.keras
layers = tf.keras.layers
models = tf.keras.models
load_model = tf.keras.models.load_model
MobileNetV2 = tf.keras.applications.MobileNetV2
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
import cv2, os
import numpy as np
import csv
from flask import flash
from deep_translator import GoogleTranslator
from aksharamukha import transliterate
from sklearn.metrics import roc_curve, f1_score


app = Flask(__name__)
app.secret_key = "cardio_cura_secret_key"

SUPPORTED_LANGUAGES = {
    "English": "en",
    "Urdu": "ur",
    "Punjabi": "pa",
    "Saraiki": "skr"
}



def load_csv(path):
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")


# --------------------------------------------
# 1. LAB DATA TRAINING LOGIC (RUNS ONCE)
# --------------------------------------------
MODEL_PATH = r"C:/Users/PMLS/Desktop/Main/Final_Heart_Disease_Chatbot/models/NEW_SMOTE_heart_risk_model.pkl"
DATA_PATH =  r"C:/Users/PMLS/Desktop/Main/Final_Heart_Disease_Chatbot/data/heart_attack_prediction_dataset.csv"
VISION_MODEL_PATH = r"C://Users//PMLS//Desktop//Main//Final_Heart_Disease_Chatbot//models//final_heart_disease_model.h5" # Path for your ECG Image model


# Load Text Model
try:
    text_model = joblib.load(MODEL_PATH)
    print("✅ Text-based Symptom Model Loaded.")
except Exception as e:
    text_model = None
    print(f"❌ Text Model not found: {e}")

# Load Vision Model
try:
    vision_model = load_model(VISION_MODEL_PATH)
    print("✅ Deep Learning Vision Model Loaded.")
except Exception as e:
    vision_model = None
    print(f"❌ Vision Model not found. (Will be ready after training)")

def train_lab_model():
    if not os.path.exists(DATA_PATH):
        print(f"Data file not found at {DATA_PATH}")
        return
    
    df = pd.read_csv(DATA_PATH)
    df[['Systolic', 'Diastolic']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)
    df = df.drop('Blood Pressure', axis=1)
    
    noise_cols = ['Patient ID', 'Country', 'Continent', 'Hemisphere', 'Income']
    df = df.drop(columns=[c for c in noise_cols if c in df.columns])
    
    le_sex = LabelEncoder()
    df['Sex'] = le_sex.fit_transform(df['Sex'])
    le_diet = LabelEncoder()
    df['Diet'] = le_diet.fit_transform(df['Diet'])
    
    X = df.drop('Heart Attack Risk', axis=1)
    y = df['Heart Attack Risk']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    if os.path.exists(MODEL_PATH):
        print("\n--- Model found. Loading... ---")
        data_bundle = joblib.load(MODEL_PATH)
        model = data_bundle['model']
        scaler = data_bundle.get('scaler')
        if scaler:
            X_test_eval = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
        else:
            X_test_eval = X_test

    else:
        print("\n--- Training new model... ---")
        
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test_eval = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        sm = SMOTE(random_state=42)
        X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)

        model = RandomForestClassifier(
            n_estimators=1500,       # More trees = more stable on noisy data
            max_depth=20,            # Deeper than before
            min_samples_split=4,
            min_samples_leaf=1,
            max_features='sqrt',     # Standard best practice for RF
            bootstrap=True,
            class_weight='balanced', # Extra balance on top of SMOTE
            random_state=42,
            n_jobs=-1                # Use all CPU cores — faster training
        )
        model.fit(X_train_res, y_train_res)

        # Threshold search
        y_probs = model.predict_proba(X_test_eval)[:, 1]
        best_thresh = 0.50
        best_f1 = 0

        for thresh in [x / 100 for x in range(30, 70)]:
            y_pred_temp = (y_probs >= thresh).astype(int)
            if len(set(y_pred_temp)) < 2:
                continue
            f1 = f1_score(y_test, y_pred_temp)
            r1 = recall_score(y_test, y_pred_temp)
            r0 = recall_score(y_test, y_pred_temp, pos_label=0)

            if r1 >= 0.80 and r0 >= 0.85 and f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        # Fallback: if no threshold satisfied both constraints,
        # pick the threshold with best F1 ignoring recall limits
        if best_f1 == 0:
            print("[WARN] No threshold met both recall constraints. "
                  "Picking best overall F1 threshold.")
            for thresh in [x / 100 for x in range(30, 70)]:
                y_pred_temp = (y_probs >= thresh).astype(int)
                if len(set(y_pred_temp)) < 2:
                    continue
                f1 = f1_score(y_test, y_pred_temp)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = thresh

        print(f"[INFO] Optimal threshold: {best_thresh:.2f} | F1: {best_f1:.3f}")

        # ── NEUTRAL DEFAULTS ────────────────────────────────────────────
        # This dataset has corrupted medians (Smoking=1, Diabetes=1, Obesity=1
        # as population median). We use clinically neutral defaults instead
        # so unknown features don't push every patient toward High Risk.
        neutral_defaults = {
            'Age': 45.0,
            'Sex': 1.0,                       # Male
            'Cholesterol': 200.0,             # Borderline normal
            'Heart Rate': 75.0,               # Normal
            'Diabetes': 0.0,                  # No diabetes
            'Family History': 0.0,            # No family history
            'Smoking': 0.0,                   # Non-smoker
            'Obesity': 0.0,                   # Not obese
            'Alcohol Consumption': 0.0,       # No alcohol
            'Exercise Hours Per Week': 3.0,   # Light exercise
            'Diet': 1.0,                      # Healthy (LabelEncoder: Average=0, Healthy=1, Unhealthy=2)
            'Previous Heart Problems': 0.0,   # No prior issues
            'Medication Use': 0.0,            # No medication
            'Stress Level': 3.0,              # Low-moderate stress
            'Sedentary Hours Per Day': 6.0,   # Moderate sitting
            'BMI': 24.0,                      # Normal BMI
            'Triglycerides': 150.0,           # Normal
            'Physical Activity Days Per Week': 3.0,
            'Sleep Hours Per Day': 7.0,
            'Systolic': 120.0,                # Normal BP
            'Diastolic': 80.0,
        }
        # ───────────────────────────────────────────────────────────────

        joblib.dump({
            'model': model,
            'scaler': scaler,
            'medians': neutral_defaults,      # ← neutral defaults, not dataset medians
            'le_sex': le_sex,
            'le_diet': le_diet,
            'threshold': best_thresh
        }, MODEL_PATH)

    # Evaluation
    y_pred = model.predict(X_test_eval)
    print(f"\nAccuracy : {accuracy_score(y_test, y_pred):.2f}")
    print(f"Recall 1 : {recall_score(y_test, y_pred):.2f}")
    print(f"Recall 0 : {recall_score(y_test, y_pred, pos_label=0):.2f}")
    print(classification_report(y_test, y_pred))

# --------------------------------------------
# 2. LAB DATA EXTRACTION LOGIC
# --------------------------------------------
def check_clinical_override(patient_data):
    """
    Hard clinical rules based on established medical thresholds (WHO / ACC/AHA).
    Returns (is_high_risk: bool, reasons: list).
    This acts as a safety net when the ML model probability is borderline.
    """
    reasons = []

    systolic  = patient_data.get('Systolic', 0)
    diastolic = patient_data.get('Diastolic', 0)
    cholesterol   = patient_data.get('Cholesterol', 0)
    triglycerides = patient_data.get('Triglycerides', 0)
    bmi       = patient_data.get('BMI', 0)
    smoking   = patient_data.get('Smoking', 0)
    diabetes  = patient_data.get('Diabetes', 0)
    prev_heart = patient_data.get('Previous Heart Problems', 0)
    family    = patient_data.get('Family History', 0)
    age       = patient_data.get('Age', 0)
    exercise  = patient_data.get('Exercise Hours Per Week', 99)
    sedentary = patient_data.get('Sedentary Hours Per Day', 0)
    stress    = patient_data.get('Stress Level', 0)

    # Rule 1: Hypertensive crisis — medical emergency (WHO threshold)
    if systolic >= 180 or diastolic >= 120:
        reasons.append(f"Hypertensive crisis (BP {systolic}/{diastolic})")

    # Rule 2: Stage 2 hypertension in elderly
    if systolic >= 150 and age >= 60:
        reasons.append(f"High BP + age risk (BP {systolic}, Age {age})")

    # Rule 3: Very high triglycerides (>= 400 mg/dL is severely elevated)
    if triglycerides >= 400:
        reasons.append(f"Severely elevated triglycerides ({triglycerides} mg/dL)")

    # Rule 4: Very high cholesterol
    if cholesterol >= 300:
        reasons.append(f"Severely elevated cholesterol ({cholesterol})")

    # Rule 5: Previous heart problems — always flag High Risk
    if prev_heart == 1:
        reasons.append("Previous heart problems reported")

    # Rule 6: Combination of 3+ major independent risk factors
    risk_factors = [
        smoking == 1,
        diabetes == 1,
        bmi >= 30,
        family == 1,
        age >= 65,
        systolic >= 140,
        triglycerides >= 200,
        cholesterol >= 240,
        exercise == 0,
        sedentary >= 10,
        stress >= 8,
    ]
    count = sum(risk_factors)
    if count >= 3:
        reasons.append(f"Multiple combined risk factors ({count} factors present)")

    return len(reasons) > 0, reasons


def extract_lab_data(user_msg):
    """
    Parses patient message (English / Urdu / Punjabi / Saraiki) and fills
    a feature dictionary for the ML model.
    Starts from clinically NEUTRAL defaults — not dataset medians —
    so unknown features do not bias the prediction toward High Risk.
    """
    data = joblib.load(MODEL_PATH)
    # Use neutral defaults stored at training time (NOT dataset medians)
    patient_data = data['medians'].copy()

    msg = user_msg.lower()

    # ─────────────────────────────────────────────────────────────────
    # NUMERIC PATTERNS  (English + Urdu + Punjabi + Saraiki keywords)
    # ─────────────────────────────────────────────────────────────────
    patterns = {
        # Age  — "65 years", "65 سال", "age 65", "عمر 65"
        'Age': (
            r'(?:age|عمر|umra)[\s:]*(\d+)'
            r'|(\d+)\s*(?:years?\s*old|years?|yr|saal|varhe|سال|ورھے)'
        ),

        # Cholesterol — "cholesterol 180", "chol is 180"
        'Cholesterol': (
            r'(?:cholesterol|chol|کولیسٹرول)\s*(?:is|of|level|=|:)?\s*(\d+)'
        ),

        # Heart Rate — "heart rate 90", "pulse 90", "90 bpm"
        'Heart Rate': (
            r'(?:heart\s*rate|pulse|bpm|dil\s*ki\s*dharhkan|نبض|دھڑکن)'
            r'\s*(?:is|of|=|:)?\s*(\d+)'
            r'|(\d+)\s*bpm'
        ),

        # Blood Pressure — "260/150", "bp 260/150"
        'Systolic':  r'(?:bp|blood\s*pressure|بلڈ\s*پریشر)?[\s:]*(\d{2,3})/\d{2,3}',
        'Diastolic': r'(?:bp|blood\s*pressure|بلڈ\s*پریشر)?[\s:]*\d{2,3}/(\d{2,3})',

        # BMI — "bmi 38", "bmi is 22.5"
        'BMI': r'(?:bmi|باڈی\s*ماس)\s*(?:is|of|=|:)?\s*(\d+\.?\d*)',

        # Triglycerides — "triglycerides 450", "tri 400"
        'Triglycerides': (
            r'(?:triglycerides?|tri(?:glycerides?)?|ٹرائی\s*گلیسرائیڈز)'
            r'\s*(?:is|of|=|:|are)?\s*(\d+)'
        ),

        # Stress level numeric — "stress level 8"
        'Stress Level': (
            r'(?:stress\s*level|ذہنی\s*دباؤ)\s*(?:is|of|=|:)?\s*(\d+)'
        ),

        # Exercise hours — "exercise 3 hours", "workout 2 hours a week"
        'Exercise Hours Per Week': (
            r'(?:exercise|workout|ورزش|physical\s*activity)'
            r'\s*(?:is|of|for|about)?\s*(\d+\.?\d*)\s*(?:hours?|hrs?|گھنٹے)'
        ),

        # Sedentary hours — "sitting 14 hours", "14 hours a day sitting"
        'Sedentary Hours Per Day': (
            r'(\d+)\s*(?:hours?|گھنٹے|گھنٹہ)\s*(?:a\s*day|per\s*day|daily)?'
            r'\s*(?:sitting|sedentary|بیٹھ|بیٹھے)'
            r'|(?:sitting|sedentary|بیٹھ)\s*(?:for)?\s*(\d+)\s*(?:hours?|گھنٹے)'
        ),

        # Sleep hours — "sleep 6 hours", "6 hours of sleep"
        'Sleep Hours Per Day': (
            r'(\d+)\s*(?:hours?|گھنٹے)?\s*(?:of\s*)?sleep'
            r'|(?:sleep|سونا)\s*(?:for|=|:)?\s*(\d+)'
        ),

        # Physical activity days — "active 4 days a week"
        'Physical Activity Days Per Week': (
            r'(?:active|activity|physical)\s*(\d+)\s*days?'
            r'|(\d+)\s*days?\s*(?:a\s*week|per\s*week)\s*(?:active|exercise|workout)'
        ),
    }

    found_any = False

    for feature, pattern in patterns.items():
        match = re.search(pattern, msg, re.IGNORECASE)
        if match:
            found_any = True
            value = next((g for g in match.groups() if g is not None), None)
            if value is not None:
                patient_data[feature] = float(value)

    # ─────────────────────────────────────────────────────────────────
    # BINARY / KEYWORD FEATURES
    # ─────────────────────────────────────────────────────────────────

    # --- Smoking ---
    if re.search(
        r'\b(smok(?:er|ing|e)?|cigarette|tobacco|سگریٹ|تمباکو|پیتا\s*ہوں|نشہ)\b',
        msg
    ):
        patient_data['Smoking'] = 1
        found_any = True
    elif re.search(
        r'\b(non[\-\s]?smok|not\s+smok|never\s+smok|quit\s+smok|نہیں\s*پیتا|سگریٹ\s*نہیں)\b',
        msg
    ):
        patient_data['Smoking'] = 0

    # --- Diabetes ---
    if re.search(
        r'\b(diabet(?:es|ic)?|sugar\s*patient|high\s*sugar|ذیابیطس|شوگر\s*(?:کا\s*مریض|ہے|ہوئی))\b',
        msg
    ):
        patient_data['Diabetes'] = 1
        found_any = True
    elif re.search(
        r'\b(no\s*diabet|no\s*sugar|شوگر\s*نہیں)\b',
        msg
    ):
        patient_data['Diabetes'] = 0

    # --- Family History of Heart Disease ---
    if re.search(
        r'(?:family|father|mother|brother|sister|parent|دادا|نانا|والد|والدہ|بھائی|بہن|خاندان|رشتہ\s*دار)'
        r'.{0,30}(?:heart|cardiac|attack|دل|قلب)',
        msg
    ) or re.search(
        r'(?:heart|cardiac|دل).{0,30}(?:family|relative|history|خاندان|رشتہ\s*دار)',
        msg
    ):
        patient_data['Family History'] = 1
        found_any = True

    # --- Previous Heart Problems ---
    if re.search(
        r'\b(?:previous|past|history|had\s+a|پہلے|پرانا|پہلا)\b.{0,30}'
        r'\b(?:heart|cardiac|attack|bypass|stent|دل|قلب)\b',
        msg
    ) or re.search(
        r'\b(?:heart\s*attack|cardiac\s*arrest|bypass|stent|angioplasty|دل\s*کا\s*دورہ)\b',
        msg
    ):
        patient_data['Previous Heart Problems'] = 1
        found_any = True

    # --- Medication ---
    if re.search(
        r'\b(?:medication|medicine|on\s*meds|taking\s*(?:pills?|tablets?|drugs?)|'
        r'دوائی|دوا|گولی|tablet|علاج\s*ہو\s*رہا)\b',
        msg
    ):
        patient_data['Medication Use'] = 1
        found_any = True

    # --- Obesity (keyword, when no numeric BMI provided) ---
    if re.search(r'\b(?:obese|obesity|very\s*overweight|موٹاپا|بہت\s*موٹا|بھاری)\b', msg):
        if patient_data.get('BMI', 0) < 30:
            patient_data['BMI'] = 35.0
        patient_data['Obesity'] = 1
        found_any = True

    # --- Obesity flag from numeric BMI ---
    if patient_data.get('BMI', 0) >= 30:
        patient_data['Obesity'] = 1

    # --- No exercise ---
    if re.search(
        r'\b(?:no\s*exercise|don[\'\s]?t\s*exercise|never\s*exercise|'
        r'do\s*not\s*exercise|ورزش\s*نہیں|کوئی\s*ورزش\s*نہیں)\b',
        msg
    ):
        patient_data['Exercise Hours Per Week'] = 0.0
        found_any = True

    # --- Sedentary lifestyle (keyword) ---
    if re.search(
        r'\b(?:sitting\s*all\s*day|sedentary|inactive|no\s*activity|'
        r'بیٹھا\s*رہتا|حرکت\s*نہیں)\b',
        msg
    ):
        if patient_data.get('Sedentary Hours Per Day', 0) < 10:
            patient_data['Sedentary Hours Per Day'] = 12.0
        found_any = True

    # --- Alcohol ---
    if re.search(r'\b(?:alcohol|drink(?:ing)?|شراب|الکوحل)\b', msg):
        patient_data['Alcohol Consumption'] = 1
        found_any = True
    elif re.search(r'\b(?:no\s*alcohol|don[\'\s]?t\s*drink|شراب\s*نہیں)\b', msg):
        patient_data['Alcohol Consumption'] = 0

    # --- High stress (keyword) ---
    if re.search(
        r'\b(?:very\s*stress(?:ed)?|high\s*stress|extreme\s*stress|'
        r'بہت\s*پریشان|بہت\s*ذہنی\s*دباؤ|ٹینشن)\b',
        msg
    ):
        patient_data['Stress Level'] = 9
        found_any = True

    # ─────────────────────────────────────────────────────────────────
    # SEX  (Female=0, Male=1  per LabelEncoder fit order)
    # ─────────────────────────────────────────────────────────────────
    if re.search(r'\b(?:female|woman|girl|lady|عورت|لڑکی|خاتون|خاتوں)\b', msg):
        patient_data['Sex'] = 0
        found_any = True
    elif re.search(r'\b(?:male|man|boy|gentleman|مرد|لڑکا|آدمی|صاحب)\b', msg):
        patient_data['Sex'] = 1
        found_any = True

    # ─────────────────────────────────────────────────────────────────
    # DIET  (Average=0, Healthy=1, Unhealthy=2  per LabelEncoder)
    # ─────────────────────────────────────────────────────────────────
    if re.search(
        r'\b(?:unhealthy|junk|fast\s*food|oily|fried|غیر\s*صحت|کباب|برگر|تلا\s*ہوا)\b',
        msg
    ):
        patient_data['Diet'] = 2          # Unhealthy
        found_any = True
    elif re.search(
        r'\b(?:healthy|balanced|salad|fruits?|vegetables?|سبزی|صحت\s*مند|متوازن)\b',
        msg
    ):
        patient_data['Diet'] = 1          # Healthy
    # else stays at neutral default (1 = Healthy / Average)

    return patient_data if found_any else None
# --------------------------------------------
# 3. CHAT ROUTES (Symptom Matching + Lab Risk)
# --------------------------------------------
# [Keep your previous symptoms_df and load_csv functions here...]

symptoms_df = load_csv(r"C:/Users/PMLS/Desktop/Main/Final_Heart_Disease_Chatbot/data/heart_diseases&symptoms.csv")
desc_prec_df = load_csv(r"C:/Users/PMLS/Desktop/Main/Final_Heart_Disease_Chatbot/data/Heart_Disease_description&precautions.csv")

# Standardize column names
symptoms_df.columns = [c.strip().lower() for c in symptoms_df.columns]
desc_prec_df.columns = [c.strip() for c in desc_prec_df.columns]

def find_matching_diseases(user_input):
    user_input_l = user_input.lower()
    
    matches = []
    for _, row in symptoms_df.iterrows():
        disease_name = str(row['disease']).strip()
        # Clean the dataset symptoms
        db_symptoms_str = str(row['symptoms']).lower().replace('"', '')
        db_symptoms_list = [s.strip() for s in db_symptoms_str.split(',')]
        
        # Count matches
        score = 0
        for sym in db_symptoms_list:
            if sym in user_input_l:
                score += 1
        
        if score >= 1:
            matches.append({"disease": disease_name, "count": score})
    
    # SORTING LOGIC:
    # We sort by the highest count first. 
    # If the top result has a much higher score than the second, we only show the top one.
    sorted_matches = sorted(matches, key=lambda x: x['count'], reverse=True)
    
    return sorted_matches

# Update your find_matching_diseases function
def find_translated_diseases(user_input):
    user_input_l = user_input.lower()
    # Remove extra words like "feel", "sometimes", "very" to focus on body parts
    stop_words = ["sometimes", "feel", "i", "have", "my", "is", "very"]
    user_words = [w for w in user_input_l.split() if w not in stop_words]
    
    matches = []
    for _, row in symptoms_df.iterrows():
        db_symptoms = str(row['symptoms']).lower()
        # Check how many of the user's words appear in the disease symptoms
        score = sum(1 for word in user_words if word in db_symptoms)
        
        if score >= 1:
            matches.append({"disease": row['disease'], "count": score})
    
    return sorted(matches, key=lambda x: x['count'], reverse=True)

def get_disease_details(disease_name):
    match = desc_prec_df[desc_prec_df['Disease'].str.lower() == disease_name.lower()]
    if not match.empty:
        return {
            "description": match.iloc[0]['Description'],
            "precautions": match.iloc[0]['Precautions']
        }
    return None

@app.route("/")
@app.route("/")
def home():
    """Main Website Landing Page"""
    session.clear() # Optional: clears chat history when returning to home
    return render_template("index.html")

@app.route("/medibot")
def medibot():
    """Chatbot Interface Page"""
    return render_template("chat.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact", methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        # 1. Capture data from the form
        name = request.form.get('name')
        email = request.form.get('email')
        phone = request.form.get('phone')
        message = request.form.get('message')

        # 2. Define the CSV path
        csv_file = r"C://Users//PMLS//Desktop//Main//Final_Heart_Disease_Chatbot//data//Contact.csv"
        file_exists = os.path.isfile(csv_file)

        # 3. Store data in CSV
        with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            
            # Write header only if file is being created for the first time
            if not file_exists:
                writer.writerow(['Name', 'Email', 'Phone', 'Message'])
            
            writer.writerow([name, email, phone, message])

        # 4. Redirect back to contact page (or a success page)
        # Inside the if request.method == 'POST' block:
        flash('Thank you! Your message has been saved.', 'success')
        return redirect('/contact')

    return render_template('contact.html')

@app.route("/patient")
def patient():
    return render_template("patient.html")
@app.route("/doctor")
def doctor():
    return render_template("doctor.html")

@app.route("/blog")
def blog():
    return render_template("blog.html")

@app.route("/set_languages", methods=["POST"])
def set_languages():
    # This captures the choice from your frontend dropdown/buttons
    input_lang = request.form.get("input_lang")
    output_lang = request.form.get("output_lang")
    session["input_lang"] = input_lang
    session["output_lang"] = output_lang
    return jsonify({"success": True})


@app.route("/get", methods=["POST"])
def chat():
    user_msg = request.form.get("msg", "").strip()
    
    # 1. Get languages from session (Set via your /set_languages route)
    input_lang = session.get("input_lang", "en")
    output_lang = session.get("output_lang", "en")

    try:
       
        try:
            if input_lang != "en":
                translated_input = GoogleTranslator(source='auto', target="en").translate(user_msg)
            else:
                translated_input = user_msg
        except:
            # Fallback: Many Saraiki/Punjabi speakers use Urdu script/vocabulary
            translated_input = GoogleTranslator(source='ur', target="en").translate(user_msg)

        # --- STEP B: Your Core Logic (Using translated_input) ---
        
        # Phase 1: Name Extraction
        if "user_name" not in session:
            user_msg_l = translated_input.lower()
            if "name is" in user_msg_l:
                name = user_msg_l.split("name is")[-1].strip().split()[0].capitalize()
            elif "i'm" in user_msg_l:
                name = user_msg_l.split("i'm")[-1].strip().split()[0].capitalize()
            elif "i am" in user_msg_l:
                name = user_msg_l.split("i am")[-1].strip().split()[0].capitalize()
            else:
                words = translated_input.split()
                name = words[-1].capitalize() if len(words) > 1 else words[0].capitalize()
        
            session["user_name"] = name
            final_response = f"Hello {name}! It's nice to meet you. I have noted your details. To help you better, could you please tell me about the symptoms you are currently feeling?"
        
        else:
            user_name = session["user_name"]
            
            # Phase 2: Lab Report Detection
            lab_data = extract_lab_data(translated_input)
            if lab_data:
                data_bundle = joblib.load(MODEL_PATH)
                model = data_bundle['model']
                scaler = data_bundle['scaler']
                threshold = data_bundle.get('threshold', 0.50)

                df_input = pd.DataFrame([lab_data])
                df_input_scaled = pd.DataFrame(
                    scaler.transform(df_input), columns=df_input.columns
                )

                prob = model.predict_proba(df_input_scaled)[0][1]

                # Clinical rule override (catches cases the weak-signal model misses)
                is_override, override_reasons = check_clinical_override(lab_data)

                # Debug — remove after testing
                print("\n[DEBUG] Features sent to model:")
                for col, val in zip(df_input.columns, df_input.values[0]):
                    print(f"  {col}: {val}")
                print(f"[DEBUG] ML prob: {prob:.3f} | Threshold: {threshold:.2f}")
                print(f"[DEBUG] Override: {is_override} | Reasons: {override_reasons}")

                if is_override or prob >= threshold:
                    risk_text = "High Risk"
                    risk_style = "color:red; font-weight:bold;"
                else:
                    risk_text = "Low Risk"
                    risk_style = "color:green; font-weight:bold;"

                final_response = (f"Analysis for {user_name}: Based on the data provided, your predicted risk level is <span style='{risk_style}'>{risk_text}</span>.<br><b>Precautions:</b> Keep your diet healthy, exercise regularly, and avoid excessive salt or processed sugars.<br><b>Note:</b><i>This AI-generated analysis is for educational and screening purposes only. It is not a substitute for professional medical advice. Always consult a qualified cardiologist for a definitive diagnosis.")
            
            else:
                if input_lang != "en":
                    potential_matches = find_translated_diseases(translated_input)
                else:
                    potential_matches = find_matching_diseases(translated_input)              

                if not potential_matches:
                    final_response = f"I'm sorry {user_name}, I couldn't find a match. Please list your symptoms clearly."
                else:
                    top_score = potential_matches[0]['count']
                    if len(potential_matches) == 1 or (len(potential_matches) > 1 and top_score > potential_matches[1]['count']):
                        d_name = potential_matches[0]['disease']
                        details = get_disease_details(d_name)
                        final_response = f"Based on your symptoms {user_name}, you are most likely suffering from <b>{d_name}</b>.<br><br>"
                        if details:
                            final_response += f"<b>Description:</b> {details['description']}<br><b>Precautions:</b> {details['precautions']}"
                    else:
                        final_response = (f"Thank you, {user_name}. Your symptoms match multiple conditions equally. "
                                        f"You may be suffering from <b>{potential_matches[0]['disease']}</b> "
                                        f"or <b>{potential_matches[1]['disease']}</b>.<br><br>")
                        for item in potential_matches[:2]:
                            d_name = item['disease']
                            details = get_disease_details(d_name)
                            if details:
                                final_response += f"<b>{d_name}:</b> {details['description']}<br><b>Precautions:</b> {details['precautions']}<br><br>"

        # --- STEP C: Translate Output back to User's Language ---
        if output_lang != "en":
            if output_lang == "pa":
                # Step 1: English → Punjabi (Gurmukhi)
                punjabi_text = GoogleTranslator(source="en", target="pa").translate(final_response)

                try:
                    # Step 2: Gurmukhi → Shahmukhi
                    translated_output = transliterate.process('Gurmukhi', 'Shahmukhi', punjabi_text)
                except:
                    # fallback if conversion fails
                    translated_output = punjabi_text 
            else:                   
                try:
                # Use source="en" and let the library handle the target
                   translated_output = GoogleTranslator(source="en", target=output_lang).translate(final_response)
                except Exception as e:
                    print(f"Primary Translation Error: {e}")
                # Fallback to Urdu if regional codes fail
                    translated_output = GoogleTranslator(source="en", target="ur").translate(final_response)
        else:
            translated_output = final_response
        
        return translated_output
  
    except Exception as e:
        print(f"Translation/Processing Error: {e}")
        # Localized fallback messages
        fallback = "I'm sorry, I encountered an error. Please list your symptoms clearly."

        if output_lang == "ur":
            fallback = "معذرت، مجھے غلطی کا سامنا کرنا پڑا۔ براہ کرم اپنی علامات واضح طور پر لکھیں۔"
        elif output_lang == "pa":
            fallback = "معذرت، مینوں اک غلطی دا سامنا کرنا پیا۔ براہ مہربانی اپنیاں علامات واضح طور تے لکھو۔"
        elif output_lang == "skr":
            fallback = "معذرت، میکوں ہک غلطی دا سامݨا کرݨا پئے گیا۔ مہربانی کر تے اپݨیاں علامتاں صاف صاف لکھو۔"

        return fallback
    


@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"response": "No file uploaded."})
        
    file = request.files['file']
    file_path = "temp_prediction.jpg"
    file.save(file_path)
    
    # 1. Image Preprocessing (Matches your training setup)
    img = cv2.imread(file_path)
    if img is None:
        return jsonify({"response": "Error reading image."})
    
    # Remove pink grid (HSV mask)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
    img[mask > 0] = [255, 255, 255]
    
    # Prepare for MobileNetV2 (RGB, 224x224, Normalized)
    processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    processed_img = cv2.resize(processed_img, (224, 224)) / 255.0
    processed_img = np.expand_dims(processed_img, axis=0)
    
    # 2. Predict using the 4-class model
    predictions = vision_model.predict(processed_img)[0]
    class_idx = np.argmax(predictions)
    confidence = float(predictions[class_idx] * 100)
    
    # Map indices to your trained labels
    classes = [
        'Abnormal Heartbeat', 
        'Myocardial Infarction', 
        'Normal', 
        'Post-MI History'
    ]
    result = classes[class_idx]
    session['ecg_status'] = result

    # 3. Custom Guidance based on the 4 classes
    if result == "Normal":
        guidance = ("Your ECG results appear within the normal range. To maintain this, keep your diet healthy, exercise regularly, and avoid excessive salt or processed sugars.<br><br> Note: This AI-generated analysis is for educational and screening purposes only. It is not a substitute for professional medical advice. Always consult a qualified cardiologist for a definitive diagnosis.")
    elif result == "Myocardial Infarction":
        guidance = ("<b>CRITICAL:</b> Our AI suggests patterns consistent with a Myocardial Infarction (Heart Attack). "
                    "Irregularities detected in your ECG. Please provide your <b>Full Name</b> and describe any <b>symptoms</b> you are feeling (like chest pain or shortness of breath)ss your specific risk level." 
                   "<br><br><b>Note:</b> This is for educational screening using AI. Please seek emergency medical care immediately. Consult a cardiologist for a definitive diagnosis.")
    elif result == "Abnormal Heartbeat":
        guidance = ("Irregularities (Arrhythmia) detected. Please describe any symptoms you are feeling.<br> "
                    "Irregularities detected in your ECG. Please provide your <b>Full Name</b> and describe any <b>symptoms</b> you are feeling (like chest pain or shortness of breath)ss your specific risk level." 
                   "<br><br><b>Note:</b> This is for educational screening using AI. It is recommended to consult a cardiologist for a stress test or Holter monitoring.")
    else: # Post-MI History
        guidance = ("Patterns suggest a history of heart issues or previous Myocardial Infarction.Ensure you are following your prescribed recovery plan and regular checkups. "
                    "Irregularities detected in your ECG. Please provide your <b>Full Name</b> and describe any <b>symptoms</b> you are feeling (like chest pain or shortness of breath)ss your specific risk level." 
                   "<br><br><b>Note:</b> This is for educational screening using AI. Consult a cardiologist for a definitive diagnosis.")

    # NEW: Add translation logic before returning
    output_lang = session.get("output_lang", "en")
    response_text = f"<b>Result:</b> {result}<br><b>Confidence:</b> {confidence:.2f}%<br><br>{guidance}"

    if output_lang != "en":
        try:
            if output_lang == "pa":
                # Step 1: English → Punjabi (Gurmukhi)
                res_gurmukhi = GoogleTranslator(source="en", target="pa").translate(result)
                guidance_gurmukhi = GoogleTranslator(source="en", target="pa").translate(guidance)

                try:
                    # Step 2: Gurmukhi → Shahmukhi (The script used in Multan/Pakistan)
                    translated_result = transliterate.process('Gurmukhi', 'Shahmukhi', res_gurmukhi)
                    translated_guidance = transliterate.process('Gurmukhi', 'Shahmukhi', guidance_gurmukhi)
                except:
                    translated_result = res_gurmukhi
                    translated_guidance = guidance_gurmukhi
            else:
                # Regular translation for Urdu or other languages
                translated_result = GoogleTranslator(source="en", target=output_lang).translate(result)
                translated_guidance = GoogleTranslator(source="en", target=output_lang).translate(guidance)
            
            # CRITICAL: Re-build the final strings using the translated variables
            result = translated_result
            guidance = translated_guidance
            
            # Localization of the labels based on language
            label_result = "نتیجہ" if output_lang in ["ur", "pa", "skr"] else "Result"
            label_conf = "اعتماد" if output_lang in ["ur", "pa", "skr"] else "Confidence"
            
            response_text = (f"<b>{label_result}:</b> {result}<br>"
                             f"<b>{label_conf}:</b> {confidence:.2f}%<br><br>"
                             f"{guidance}")

        except Exception as e:
            print(f"Translation error: {e}")
            response_text = f"<b>Result:</b> {result}<br><b>Confidence:</b> {confidence:.2f}%<br><br>{guidance}"
    else:
        # Default English response
        response_text = f"<b>Result:</b> {result}<br><b>Confidence:</b> {confidence:.2f}%<br><br>{guidance}"

    return jsonify({
        "result": result,
        "confidence": round(confidence, 2),
        "guidance": guidance,
        "response": response_text
    })
    
    # return jsonify({"response": "ECG Vision model is not loaded yet. Please complete training."})
@app.route('/sw.js')
def service_worker():
    return app.send_static_file('sw.js'), 200, {
        'Content-Type': 'application/javascript'
    }

if __name__ == "__main__":
    print("\n[SYSTEM] Initializing Heart Disease Model...")
    train_lab_model()
    app.run(host='0.0.0.0', port=5000, debug=True)