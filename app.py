import os
import pickle
import traceback
import csv
import datetime
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from PIL import Image
import io

# --- FLASK CONFIG ---
app = Flask(__name__, template_folder='.')
CORS(app)

# --- GLOBAL VARIABLES ---
kmeans_model = None
interaction_df = None
drug_encoder = LabelEncoder()
all_drugs = []
adr_category_model = None
adr_drug_vectorizer = None
adr_label_encoder = None
skin_model = None
skin_class_names = None

# --- MEDICAL KNOWLEDGE BASE (SKIN) ---
SKIN_CONDITION_DB = {
    'acne': {
        'condition': 'Acne Vulgaris',
        'desc': 'Clogged hair follicles under the skin causing pimples or cysts.',
        'otc': ['Benzoyl Peroxide Gel', 'Salicylic Acid Cleanser', 'Adapalene (Differin)'],
        'marketed_drugs': ['Accutane (Isotretinoin)', 'Tretinoin Cream', 'Clindamycin Phosphate'],
        'urgency': 'Low - Treat at home',
        'color': 'green'
    },
    'eczema': {
        'condition': 'Eczema / Atopic Dermatitis',
        'desc': 'Inflammation causing itchy, red, swollen, and cracked skin.',
        'otc': ['Hydrocortisone Cream (1%)', 'Cerave/Cetaphil Moisturizers', 'Oral Antihistamines'],
        'marketed_drugs': ['Dupixent (Dupilumab)', 'Eucrisa (Crisaborole)', 'Protopic (Tacrolimus)'],
        'urgency': 'Moderate - See doctor if infection occurs',
        'color': 'yellow'
    },
    'tinea': {
        'condition': 'Tinea (Ringworm/Athlete\'s Foot)',
        'desc': 'Fungal infection causing a ring-shaped rash or itchy feet.',
        'otc': ['Clotrimazole (Lotrimin)', 'Terbinafine (Lamisil)', 'Miconazole'],
        'marketed_drugs': ['Griseofulvin', 'Fluconazole', 'Itraconazole'],
        'urgency': 'Low - Highly contagious but treatable',
        'color': 'orange'
    },
    'ringworm': {
        'condition': 'Tinea (Ringworm)',
        'desc': 'Fungal infection causing a ring-shaped rash.',
        'otc': ['Clotrimazole Cream', 'Terbinafine Cream'],
        'marketed_drugs': ['Griseofulvin', 'Fluconazole'],
        'urgency': 'Low',
        'color': 'orange'
    },
    'wart': {
        'condition': 'Viral Warts',
        'desc': 'Small, grainy skin growths caused by HPV.',
        'otc': ['Salicylic Acid Pads', 'Cryotherapy Kits (Freeze-off)'],
        'marketed_drugs': ['Imiquimod (Aldara)', 'Cantharidin'],
        'urgency': 'Low',
        'color': 'green'
    },
    'melanoma': {
        'condition': 'Melanoma (Skin Cancer)',
        'desc': 'The most serious type of skin cancer. Asymmetrical moles with irregular borders.',
        'otc': ['NONE - SEEK IMMEDIATE MEDICAL ATTENTION'],
        'marketed_drugs': ['Opdivo (Nivolumab)', 'Keytruda (Pembrolizumab)', 'Yervoy (Ipilimumab)'],
        'urgency': 'CRITICAL - Immediate Attention Required',
        'color': 'red'
    },
    'psoriasis': {
        'condition': 'Psoriasis',
        'desc': 'Autoimmune disease causing red, itchy scaly patches.',
        'otc': ['Coal Tar Shampoo', 'Salicylic Acid', 'Hydrocortisone'],
        'marketed_drugs': ['Humira (Adalimumab)', 'Otezla', 'Cosentyx'],
        'urgency': 'Moderate',
        'color': 'yellow'
    }
}

# --- HELPER FUNCTIONS ---
def analyze_effect_type(description):
    desc_lower = description.lower()
    agonist_keywords = ["increase", "enhance", "higher", "elevate", "synerg", "accumulate", "risk", "severity"]
    antagonist_keywords = ["decrease", "reduce", "lower", "inhibit", "diminish", "lessen", "efficacy"]

    score = 0
    for word in agonist_keywords:
        if word in desc_lower: score += 1
    for word in antagonist_keywords:
        if word in desc_lower: score -= 1
            
    if score > 0: return "Agonist (Effect/Toxicity Increased)"
    elif score < 0: return "Antagonist (Efficacy Reduced)"
    else: return "Complex/Unspecified Mechanism"

def match_skin_advice(predicted_label):
    label_lower = predicted_label.lower()
    for key in SKIN_CONDITION_DB.keys():
        if key in label_lower:
            return SKIN_CONDITION_DB[key]
    
    if "basal cell" in label_lower: return SKIN_CONDITION_DB.get('melanoma')
    if "nevus" in label_lower or "nevi" in label_lower: 
        return {
            'condition': 'Benign Nevus (Mole)',
            'desc': 'Common mole. Usually harmless but monitor for changes.',
            'otc': ['Monitor for changes', 'Sunscreen'],
            'marketed_drugs': ['N/A'],
            'urgency': 'Low',
            'color': 'green'
        }
    
    return {
        'condition': predicted_label,
        'desc': 'Consult a dermatologist for specific diagnosis.',
        'otc': ['Keep area clean'],
        'marketed_drugs': ['Consult Doctor'],
        'urgency': 'Unknown',
        'color': 'gray'
    }

def load_resources():
    global kmeans_model, interaction_df, all_drugs, drug_encoder
    global adr_category_model, adr_drug_vectorizer, adr_label_encoder
    global skin_model, skin_class_names

    print("--- Loading Resources ---")
    try:
        interaction_df = pd.read_csv('final_df_for_lookup.csv')
        with open('kmeans_model.pkl', 'rb') as f: kmeans_model = pickle.load(f)
        with open('drug_encoder.pkl', 'rb') as f: 
            drug_encoder = pickle.load(f)
            if hasattr(drug_encoder, 'classes_'): all_drugs = drug_encoder.classes_.tolist()

        with open('adr_category_model.pkl', 'rb') as f: adr_category_model = pickle.load(f)
        with open('adr_drug_vectorizer.pkl', 'rb') as f: adr_drug_vectorizer = pickle.load(f)
        with open('adr_label_encoder.pkl', 'rb') as f: adr_label_encoder = pickle.load(f)

        if os.path.exists('medical_skin_model.h5'):
            skin_model = tf.keras.models.load_model('medical_skin_model.h5')
            print("Skin Model Loaded.")
        
        if os.path.exists('class_indices.npy'):
            skin_class_names = np.load('class_indices.npy', allow_pickle=True)
            if skin_class_names.ndim == 0: skin_class_names = skin_class_names.item()
            
        print("All resources loaded successfully.")
    except Exception as e:
        print(f"Error loading resources: {e}")

load_resources()

# --- ROUTES ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/drugs', methods=['GET'])
def get_drugs():
    return jsonify({"drugs": all_drugs})

@app.route('/submit-feedback', methods=['POST'])
def submit_feedback():
    try:
        data = request.json
        message = data.get('message', '')
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if message:
            # Check if file exists to write header
            file_exists = os.path.isfile('user_feedback.csv')
            
            with open('user_feedback.csv', 'a', newline='') as csvfile:
                fieldnames = ['timestamp', 'feedback']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                if not file_exists:
                    writer.writeheader()
                
                writer.writerow({'timestamp': timestamp, 'feedback': message})
                
            return jsonify({"status": "success", "message": "Feedback saved"})
        return jsonify({"status": "error", "message": "Empty message"}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict_interaction():
    try:
        data = request.json
        d1, d2 = data.get('drug1'), data.get('drug2')
        
        match = interaction_df[
            ((interaction_df['drug1'].str.lower() == d1.lower()) & (interaction_df['drug2'].str.lower() == d2.lower())) |
            ((interaction_df['drug1'].str.lower() == d2.lower()) & (interaction_df['drug2'].str.lower() == d1.lower()))
        ]

        if not match.empty:
            row = match.iloc[0]
            desc = row['interaction']
            cluster = int(row['interaction_cluster'])
            effect = analyze_effect_type(desc)
            return jsonify({
                "prediction": cluster,
                "description": desc,
                "effectType": effect,
                "risk": "High" if effect.startswith("Agonist") or cluster in [1,3,6,7] else "Medium",
                "source": "Verified Database"
            })
        
        d1_enc = drug_encoder.transform([d1])[0]
        d2_enc = drug_encoder.transform([d2])[0]
        cluster = int(kmeans_model.predict([[d1_enc, d2_enc]])[0])
        return jsonify({
            "prediction": cluster,
            "description": f"Predicted interaction in cluster {cluster}. Monitor patient closely.",
            "effectType": "Potential Interaction (ML Predicted)",
            "risk": "Medium",
            "source": "AI Prediction"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict-skin', methods=['POST'])
def predict_skin():
    if 'file' not in request.files: return jsonify({'error': 'No file'}), 400
    
    try:
        file = request.files['file']
        img = Image.open(file.stream)
        if img.mode != 'RGB': img = img.convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        predictions = skin_model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        top_idx = np.argmax(score)
        confidence = float(np.max(score) * 100)
        
        raw_label = str(top_idx)
        if skin_class_names is not None:
            if isinstance(skin_class_names, (list, np.ndarray)):
                if top_idx < len(skin_class_names): raw_label = str(skin_class_names[top_idx])
            elif isinstance(skin_class_names, dict):
                inv_map = {v: k for k, v in skin_class_names.items()}
                raw_label = inv_map.get(top_idx, str(top_idx))

        advice = match_skin_advice(raw_label)
        return jsonify({
            "prediction": advice['condition'],
            "confidence": f"{confidence:.2f}%",
            "description": advice['desc'],
            "urgency": advice['urgency'],
            "otc": advice['otc'],
            "marketed_drugs": advice['marketed_drugs'],
            "color_code": advice['color']
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/adr-from-text', methods=['POST'])
def predict_adr():
    try:
        text = request.json.get('text', '')
        vec = adr_drug_vectorizer.transform([text])
        pred_idx = adr_category_model.predict(vec)[0]
        category = adr_label_encoder.inverse_transform([pred_idx])[0]
        return jsonify({
            "detections": [
                {"drug": "Based on text context", "reaction": category}
            ]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)