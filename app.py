import pickle
import traceback
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.preprocessing import LabelEncoder
import numpy as np
import requests 
import os

# --- FLASK APP INITIALIZATION ---
app = Flask(__name__)
CORS(app)

# --- GLOBAL VARIABLES ---
kmeans_model = None
interaction_df = None
drug_encoder = LabelEncoder()
all_drugs = []
adr_category_model = None
adr_drug_vectorizer = None
adr_label_encoder = None

# --- Function to download files from a URL ---
def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename} from cloud storage...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status() # Raise an exception for bad status codes
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"✅ SUCCESS: Downloaded {filename}.")
        except requests.exceptions.RequestException as e:
            print(f"❌ FAILED to download {filename}. Error: {e}")
            return False
    else:
        print(f"'{filename}' already exists locally. Skipping download.")
    return True

# --- Define Cluster Themes & Risk Levels ---
cluster_themes = {
    0: "Metabolism Interference (Decreased)",
    1: "Serotonergic Effects (Increased Risk)",
    2: "General Adverse Effects (Increased Severity)",
    3: "Cardiovascular Risk (QTc Prolongation)",
    4: "Antihypertensive Activity Interference",
    5: "Metabolism Interference (Increased)",
    6: "CNS Depressant Effects (Increased Risk)",
    7: "Bleeding Risk",
    8: "Reduced Therapeutic Efficacy",
    9: "General Drug Concentration Increase"
}
cluster_risk_levels = {
    0: "Medium", 1: "High", 2: "Medium", 3: "High", 4: "Medium",
    5: "Medium", 6: "High", 7: "High", 8: "Low", 9: "Medium"
}


# --- LOAD MODELS AND DATA AT STARTUP ---
def load_resources():
    global kmeans_model, interaction_df, all_drugs
    global adr_category_model, adr_drug_vectorizer, adr_label_encoder
    
    # --- Define URL and download the large adr_category_model.pkl file ---
    ADR_MODEL_URL = "https://drive.google.com/uc?id=1yZClb3T57MKrx91kdU7N47cC9fjggu_3&export=download"
    download_file(ADR_MODEL_URL, 'adr_category_model.pkl')
    
    print("--- Loading all models and data ---")
    try:
        interaction_df = pd.read_csv('final_df_for_lookup.csv')
        interaction_df['drug1'] = interaction_df['drug1'].str.lower()
        interaction_df['drug2'] = interaction_df['drug2'].str.lower()
        print("✅ SUCCESS: Interaction dataset loaded.")

        all_drugs_list = pd.concat([interaction_df['drug1'], interaction_df['drug2']]).unique()
        drug_encoder.fit(all_drugs_list)
        all_drugs = drug_encoder.classes_.tolist()
        print(f"✅ SUCCESS: Drug encoder created with {len(all_drugs)} unique drugs.")

        with open('kmeans_model.pkl', 'rb') as f:
            kmeans_model = pickle.load(f)
        print("✅ SUCCESS: KMeans model loaded.")

        with open('adr_category_model.pkl', 'rb') as f:
            adr_category_model = pickle.load(f)
        print("✅ SUCCESS: ADR category model loaded.")

        with open('adr_drug_vectorizer.pkl', 'rb') as f:
            adr_drug_vectorizer = pickle.load(f)
        print("✅ SUCCESS: ADR drug vectorizer loaded.")

        with open('adr_label_encoder.pkl', 'rb') as f:
            adr_label_encoder = pickle.load(f)
        print("✅ SUCCESS: ADR label encoder loaded.")
        
        print("\n--- Backend is ready ---")

    except FileNotFoundError as e:
        print(f"❌ FATAL ERROR: Could not find a required file: {e}")
    except Exception as e:
        print(f"❌ FATAL ERROR during setup: {e}")

# --- API ENDPOINTS ---
@app.route('/')
def home():
    return jsonify({"status": "healthy", "message": "Backend is running."})

@app.route('/drugs')
def get_all_drugs():
    return jsonify(all_drugs)

@app.route('/interaction', methods=['GET'])
def get_interaction():
    drug1 = request.args.get('drug1', '').lower().strip()
    drug2 = request.args.get('drug2', '').lower().strip()

    if not drug1 or not drug2:
        return jsonify({"error": "Please provide both drug names."}), 400
    
    if drug1 not in drug_encoder.classes_ or drug2 not in drug_encoder.classes_:
        return jsonify({"error": "One or both drugs are not in the dataset."}), 404

    try:
        drug1_encoded = drug_encoder.transform([drug1])
        drug2_encoded = drug_encoder.transform([drug2])
        input_vector = np.array([drug1_encoded[0], drug2_encoded[0]]).reshape(1, -1)

        prediction = kmeans_model.predict(input_vector)[0]
        distances = kmeans_model.transform(input_vector)[0]
        
        sorted_distances = np.sort(distances)
        confidence_score = (1 - (sorted_distances[0] / sorted_distances[1])) * 100 if len(sorted_distances) > 1 else 100.0

        match_df = interaction_df[
            ((interaction_df['drug1'] == drug1) & (interaction_df['drug2'] == drug2)) |
            ((interaction_df['drug1'] == drug2) & (interaction_df['drug2'] == drug1))
        ]
        
        if not match_df.empty:
            example_description = match_df.iloc[0]['interaction']
        else:
            example_description = interaction_df[interaction_df['interaction_cluster'] == prediction].iloc[0]['interaction']
        
        prediction_result = {
            "riskLevel": cluster_risk_levels.get(prediction, "Unknown"),
            "confidenceScore": f"{confidence_score:.2f}%",
            "interactionTheme": f"{cluster_themes.get(prediction, 'Uncategorized')} (Cluster {prediction})",
            "description": example_description
        }
        return jsonify(prediction_result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Backend Error: {str(e)}"}), 500

@app.route('/adr-from-text', methods=['POST'])
def detect_adr_from_text():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Please provide text to analyze."}), 400
    text = data.get('text')
    try:
        if not all([adr_category_model, adr_drug_vectorizer, adr_label_encoder]):
            return jsonify({"error": "ADR Detector model components are not available."}), 500

        input_vector = adr_drug_vectorizer.transform([text])
        predicted_category_encoded = adr_category_model.predict(input_vector)
        predicted_drug_name = adr_label_encoder.inverse_transform(predicted_category_encoded)[0]
        
        detections_result = {
            "detections": [
                {"drug": predicted_drug_name, "reaction": "Reaction described in text"}
            ]
        }
        return jsonify(detections_result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Backend Error: {str(e)}"}), 500

# --- SCRIPT EXECUTION ---
if __name__ == '__main__':
    load_resources()
    app.run(debug=True, port=5000)
