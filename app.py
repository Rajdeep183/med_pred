import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import warnings
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
warnings.filterwarnings('ignore')

# Import SmartPreprocessor from separate module to avoid circular imports
from preprocessor import SmartPreprocessor

# Enhanced model loading function
@st.cache_resource
def load_model():
    try:
        # Try enhanced model first
        with open('best_model.pkl', 'rb') as f:
            best_model = pickle.load(f)
        with open('preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        return best_model, preprocessor, True
    except FileNotFoundError:
        try:
            # Fallback to basic SVC model
            with open('svc.pkl', 'rb') as f:
                basic_model = pickle.load(f)
            return basic_model, None, False
        except FileNotFoundError:
            st.error("❌ No model files found! Please run the training script first.")
            st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

# Load data
@st.cache_data
def load_data():
    try:
        data = {
            'precautions': pd.read_csv("precautions_df.csv"),
            'workout': pd.read_csv("workout_df.csv"),
            'description': pd.read_csv("description.csv"),
            'medications': pd.read_csv('medications.csv'),
            'diets': pd.read_csv("diets.csv")
        }
        return data
    except FileNotFoundError as e:
        st.error(f"❌ Error loading data files: {str(e)}")
        st.stop()

# Load model and data
model, preprocessor, is_enhanced = load_model()
data = load_data()
precautions = data['precautions']
workout = data['workout']
description = data['description']
medications = data['medications']
diets = data['diets']

# Define symptoms and diseases
symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}

diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

# Helper function to get disease details
def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc]) if not desc.empty else "Description not available"

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values] if not pre.empty else [["No precautions available"]]

    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med.values] if not med.empty else ["No medications available"]

    die = diets[diets['Disease'] == dis]['Diet']
    die = [die for die in die.values] if not die.empty else ["No diet information available"]

    wrkout = workout[workout['disease'] == dis]['workout']
    wrkout = [wrkout for wrkout in wrkout.values] if not wrkout.empty else ["No workout information available"]

    return desc, pre, med, die, wrkout

# Enhanced prediction function
def get_predicted_value_enhanced(patient_symptoms):
    # Create symptom vector
    input_vector = np.zeros(len(symptoms_dict))
    for symptom in patient_symptoms:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1
    
    # Create DataFrame for preprocessing
    input_df = pd.DataFrame([input_vector], columns=list(symptoms_dict.keys()))
    
    # Preprocess using saved pipeline
    input_processed = preprocessor.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_processed)[0]
    predicted_disease = diseases_list[prediction]
    
    # Get confidence and alternatives if model supports probabilities
    confidence = 0.0
    top_diseases = []
    
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(input_processed)[0]
        confidence = probabilities[prediction]
        
        # Get top 3 predictions
        top_indices = np.argsort(probabilities)[-3:][::-1]
        top_diseases = [(diseases_list[idx], probabilities[idx]) for idx in top_indices]
    
    return predicted_disease, confidence, top_diseases

# Basic prediction function (fallback)
def get_predicted_value_basic(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        if item in symptoms_dict:
            input_vector[symptoms_dict[item]] = 1
    return diseases_list[model.predict([input_vector])[0]], 0.0, []

# Set background
def get_base64(image_file):
    try:
        with open(image_file, "rb") as file:
            encoded_string = base64.b64encode(file.read()).decode()
        return encoded_string
    except FileNotFoundError:
        return None

def set_background(image_file):
    encoded_image = get_base64(image_file)
    if encoded_image:
        background_style = f"""
        <style>
        .stApp {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        .stSelectbox > div > div {{
            background-color: rgba(255, 255, 255, 0.1);
            color: white;
        }}
        .stButton > button {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 20px;
            padding: 0.5rem 1rem;
            font-weight: bold;
        }}
        </style>
        """
        st.markdown(background_style, unsafe_allow_html=True)

# Set the background
set_background("background.jpg")

# Enhanced CSS for better visibility and modern design
st.markdown("""
<style>
/* Main app styling */
.stApp {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    color: #2c3e50;
}

/* Main header styling */
.main-header {
    font-size: 3.5rem;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 2rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    padding: 1rem 0;
}

/* Status cards with better visibility */
.status-enhanced {
    background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    color: #155724;
    padding: 20px;
    border-radius: 15px;
    margin: 20px 0;
    border: 2px solid #28a745;
    box-shadow: 0 8px 32px rgba(40, 167, 69, 0.2);
    font-size: 1.1rem;
    font-weight: 600;
}

.status-basic {
    background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
    color: #856404;
    padding: 20px;
    border-radius: 15px;
    margin: 20px 0;
    border: 2px solid #ffc107;
    box-shadow: 0 8px 32px rgba(255, 193, 7, 0.2);
    font-size: 1.1rem;
    font-weight: 600;
}

/* Prediction card with enhanced visibility */
.prediction-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 25px;
    border-radius: 20px;
    color: white;
    margin: 25px 0;
    box-shadow: 0 15px 35px rgba(102, 126, 234, 0.3);
    border: 3px solid rgba(255, 255, 255, 0.2);
    text-align: center;
}

.prediction-card h3 {
    font-size: 1.5rem;
    margin-bottom: 10px;
    color: #e8f4fd;
}

.prediction-card h2 {
    font-size: 2.2rem;
    font-weight: 900;
    margin: 15px 0;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

/* Confidence indicators with better visibility */
.confidence-high { 
    color: #28a745; 
    font-weight: 900; 
    font-size: 1.3rem;
    background: rgba(40, 167, 69, 0.1);
    padding: 8px 15px;
    border-radius: 25px;
    display: inline-block;
    margin: 10px 0;
    border: 2px solid #28a745;
}

.confidence-medium { 
    color: #fd7e14; 
    font-weight: 900; 
    font-size: 1.3rem;
    background: rgba(253, 126, 20, 0.1);
    padding: 8px 15px;
    border-radius: 25px;
    display: inline-block;
    margin: 10px 0;
    border: 2px solid #fd7e14;
}

.confidence-low { 
    color: #dc3545; 
    font-weight: 900; 
    font-size: 1.3rem;
    background: rgba(220, 53, 69, 0.1);
    padding: 8px 15px;
    border-radius: 25px;
    display: inline-block;
    margin: 10px 0;
    border: 2px solid #dc3545;
}

/* Alternative predictions styling */
.alternative-predictions {
    background: rgba(255, 255, 255, 0.9);
    padding: 20px;
    border-radius: 15px;
    margin: 20px 0;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    border: 2px solid #e9ecef;
}

.alternative-predictions h3 {
    color: #495057;
    margin-bottom: 15px;
    font-size: 1.4rem;
    font-weight: 700;
}

/* Symptom selection area */
.symptom-container {
    background: rgba(255, 255, 255, 0.95);
    padding: 25px;
    border-radius: 20px;
    margin: 20px 0;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
    border: 2px solid #e9ecef;
}

.symptom-container h3 {
    color: #495057;
    font-size: 1.6rem;
    font-weight: 700;
    margin-bottom: 20px;
}

/* Results container */
.results-container {
    background: rgba(255, 255, 255, 0.95);
    padding: 25px;
    border-radius: 20px;
    margin: 20px 0;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
    border: 2px solid #e9ecef;
    min-height: 400px;
}

.results-container h3 {
    color: #495057;
    font-size: 1.6rem;
    font-weight: 700;
    margin-bottom: 20px;
}

/* Enhanced button styling */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 25px;
    padding: 15px 30px;
    font-weight: 700;
    font-size: 1.1rem;
    transition: all 0.3s ease;
    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 10px 5px;
}

.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(102, 126, 234, 0.4);
}

/* Multiselect styling */
.stMultiSelect > div > div {
    background-color: white;
    border: 2px solid #e9ecef;
    border-radius: 15px;
    padding: 10px;
    font-size: 1.1rem;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
}

.stMultiSelect > div > div > div {
    color: #495057;
    font-weight: 600;
}

/* Information buttons grid */
.info-buttons {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 15px;
    margin: 25px 0;
}

.info-button {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    border: 2px solid #dee2e6;
    border-radius: 15px;
    padding: 15px;
    text-align: center;
    transition: all 0.3s ease;
    cursor: pointer;
}

.info-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
}

/* Section headers */
.section-header {
    font-size: 1.8rem;
    font-weight: 800;
    color: #495057;
    margin: 30px 0 20px 0;
    padding-bottom: 10px;
    border-bottom: 3px solid #667eea;
    text-align: center;
}

/* Information display boxes */
.info-box {
    background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
    border: 2px solid #dee2e6;
    border-radius: 15px;
    padding: 20px;
    margin: 15px 0;
    box-shadow: 0 6px 25px rgba(0, 0, 0, 0.1);
}

.info-box h4 {
    color: #495057;
    font-size: 1.3rem;
    font-weight: 700;
    margin-bottom: 15px;
}

.info-box p, .info-box li {
    color: #6c757d;
    font-size: 1.1rem;
    line-height: 1.6;
    margin-bottom: 8px;
}

/* Disclaimer styling */
.disclaimer {
    background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
    border: 2px solid #ffc107;
    border-radius: 15px;
    padding: 25px;
    margin: 30px 0;
    text-align: center;
    font-size: 1.1rem;
    font-weight: 600;
    color: #856404;
    box-shadow: 0 8px 32px rgba(255, 193, 7, 0.2);
}

/* Responsive design */
@media (max-width: 768px) {
    .main-header {
        font-size: 2.5rem;
    }
    
    .prediction-card h2 {
        font-size: 1.8rem;
    }
    
    .symptom-container, .results-container {
        padding: 15px;
    }
    
    .stButton > button {
        padding: 12px 20px;
        font-size: 1rem;
    }
}

/* Sidebar styling */
.css-1d391kg {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Custom scrollbar */
::-webkit-scrollbar {
    width: 12px;
}

::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 10px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
}
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">🏥 AI Medical Diagnosis System</h1>', unsafe_allow_html=True)

# Display model status
if is_enhanced:
    st.markdown("""
    <div class="status-enhanced">
        ✅ <strong>Enhanced Model Active</strong> - Using advanced ML pipeline with 99.80% accuracy
        <br>• Feature Selection & Scaling
        <br>• Confidence Scoring
        <br>• Alternative Predictions
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="status-basic">
        ⚠️ <strong>Basic Model Active</strong> - Using fallback SVC model
        <br>• Standard prediction only
        <br>• Run training script for enhanced features
    </div>
    """, unsafe_allow_html=True)

# Create two columns for better layout with enhanced containers
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    <div class="symptom-container">
        <h3>🔍 Select Your Symptoms</h3>
    </div>
    """, unsafe_allow_html=True)
    
    symptoms_list = list(symptoms_dict.keys())
    selected_symptoms = st.multiselect(
        "Choose symptoms you're experiencing:",
        symptoms_list,
        help="Select multiple symptoms that match your condition"
    )
    
    # Predict button with enhanced styling
    predict_button = st.button("🔮 Predict Disease", type="primary")
    
    # Add symptom count display
    if selected_symptoms:
        st.markdown(f"""
        <div style="background: rgba(102, 126, 234, 0.1); padding: 15px; border-radius: 10px; margin: 15px 0; border: 2px solid #667eea;">
            <strong>Selected Symptoms: {len(selected_symptoms)}</strong>
            <br><small>Symptoms: {', '.join(selected_symptoms[:3])}{'...' if len(selected_symptoms) > 3 else ''}</small>
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="results-container">
        <h3>📊 Prediction Results</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'predicted_disease' not in st.session_state:
        st.session_state['predicted_disease'] = None
    if 'confidence' not in st.session_state:
        st.session_state['confidence'] = 0.0
    if 'top_diseases' not in st.session_state:
        st.session_state['top_diseases'] = []
    if 'desc' not in st.session_state:
        st.session_state['desc'] = None
    if 'pre' not in st.session_state:
        st.session_state['pre'] = None
    if 'med' not in st.session_state:
        st.session_state['med'] = None
    if 'die' not in st.session_state:
        st.session_state['die'] = None
    if 'wrkout' not in st.session_state:
        st.session_state['wrkout'] = None

    # Prediction logic
    if predict_button:
        if selected_symptoms:
            if is_enhanced:
                predicted_disease, confidence, top_diseases = get_predicted_value_enhanced(selected_symptoms)
            else:
                predicted_disease, confidence, top_diseases = get_predicted_value_basic(selected_symptoms)
            
            # Store in session state
            st.session_state['predicted_disease'] = predicted_disease
            st.session_state['confidence'] = confidence
            st.session_state['top_diseases'] = top_diseases
            
            # Get additional information
            desc, pre, med, die, wrkout = helper(predicted_disease)
            st.session_state['desc'] = desc
            st.session_state['pre'] = pre
            st.session_state['med'] = med
            st.session_state['die'] = die
            st.session_state['wrkout'] = wrkout
        else:
            st.warning("⚠️ Please select at least one symptom to get a prediction.")

    # Display results
    if st.session_state['predicted_disease']:
        st.markdown(f"""
        <div class="prediction-card">
            <h3>🏥 Predicted Disease</h3>
            <h2>{st.session_state['predicted_disease']}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        if is_enhanced and st.session_state['confidence'] > 0:
            confidence_pct = st.session_state['confidence'] * 100
            if confidence_pct >= 80:
                conf_class = "confidence-high"
            elif confidence_pct >= 60:
                conf_class = "confidence-medium"
            else:
                conf_class = "confidence-low"
                
            st.markdown(f"""
            <p class="{conf_class}">📈 Confidence: {confidence_pct:.1f}%</p>
            """, unsafe_allow_html=True)
            
            if st.session_state['top_diseases']:
                st.markdown("""
                <div class="alternative-predictions">
                    <h3>🏆 Alternative Predictions</h3>
                </div>
                """, unsafe_allow_html=True)
                
                for i, (disease, prob) in enumerate(st.session_state['top_diseases'], 1):
                    confidence_pct = prob * 100
                    if confidence_pct >= 80:
                        color = "#28a745"
                    elif confidence_pct >= 60:
                        color = "#fd7e14"
                    else:
                        color = "#dc3545"
                    
                    st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.9); padding: 12px; margin: 8px 0; border-radius: 10px; border-left: 4px solid {color}; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                        <strong style="color: {color};">{i}. {disease}</strong>
                        <div style="float: right; background: {color}; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.9rem; font-weight: bold;">
                            {confidence_pct:.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

# Information buttons in a new section
if st.session_state['predicted_disease']:
    st.markdown("---")
    st.subheader("📋 Detailed Medical Information")
    
    # Create buttons in columns
    info_col1, info_col2, info_col3 = st.columns(3)
    
    with info_col1:
        if st.button("📖 Description"):
            st.info(f"**Description:** {st.session_state['desc']}")
        
        if st.button("⚠️ Precautions"):
            st.warning("**Precautions to take:**")
            if st.session_state['pre'] and len(st.session_state['pre']) > 0:
                for i, p in enumerate(st.session_state['pre'][0], 1):
                    if p and p.strip():
                        st.write(f"{i}. {p}")
    
    with info_col2:
        if st.button("💊 Medications"):
            st.success("**Recommended Medications:**")
            for i, m in enumerate(st.session_state['med'], 1):
                if m and m.strip():
                    st.write(f"{i}. {m}")
        
        if st.button("🏃‍♂️ Workout"):
            st.info("**Suggested Workouts:**")
            for i, w in enumerate(st.session_state['wrkout'], 1):
                if w and w.strip():
                    st.write(f"{i}. {w}")
    
    with info_col3:
        if st.button("🥗 Diet Plan"):
            st.success("**Recommended Diet:**")
            for i, d in enumerate(st.session_state['die'], 1):
                if d and d.strip():
                    st.write(f"{i}. {d}")

# Footer disclaimer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    ⚠️ <strong>Medical Disclaimer:</strong> This system is for educational and informational purposes only. 
    Always consult with qualified healthcare professionals for proper medical advice, diagnosis, and treatment.
</div>
""", unsafe_allow_html=True)
