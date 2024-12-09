from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

app = Flask(__name__)

# Load and preprocess the dataset
df = pd.read_csv('dataset.csv')

# List of categorical columns to be encoded
categorical_cols = ['Disease', 'Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 
                    'Blood Pressure', 'Cholesterol Level', 'Outcome Variable', 'Gender']

# Create a LabelEncoder instance for each categorical column
label_encoders = {col: LabelEncoder() for col in categorical_cols}
categorical_dicts = {}

# Encode each categorical column
for col in categorical_cols:
    df[col] = label_encoders[col].fit_transform(df[col])
    categorical_dicts[col] = {label: value for label, value in zip(label_encoders[col].classes_, range(len(label_encoders[col].classes_)))}

# Prepare the feature columns (X) and target column (y)
X = df[categorical_cols[:-1]]  # all columns except the outcome
y = df['Outcome Variable']  # target column

# Train the Random Forest model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

@app.route('/')
def index():
    df = pd.read_csv('dataset.csv')
    # Render the HTML form with encoded lists passed as context
    Disease_lists = df['Disease'].unique().tolist()
    Fever_lists = df['Fever'].unique().tolist()
    Cough_lists = df['Cough'].unique().tolist()
    Fatigue_lists = df['Fatigue'].unique().tolist()
    Difficulty_Breathing_lists = df['Difficulty Breathing'].unique().tolist()
    Blood_Pressure_lists = df['Blood Pressure'].unique().tolist()
    Cholesterol_Level_lists = df['Cholesterol Level'].unique().tolist()
    Outcome_Variable_lists = df['Outcome Variable'].unique().tolist()
    Gender_lists = df['Gender'].unique().tolist()
    print(Outcome_Variable_lists)

    return render_template('disease_form.html', 
                           Disease_lists=Disease_lists, 
                           Fever_lists=Fever_lists, 
                           Cough_lists=Cough_lists, 
                           Fatigue_lists=Fatigue_lists, 
                           Difficulty_Breathing_lists=Difficulty_Breathing_lists, 
                           Blood_Pressure_lists=Blood_Pressure_lists, 
                           Cholesterol_Level_lists=Cholesterol_Level_lists,
                           Outcome_Variable_lists=Outcome_Variable_lists, 
                           Gender_lists=Gender_lists)


@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    disease = request.form['disease']
    fever = request.form['fever']
    cough = request.form['cough']
    fatigue = request.form['fatigue']
    difficulty_breathing = request.form['difficulty_breathing']
    blood_pressure = request.form['blood_pressure']
    cholesterol_level = request.form['cholesterol_level']
    age = int(request.form['age'])
    gender = request.form['gender']

    # Convert input values to encoded values using the label encoders
    def encode_feature(feature, encoder, default_value):
        try:
            return encoder.transform([feature])[0]
        except ValueError:
            return default_value

    disease_encoded = encode_feature(disease, label_encoders['Disease'], label_encoders['Disease'].classes_[0])
    fever_encoded = encode_feature(fever, label_encoders['Fever'], label_encoders['Fever'].classes_[0])
    cough_encoded = encode_feature(cough, label_encoders['Cough'], label_encoders['Cough'].classes_[0])
    fatigue_encoded = encode_feature(fatigue, label_encoders['Fatigue'], label_encoders['Fatigue'].classes_[0])
    difficulty_breathing_encoded = encode_feature(difficulty_breathing, label_encoders['Difficulty Breathing'], label_encoders['Difficulty Breathing'].classes_[0])
    blood_pressure_encoded = encode_feature(blood_pressure, label_encoders['Blood Pressure'], label_encoders['Blood Pressure'].classes_[0])
    cholesterol_level_encoded = encode_feature(cholesterol_level, label_encoders['Cholesterol Level'], label_encoders['Cholesterol Level'].classes_[0])
    gender_encoded = encode_feature(gender, label_encoders['Gender'], label_encoders['Gender'].classes_[0])

    # Handle missing or unexpected inputs in age (e.g., negative or zero)
    if age <= 0 and age>=60:
        age = 30  # Default to a neutral age value or use other domain knowledge

    # Construct input data for prediction
    input_data = np.array([[disease_encoded, fever_encoded, cough_encoded, fatigue_encoded,
                            difficulty_breathing_encoded, blood_pressure_encoded, cholesterol_level_encoded, gender_encoded]])

    # Make prediction using the trained Random Forest model
    prediction = rf.predict(input_data)[0]
    print("output is:",prediction)
    # Decode the prediction back to the original label
    outcome = label_encoders['Outcome Variable']

    if prediction == 0:
        outcome="safe to do work"
        return render_template('result.html', outcome=outcome)
    else:
        outcome="not safe to do work"
        return render_template('result1.html', outcome=outcome)
    

if __name__ == '__main__':
    app.run(debug=True)
