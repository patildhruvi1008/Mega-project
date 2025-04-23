from flask import Flask, render_template_string, request
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load and preprocess the dataset
file_path = "branch_predictor_dataset_modified.csv"
df = pd.read_csv(file_path)

label_encoders = {}
categorical_cols = ['Category', 'Branch']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Model Training
def train_model(score_column):
    X = df[['Category', score_column, '10th Marks', '12th Marks']]
    y = df['Branch']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    return model

@app.route('/', methods=['GET', 'POST'])
def predict_branch():
    prediction = None
    if request.method == 'POST':
        exam_type = request.form['exam_type']
        category = request.form['category']
        score = float(request.form['score'])
        tenth = float(request.form['tenth'])
        twelfth = float(request.form['twelfth'])

        score_column = "JEE Marks" if exam_type == "JEE" else "MHT-CET Marks"
        max_score = 300 if exam_type == "JEE" else 200

        if category not in label_encoders['Category'].classes_:
            prediction = "❌ Invalid category!"
        elif not (0 <= score <= max_score and 0 <= tenth <= 100 and 0 <= twelfth <= 100):
            prediction = "❌ One or more inputs are out of valid range!"
        elif (exam_type == "MHT-CET" and 
              ((category.lower() == "general" and score < 90) or 
               (category.lower() != "general" and score < 80))):
            prediction = "❌ MHT-CET cutoff not met!"
        elif (exam_type == "JEE" and 
              (category.lower() == "general" and score < 90)):
            prediction = "❌ JEE cutoff not met!"
        else:
            model = train_model(score_column)
            encoded_category = label_encoders['Category'].transform([category])[0]
            input_data = pd.DataFrame([[encoded_category, score, tenth, twelfth]], 
                                      columns=['Category', score_column, '10th Marks', '12th Marks'])
            predicted_branch_encoded = model.predict(input_data)[0]
            predicted_branch = label_encoders['Branch'].inverse_transform([predicted_branch_encoded])[0]
            prediction = predicted_branch

    return render_template_string('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
