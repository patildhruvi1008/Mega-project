import os
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
import os
file_path = os.path.join(os.getcwd(), "dataset", "branch_predictor_dataset_modified.csv")
print(f"Looking for dataset at: {file_path}")

if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}.")
    exit()

import pandas as pd
df = pd.read_csv(file_path)

# Encode categorical variables
label_encoders = {}
categorical_cols = ['Category', 'Branch']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Select input features and target
X = df[['Category', 'JEE Marks', '10th Marks', '12th Marks']]
y = df['Branch']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train KNN Model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2f}")

# Save model
with open("models/branch_predictor.pkl", "wb") as file:
    pickle.dump(model, file)
