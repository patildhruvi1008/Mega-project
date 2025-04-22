import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# Load dataset
file_path = "C:/Users/kaneb/OneDrive/Desktop/FINAL YEAR MEGA PROJECT/branch_predictor_dataset_modified.csv"

# Check if the file exists
if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}. Please check the file path.")
    exit()

df = pd.read_csv(file_path)

# Encode categorical variables
label_encoders = {}
categorical_cols = ['Category', 'Branch']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Ask user for exam type
exam_type = input("Enter the exam type (JEE or MHT-CET): ").strip().upper()
if exam_type == "JEE":
    score_column = "JEE Marks"
elif exam_type == "MHT-CET":
    score_column = "MHT-CET Marks"
else:
    print("Invalid exam type. Please enter either 'JEE' or 'MHT-CET'.")
    exit()

# Define features and target
X = df[['Category', score_column, '10th Marks', '12th Marks']]
y = df['Branch']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train KNN model
k = 5  # You can tune this value
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_encoders['Branch'].classes_)

# Display metrics
print("\nModel Performance Metrics:")
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(report)

# Take user input
student_category_input = input("Enter Student Category (e.g., SC, OBC, General, etc.): ").strip()

# Check if category is valid
if student_category_input not in label_encoders['Category'].classes_:
    print("Invalid category. Please enter a valid Student Category from the dataset.")
    print(f"Valid categories: {list(label_encoders['Category'].classes_)}")
    exit()

# Encode category input
encoded_category = label_encoders['Category'].transform([student_category_input])[0]

# Get numeric inputs with validation
try:
    exam_score_input = float(input(f"Enter {exam_type} Score: "))
    tenth_marks_input = float(input("Enter 10th Marks: "))
    twelfth_marks_input = float(input("Enter 12th Marks: "))
except ValueError:
    print("Invalid input. Please enter numeric values for scores.")
    exit()

# Convert input to DataFrame
input_data = pd.DataFrame([[encoded_category, exam_score_input, tenth_marks_input, twelfth_marks_input]],
                          columns=['Category', score_column, '10th Marks', '12th Marks'])

# Predict
predicted_branch_encoded = model.predict(input_data)[0]

# Decode prediction
predicted_branch = label_encoders['Branch'].inverse_transform([predicted_branch_encoded])[0]

# Output result
print(f"\nPredicted Preferred Branch: {predicted_branch}")
