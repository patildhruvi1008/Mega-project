import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# Load dataset
file_path = "branch_predictor_dataset_modified.csv"

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

# Select exam type
while True:
    exam_type = input("Enter the exam type (JEE or MHT-CET): ").strip().upper()
    if exam_type == "JEE":
        score_column = "JEE Marks"
        max_score = 300
        break
    elif exam_type == "MHT-CET":
        score_column = "MHT-CET Marks"
        max_score = 200
        break
    else:
        print("❌ Invalid exam type! Please enter either 'JEE' or 'MHT-CET'.")

# Train/Test Setup
X = df[['Category', score_column, '10th Marks', '12th Marks']]
y = df['Branch']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_encoders['Branch'].classes_)

print("\n✅ Model Trained")
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(report)

# Category input
while True:
    student_category_input = input("Enter Student Category (e.g., SC, OBC, General): ").strip()
    if student_category_input in label_encoders['Category'].classes_:
        encoded_category = label_encoders['Category'].transform([student_category_input])[0]
        print("✅ Category is Valid.")
        break
    else:
        print("❌ Invalid category. Try again.")

# Get exam marks with range validation
while True:
    try:
        exam_score_input = float(input(f"Enter {exam_type} Score (out of {max_score}): "))
        if 0 <= exam_score_input <= max_score:
            # Cutoff check
            if exam_type == "MHT-CET":
                if student_category_input.lower() == "general" and exam_score_input >= 90:
                    print("✅ Valid MHT-CET Score for General.")
                    break
                elif student_category_input.lower() != "general" and exam_score_input >= 80:
                    print("✅ Valid MHT-CET Score for Reserved.")
                    break
                else:
                    print("❌ Invalid: Does not meet MHT-CET cutoff requirement.")
            elif exam_type == "JEE":
                if student_category_input.lower() == "general" and exam_score_input >= 90:
                    print("✅ Valid JEE Score for General.")
                    break
                elif student_category_input.lower() != "general":
                    print("✅ Valid JEE Score for Reserved.")
                    break
                else:
                    print("❌ Invalid: General category requires 90+ in JEE.")
        else:
            print(f"❌ Invalid marks! Enter between 0 and {max_score}.")
    except ValueError:
        print("❌ Please enter a numeric value.")

# Get 10th marks
while True:
    try:
        tenth_marks_input = float(input("Enter 10th Marks (out of 100): "))
        if 0 <= tenth_marks_input <= 100:
            print("✅ 10th Marks are Valid.")
            break
        else:
            print("❌ Invalid! 10th Marks should be between 0 and 100.")
    except ValueError:
        print("❌ Please enter a numeric value.")

# Get 12th marks
while True:
    try:
        twelfth_marks_input = float(input("Enter 12th Marks (out of 100): "))
        if 0 <= twelfth_marks_input <= 100:
            print("✅ 12th Marks are Valid.")
            break
        else:
            print("❌ Invalid! 12th Marks should be between 0 and 100.")
    except ValueError:
        print("❌ Please enter a numeric value.")

# Predict
input_data = pd.DataFrame([[encoded_category, exam_score_input, tenth_marks_input, twelfth_marks_input]],
                          columns=['Category', score_column, '10th Marks', '12th Marks'])

predicted_branch_encoded = model.predict(input_data)[0]
predicted_branch = label_encoders['Branch'].inverse_transform([predicted_branch_encoded])[0]

print(f"\n🎓 Predicted Preferred Branch: {predicted_branch}")
