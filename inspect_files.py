import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
features_df = pd.read_csv('data/NUSW-NB15_features.csv', encoding='latin1')
gt_df = pd.read_csv('data/NUSW-NB15_GT.csv', encoding='latin1')
testing_set_df = pd.read_csv('data/UNSW_NB15_testing-set.csv', encoding='latin1')
training_set_df = pd.read_csv('data/UNSW_NB15_training-set.csv', encoding='latin1')
unsw_nb15_1_df = pd.read_csv('data/UNSW-NB15_1.csv', encoding='latin1')
unsw_nb15_2_df = pd.read_csv('data/UNSW-NB15_2.csv', encoding='latin1')
unsw_nb15_3_df = pd.read_csv('data/UNSW-NB15_3.csv', encoding='latin1')
unsw_nb15_4_df = pd.read_csv('data/UNSW-NB15_4.csv', encoding='latin1')
list_events_df = pd.read_csv('data/UNSW-NB15_LIST_EVENTS.csv', encoding='latin1')

# Display the first few rows to understand the structure
print("Features DataFrame Head:\n", features_df.head())
print("Ground Truth DataFrame Head:\n", gt_df.head())
print("Training Set DataFrame Head:\n", training_set_df.head())
print("Testing Set DataFrame Head:\n", testing_set_df.head())
print("UNSW-NB15_1 DataFrame Head:\n", unsw_nb15_1_df.head())
print("UNSW-NB15_2 DataFrame Head:\n", unsw_nb15_2_df.head())
print("UNSW-NB15_3 DataFrame Head:\n", unsw_nb15_3_df.head())
print("UNSW-NB15_4 DataFrame Head:\n", unsw_nb15_4_df.head())
print("List Events DataFrame Head:\n", list_events_df.head())

# Merge raw datasets if necessary
raw_data_df = pd.concat([unsw_nb15_1_df, unsw_nb15_2_df, unsw_nb15_3_df, unsw_nb15_4_df])

# Check for missing values and data types
print("Missing values in raw data:\n", raw_data_df.isnull().sum())
print("Data types in raw data:\n", raw_data_df.dtypes)

# Handle missing values if any
raw_data_df.fillna(method='ffill', inplace=True)

# Combine processed training and testing sets for further analysis
data_df = pd.concat([training_set_df, testing_set_df])

# Extract features and labels
X = data_df.drop(columns=['Label'])
y = data_df['Label']

# Split the combined data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize classifiers
dt_clf = DecisionTreeClassifier()
rf_clf = RandomForestClassifier()
svm_clf = SVC()

# Train classifiers
dt_clf.fit(X_train, y_train)
rf_clf.fit(X_train, y_train)
svm_clf.fit(X_train, y_train)

# Make predictions
dt_pred = dt_clf.predict(X_test)
rf_pred = rf_clf.predict(X_test)
svm_pred = svm_clf.predict(X_test)

# Evaluate classifiers
dt_report = classification_report(y_test, dt_pred)
rf_report = classification_report(y_test, rf_pred)
svm_report = classification_report(y_test, svm_pred)

dt_cm = confusion_matrix(y_test, dt_pred)
rf_cm = confusion_matrix(y_test, rf_pred)
svm_cm = confusion_matrix(y_test, svm_pred)

# Print classification reports
print("Decision Tree Classification Report:\n", dt_report)
print("Random Forest Classification Report:\n", rf_report)
print("SVM Classification Report:\n", svm_report)

# Plot confusion matrices
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
sns.heatmap(dt_cm, annot=True, fmt="d", cmap="Blues")
plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.subplot(1, 3, 2)
sns.heatmap(rf_cm, annot=True, fmt="d", cmap="Blues")
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.subplot(1, 3, 3)
sns.heatmap(svm_cm, annot=True, fmt="d", cmap="Blues")
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()
