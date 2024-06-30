import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import os

CHUNK_SIZE = 1000  # Adjust this if necessary

def load_data_in_chunks(filename):
    with open(filename, 'rb') as f:
        while True:
            try:
                chunk = joblib.load(f)
                yield chunk
            except EOFError:
                break

def preprocess_chunk(chunk, label_encoders):
    for col in chunk.select_dtypes(include=['object']).columns:
        if col not in label_encoders:
            label_encoders[col] = LabelEncoder()
        chunk[col] = label_encoders[col].fit_transform(chunk[col].astype(str))
    
    for col in chunk.columns:
        if chunk[col].dtype == 'object':
            chunk[col] = chunk[col].apply(lambda x: int(x, 16) if isinstance(x, str) and x.startswith('0x') else x)
    return chunk

def train_and_evaluate():
    if not os.path.exists('results2'):
        os.makedirs('results2')

    clf = SGDClassifier(loss='log_loss', random_state=42)
    scaler = StandardScaler()
    imputer = SimpleImputer(strategy='mean')
    label_encoders = {}
    X_test, y_test = None, None

    # To check class distribution
    class_counts = {0: 0, 1: 0}

    for i, chunk in enumerate(load_data_in_chunks('processed_data2.pkl')):
        chunk = preprocess_chunk(chunk, label_encoders)
        y_chunk = chunk.pop('Label')
        
        class_counts[0] += (y_chunk == 0).sum()
        class_counts[1] += (y_chunk == 1).sum()

        chunk = imputer.fit_transform(chunk)
        if i == 0:
            scaler.partial_fit(chunk)
        else:
            scaler.partial_fit(chunk)
        
        chunk = scaler.transform(chunk)
        clf.partial_fit(chunk, y_chunk, classes=[0, 1])
        
        X_test, y_test = chunk, y_chunk

    y_pred = clf.predict(X_test)
    
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv('results2/classification_report.csv', index=True)
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('results2/confusion_matrix.png')

    print(f"Classification Report:\n{df_report}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Class Distribution:\n{class_counts}")

if __name__ == "__main__":
    train_and_evaluate()
