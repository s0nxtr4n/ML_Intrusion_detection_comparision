import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    X_train, X_test, y_train, y_test = joblib.load('processed_data.pkl')
    return X_train, X_test, y_train, y_test

def save_results_to_file(report, confusion_matrix, filename):
    with open(filename, 'w') as f:
        f.write(report)
        f.write('\nConfusion Matrix:\n')
        f.write(str(confusion_matrix))

def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f'results/{title}.png')
    plt.show()

def train_and_evaluate():
    X_train, X_test, y_train, y_test = load_data()
    
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)
    
    y_pred_rf = rf.predict(X_test)
    
    report = classification_report(y_test, y_pred_rf)
    cm = confusion_matrix(y_test, y_pred_rf)
    
    save_results_to_file(report, cm, 'results/random_forest_results.txt')
    
    plot_confusion_matrix(cm, 'Random Forest Confusion Matrix')

    print("Random Forest Classification Report:\n", report)
    print("Confusion Matrix:\n", cm)

if __name__ == "__main__":
    train_and_evaluate()
