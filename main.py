import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

def load_data(train_filepath, test_filepath):
    # Load training and testing datasets
    train_data = pd.read_csv(train_filepath)
    test_data = pd.read_csv(test_filepath)
    return train_data, test_data

def preprocess_data(train_data, test_data):
    if 'id' in train_data.columns:
        train_data = train_data.drop('id', axis=1)
    if 'id' in test_data.columns:
        test_data = test_data.drop('id', axis=1)
    
    X_train = train_data.drop('label', axis=1)
    y_train = train_data['label']
    X_test = test_data.drop('label', axis=1)
    y_test = test_data['label']
    
    # Encoding categorical variables if any
    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)
    
    # Align the columns of test set to match training set
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

def save_data(X_train, X_test, y_train, y_test):
    joblib.dump((X_train, X_test, y_train, y_test), 'processed_data.pkl')

if __name__ == "__main__":
    train_filepath = 'data/UNSW_NB15_training-set.csv'
    test_filepath = 'data/UNSW_NB15_testing-set.csv'
    train_data, test_data = load_data(train_filepath, test_filepath)
    X_train, X_test, y_train, y_test = preprocess_data(train_data, test_data)
    save_data(X_train, X_test, y_train, y_test)
    print("Data processing complete and saved.")
