import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

CHUNK_SIZE = 50000  # Define the chunk size to process

# File paths
filepaths = [
    'D:/Final Year/Advanced topic in CS/partB/data/UNSW-NB15_1.csv',
    'D:/Final Year/Advanced topic in CS/partB/data/UNSW-NB15_2.csv',
    'D:/Final Year/Advanced topic in CS/partB/data/UNSW-NB15_3.csv',
    'D:/Final Year/Advanced topic in CS/partB/data/UNSW-NB15_4.csv'
]

# Columns to keep based on the training/testing sets
columns_to_keep = [
    'dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl',
    'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'dwin', 'stcpb',
    'dtcpb', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd',
    'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm',
    'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'trans_depth', 'response_body_len', 'label'
]

# Mapping for renaming columns
column_mapping = {
    '59.166.0.0': 'srcip',
    '149.171.126.6': 'dstip',
    '53': 'sport',
    'udp': 'dport',
    '1421927414': 'stime',
    'dns': 'service',
    # Add more mappings based on your findings
}

def load_and_concatenate_data(filepaths):
    chunks = []
    for filepath in filepaths:
        print(f"Loading data from {filepath}")
        for chunk in pd.read_csv(filepath, chunksize=CHUNK_SIZE, low_memory=False):
            chunks.append(chunk)
    data = pd.concat(chunks, ignore_index=True)
    return data

def check_duplicates(data):
    duplicates = data.duplicated().sum()
    print(f"Number of duplicate rows: {duplicates}")
    if duplicates > 0:
        data = data.drop_duplicates()
    return data

def preprocess_data(data):
    # Rename columns to standardize
    data.rename(columns=column_mapping, inplace=True)
    
    # Check for columns to keep
    existing_columns = [col for col in columns_to_keep if col in data.columns]
    missing_columns = set(columns_to_keep) - set(existing_columns)
    
    if missing_columns:
        print(f"Missing columns that will be ignored: {missing_columns}")
    
    data = data[existing_columns]
    
    data = check_duplicates(data)
    
    # Assuming 'Label' column is generated here
    data['label'] = 1  # Placeholder for actual label logic
    
    return data

def process_files(filepaths):
    X_list = []
    
    for filepath in filepaths:
        print(f"Processing file {filepath}")
        for chunk in pd.read_csv(filepath, chunksize=CHUNK_SIZE, low_memory=False):
            X_chunk = preprocess_data(chunk)
            if not X_chunk.empty:
                X_list.append(X_chunk)
    
    if not X_list:
        raise ValueError("No data to concatenate after processing chunks.")
    
    X = pd.concat(X_list, ignore_index=True)
    
    print(f"Final data shape - X: {X.shape}")
    return X

def save_data(X):
    joblib.dump(X, 'processed_data2.pkl')
    print("Data saved to 'processed_data2.pkl'")

if __name__ == "__main__":
    X = process_files(filepaths)
    save_data(X)
    print("Data processing complete and saved.")
