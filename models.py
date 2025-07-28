#Authors: Serban Voinea Gabreanu, Gur Rehmat Singh Chahal, Algoma University, COSC5906002 Advanced Topics of Computer Networks (25SP), Final Project.
#models.py: This script contains the definitions for the deep learning models used in the project, the label mappings, and some of the dataset processing code.

import os
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

#Used in preprocess and save data function to let the script process big files with less memory.
CHUNK_SIZE = 50000

### Model Definitions ###

#MLP Model that uses Batch Normalization, ReLU activation, and Dropout for regularization.
class EnhancedMLP(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes, dropout_rate=0.4):
        super(EnhancedMLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_layers[0]))
        layers.append(nn.BatchNorm1d(hidden_layers[0]))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(dropout_rate))
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            layers.append(nn.BatchNorm1d(hidden_layers[i+1]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(hidden_layers[-1], num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

#LSTM model that uses Bidirectional LSTM with Dropout and a fully connected layer.
class EnhancedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2, dropout_rate=0.4):
        super(EnhancedLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout_rate if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

#CNN model that uses Conv1D layers with Batch Normalization, ReLU activation, MaxPooling, and Dropout.
class EnhancedCNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(EnhancedCNN, self).__init__()
        def conv_output_size(size, kernel_size=3, stride=1, padding=1, pool_kernel=2, pool_stride=2):
            size = (size - kernel_size + 2 * padding) // stride + 1
            size = (size - pool_kernel) // pool_stride + 1
            return size

        conv1_out_len = conv_output_size(input_size)
        conv2_out_len = conv_output_size(conv1_out_len)
        #Note that if conv2_out_len becomes 0, it means the input is too small for the network depth (should not be an issue for IDS 2018 dataset!)
        if conv2_out_len <= 0:
            raise ValueError(f"Input size {input_size} is too small for the CNN architecture. Please use a larger feature set or a different model.")
        flattened_size = 128 * conv2_out_len

        self.network = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2), nn.Dropout(0.25),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2), nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(flattened_size, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Dropout(0.5), nn.Linear(512, num_classes)
        )
    def forward(self, x):
        x = x.unsqueeze(1)
        return self.network(x)

#Tab transformer model that uses a Transformer Encoder with a classification token and an MLP head, 
#it is designed to handle tabular data with multiple features. Theoretically it is not the best model for this dataset but
#it has a chance to provide interesting results. Note that this model will take longer to train and more hardware resources (more memory and compute power).
class TabTransformer(nn.Module):
    def __init__(self, *, num_features, num_classes, dim, n_heads, n_layers, dropout):
        super().__init__()
        self.feature_embeddings = nn.ModuleList([nn.Linear(1, dim) for _ in range(num_features)])
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim * 4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, x):
        embeddings = [self.feature_embeddings[i](x[:, i].unsqueeze(1)) for i in range(x.shape[1])]
        x_embedded = torch.stack(embeddings, dim=1)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x_with_cls = torch.cat((cls_tokens, x_embedded), dim=1)
        transformer_output = self.transformer_encoder(x_with_cls)
        cls_token_output = transformer_output[:, 0]
        return self.mlp_head(cls_token_output)
    


### IDS 2018 Intrusion CSVs (CSE-CIC-IDS2018) Specific Label Mappings ###
#This mapping is used to give the models more complex and specific labels to identify attack types.
#But it comes with the tradeoff of increase false positives, for example the models may identify a benign class as "infiltration" or "web-attack.""
LABEL_MAPPINGS = {
    'Benign': 'Benign',
    'BENIGN': 'Benign',

    'Botnet': 'Botnet',
    'Bot': 'Botnet',

    'Brute Force-FTP': 'Brute Force-FTP',
    'FTP-BruteForce': 'Brute Force-FTP',
    'FTP-Patator': 'Brute Force-FTP',
    'Brute Force-SSH': 'Brute Force-SSH',
    'SSH-Bruteforce': 'Brute Force-SSH',
    'SSH-Patator': 'Brute Force-SSH',

    'Brute Force-Web': 'Web-Attack',
    'Brute Force -Web': 'Web-Attack',
    'Brute Force-XSS': 'Web-Attack',
    'Brute Force -XSS': 'Web-Attack',
    'SQL Injection': 'Web-Attack',

    'DDoS-HOIC': 'DDoS-HOIC',
    'DDoS attacks-HOIC': 'DDoS-HOIC',
    'DDOS attack-HOIC': 'DDoS-HOIC',
    'DDoS-LOIC-HTTP': 'DDoS-LOIC',
    'DDoS attacks-LOIC-HTTP': 'DDoS-LOIC',
    'DDoS-LOIC-UDP': 'DDoS-LOIC',
    'DDOS attack-LOIC-UDP': 'DDoS-LOIC',

    'DoS-GoldenEye': 'DoS-GoldenEye',
    'DoS attacks-GoldenEye': 'DoS-GoldenEye',
    'DoS-Hulk': 'DoS-Hulk',
    'DoS attacks-Hulk': 'DoS-Hulk',
    'DoS-SlowHTTPTest': 'DoS-SlowHTTPTest',
    'DoS attacks-SlowHTTPTest': 'DoS-SlowHTTPTest',
    'DoS-Slowloris': 'DoS-Slowloris',
    'DoS attacks-Slowloris': 'DoS-Slowloris',

    'Infiltration': 'Infiltration',
    'Infilteration': 'Infiltration',
}

### Simplified Mapping, to increase accuray and reduce false positives at the cost of less specificity. ###
#Note the modles using this simplifeied mapping also removs the 'Infiltration' entry entirely.
LABEL_MAPPINGS_SIMPLIFIED_ = {
    'BENIGN': 'Benign',
    'Benign': 'Benign',
    'Bot': 'Botnet',
    'Botnet': 'Botnet',
    'Brute Force -Web': 'Brute Force',
    'Brute Force -XSS': 'Brute Force',
    'Brute Force-Web': 'Brute Force',
    'Brute Force-XSS': 'Brute Force',
    'SQL Injection': 'Brute Force', 
    'Infiltration': 'Infiltration',
    'Infilteration': 'Infiltration',
    'DoS attacks-GoldenEye': 'DoS',
    'DoS-GoldenEye': 'DoS',
    'DoS attacks-Slowloris': 'DoS',
    'DoS-Slowloris': 'DoS',
    'DoS attacks-SlowHTTPTest': 'DoS',
    'DoS-SlowHTTPTest': 'DoS',
    'DoS attacks-Hulk': 'DoS',
    'DoS-Hulk': 'DoS',
    'DDoS attacks-LOIC-HTTP': 'DDoS',
    'DDoS-LOIC-HTTP': 'DDoS',
    'DDOS attack-LOIC-UDP': 'DDoS',
    'DDoS-LOIC-UDP': 'DDoS',
    'DDoS attacks-HOIC': 'DDoS',
    'DDOS attack-HOIC': 'DDoS',
    'DDoS-HOIC': 'DDoS',
    'FTP-BruteForce': 'Brute Force',
    'Brute Force-FTP': 'Brute Force',
    'FTP-Patator': 'Brute Force',
    'SSH-Bruteforce': 'Brute Force',
    'Brute Force-SSH': 'Brute Force',
    'SSH-Patator': 'Brute Force',
}
#Dataset Loading class, used to load the processed data into a PyTorch Dataset and it has valdations checks
#to make sure all of the data is present and correct before training is started.
class IntrusionDataset(Dataset):
    def __init__(self, processed_data_dir):
        print(f"\nOptimized data is loading from... {processed_data_dir}...")

        required_files = ['features.pt', 'labels.pt', 'scaler.gz', 'label_encoder.gz', 'metadata.gz']
        for file in required_files:
            path = os.path.join(processed_data_dir, file)
            if not os.path.exists(path):
                raise FileNotFoundError(f"The following file is missing: {file}! ")

        self.features = torch.load(os.path.join(processed_data_dir, 'features.pt'), weights_only=False)
        self.labels = torch.load(os.path.join(processed_data_dir, 'labels.pt'), weights_only=False)

        self.scaler = joblib.load(os.path.join(processed_data_dir, 'scaler.gz'))
        self.label_encoder = joblib.load(os.path.join(processed_data_dir, 'label_encoder.gz'))
        self.metadata = joblib.load(os.path.join(processed_data_dir, 'metadata.gz'))

        print(f"Data has been loaded correctly: {len(self.labels):,} samples, {self.features.shape[1]} features.")
        self.print_class_distribution()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def get_dims(self):
        return self.features.shape[1], len(self.label_encoder.classes_)

    def get_class_names(self):
        return list(self.label_encoder.classes_)

    def print_class_distribution(self):
        print("\nThe Class Distribution in the Dataset (In Memory):")
        unique_labels, counts = np.unique(self.labels.numpy(), return_counts=True)
        for label_idx, count in zip(unique_labels, counts):
            class_name = self.label_encoder.inverse_transform([label_idx])[0]
            print(f"   - {class_name}: {count:,} samples ({count / len(self.labels) * 100:.2f}%%)")
        print("_" * 65)


#Pre processing function, which is responsible for reading the CSV files, and processing them into a format that can be used for training.
def preprocess_and_save_data(source_csv_paths, target_dir, mapping):
    os.makedirs(target_dir, exist_ok=True)
    print(f"Preprocessing: {len(source_csv_paths)} file(s)...")

    print("Column Structure is being Analyzed, this may take a moment...")
    try:
        reference_df = pd.read_csv(source_csv_paths[0], nrows=5)
        reference_df.columns = [col.strip() for col in reference_df.columns]
        label_col = 'Label'
        if label_col not in reference_df.columns:
            print(f"ERROR! The '{label_col}' column was not found in the first CSV!")
            return

        feature_cols = [col for col in reference_df.columns if col not in ['Timestamp', label_col]]
        print(f"Found {len(feature_cols)} feature columns and '{label_col}' as the label.")
    except Exception as e:
        print(f"Unable to read first CSV file to determine structure: {e}")
        return

    all_features = []
    all_labels = []

    #This for loop goes through each CSV file and processes it in chunks to avoid memory issues with large files.
    for csv_path in tqdm(source_csv_paths, desc="Processing CSV files.."):
        try:
            chunk_iter = pd.read_csv(csv_path, chunksize=CHUNK_SIZE, low_memory=False, iterator=True)

            for i, chunk in enumerate(chunk_iter):
                try:
                    chunk.columns = [col.strip() for col in chunk.columns]
                    chunk = chunk.reindex(columns=feature_cols + [label_col], fill_value=0)
                    X_chunk = chunk[feature_cols].copy()
                    y_chunk = chunk[label_col].copy()

                    for col in X_chunk.columns:
                        X_chunk[col] = pd.to_numeric(X_chunk[col], errors='coerce')

                    X_chunk.dropna(how='all', inplace=True)
                    X_chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
                    X_chunk.fillna(0, inplace=True)

                    y_chunk = y_chunk.loc[X_chunk.index].astype(str).str.strip()
                    y_chunk = y_chunk.map(mapping).fillna(y_chunk) 

                    all_features.append(X_chunk.astype(np.float32))
                    all_labels.extend(y_chunk.tolist())

                except Exception as e:
                    print(f"Warning! Corrupted chunk detected (Skipping) {i+1} in {os.path.basename(csv_path)}. Reason: {e}")
                    continue

        except Exception as e:
            print(f"Error, could not process file: {os.path.basename(csv_path)}. Reason: {e}")
            continue

    if not all_features:
        print("Catostrophic Error! No valid data has been processed from the CSV files!")
        return

    print("\nChunks process successfully, now combining all features and labels...")
    X_combined = pd.concat(all_features, ignore_index=True)

    print("Encoding labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(all_labels)

    print("Features being scaled...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)

    print("Converting to tensors...")
    features_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    labels_tensor = torch.tensor(y_encoded, dtype=torch.long)

    unique, counts = np.unique(y_encoded, return_counts=True)
    class_names = label_encoder.inverse_transform(unique)
    print("\nData distribution in the processed dataset:")
    for name, cnt in zip(class_names, counts):
        print(f"   • {name:<20} : {cnt:>8,}  ({cnt/len(all_labels)*100:.2f} %%)")
    print("_" * 65)


    print("Processed data is being saved to disk...")
    torch.save(features_tensor, os.path.join(target_dir, 'features.pt'))
    torch.save(labels_tensor, os.path.join(target_dir, 'labels.pt'))
    joblib.dump(label_encoder, os.path.join(target_dir, 'label_encoder.gz'))
    joblib.dump(scaler, os.path.join(target_dir, 'scaler.gz'))

    metadata = {
        'num_samples': len(labels_tensor),
        'num_features': features_tensor.shape[1],
        'num_classes': len(label_encoder.classes_),
        'classes': list(label_encoder.classes_),
        'feature_names': list(X_combined.columns)
    }
    joblib.dump(metadata, os.path.join(target_dir, 'metadata.gz'))

    print("\n" + "#"*55)
    print("Preprocessing Completed! Data is now ready to be used for model training!")
    print(f"Processed Data has been saved to {target_dir}")
    print(f"The final dataset contains: {metadata['num_samples']:,} samples, {metadata['num_features']} features")
    print(f"Classes ({metadata['num_classes']}): {', '.join(metadata['classes'])}")
    print("#"*55)
