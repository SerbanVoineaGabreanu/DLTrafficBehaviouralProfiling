#Authors: Serban Voinea Gabreanu, Gur Rehmat Singh Chahal, Algoma University, COSC5906002 Advanced Topics of Computer Networks (25SP), Final Project.
#DL_Script, this script is responsible for training the deep learning models (and random forest). It includes code to automatically preprocess the 
#IDS 2018 Intrusion dataset (available here: https://www.kaggle.com/datasets/solarmainframe/ids-intrusion-csv/data?select=02-21-2018.csv).
#
#The script includes standard training, where the entire dataset is used for the models to train with. Then there is the option for "Balanced" which
#reduces the amount of benign samples from ~83% to ~20%.  Then there is simlpified, which removes the 'Web-Attack' and 'Infiltration' classes entirely from training.
#Finally there is the Super simplified version, which completely simplifies the label mapings, and also removes the 'Infiltration' class entirely, this requires 
#the preprocessing to be run with the Super Simlpified option (option 2, instead of otpion 1).
#
#Option 4-7 allows for training an individual model, with options for MLP, LSTM, CNN, Tab Transformer, and Random Forest.
#Option 4 uses the entire dataset, option 5 uses the same dataset but with benign samples reduced to ~20%, option 6 is the same as option 5 except it 
#removes the 'Web-Attack' and 'Infiltration' classes entirely from training, Finaly option 7 completly changes the label mapings to a simplified version, 
#and also removes the 'Infiltration' class entirely.
#
#This script also allows for queuing up all of the modles to train using options 8,9, or 10. 
#
#Option 11 allows for a paused training session to be continued.
#
#There is also the options 12-14 to create ensemble models, which combines balnaced MLP with standard Random Forest, or Simlpified MLP with standard Random Forest,
#and finally super simplified random forest with the user's selection of the model including MLP, LSTM, CNN, or Tab Transformer.
#
#Option 15 is an experimental model that trains any model of choosing (such as MLP), but creates different variants that specialize on each class, and then
#they are all combined in a final output. (This version has not been tested yet).
#
#There is also option 16 to allow the user to validate all of the deployed models (the ones with finished training files, in the DeployedModel folder), to get
#a validation report. Option 17 is similar but generates validation graphs for each model, which can be used to compare the models visually, and also generates
#an overal comparison graph(s) for all of the models.
#Option 18 allows the best model from checkpoints to be deployed to the DeployModel folder, which can then be used for inference.
#
#Option 19 lists all of the availabe training sessions from the Checkpoints folder, and allows the user to select one to load to continue training
#
#Option 20 closes the script.
#
#NOTE: Preprocessing MUST be run before models can be trained!

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from datetime import datetime
import glob
from tqdm import tqdm
import joblib
import shutil
import warnings
import matplotlib.pyplot as plt # Added for graphing
from models import (EnhancedMLP, EnhancedLSTM, EnhancedCNN, TabTransformer, 
                    IntrusionDataset, preprocess_and_save_data, 
                    LABEL_MAPPINGS, LABEL_MAPPINGS_SIMPLIFIED_)
warnings.filterwarnings('ignore')


#File paths (using relative path, if its not working an absolute path can be set)
#Each directory is checked for relative location, if not found it will use an absolute path.
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

#Absolute path for fallback.
ABSOLUTE_BASE_DIR = '/Users/serbanvg/Documents/School 2025/2025 Spring Algoma/COSC5906 Networking/Final Project'

DATASET_DIR = os.path.join(BASE_DIR, 'Dataset')
if not os.path.isdir(DATASET_DIR):
    DATASET_DIR = os.path.join(ABSOLUTE_BASE_DIR, 'Dataset')

PROCESSED_DIR = os.path.join(BASE_DIR, 'ProcessedDataset')
if not os.path.isdir(PROCESSED_DIR):
    PROCESSED_DIR = os.path.join(ABSOLUTE_BASE_DIR, 'ProcessedDataset')

PROCESSED_DIR_SIMPLIFIED = os.path.join(BASE_DIR, 'ProcessedDatasetSimplified')
if not os.path.isdir(PROCESSED_DIR_SIMPLIFIED):
    PROCESSED_DIR_SIMPLIFIED = os.path.join(ABSOLUTE_BASE_DIR, 'ProcessedDatasetSimplified')

CHECKPOINT_DIR = os.path.join(BASE_DIR, 'Checkpoints')
if not os.path.isdir(CHECKPOINT_DIR):
    CHECKPOINT_DIR = os.path.join(ABSOLUTE_BASE_DIR, 'Checkpoints')

DEPLOY_DIR = os.path.join(BASE_DIR, 'DeployModel')
if not os.path.isdir(DEPLOY_DIR):
    DEPLOY_DIR = os.path.join(ABSOLUTE_BASE_DIR, 'DeployModel')

GRAPHS_DIR = os.path.join(BASE_DIR, 'ValidationGraphs')
if not os.path.isdir(GRAPHS_DIR):
    GRAPHS_DIR = os.path.join(ABSOLUTE_BASE_DIR, 'ValidationGraphs')

PER_MODEL_GRAPHS_DIR = os.path.join(BASE_DIR, 'PerModelGraphs')
if not os.path.isdir(PER_MODEL_GRAPHS_DIR):
    PER_MODEL_GRAPHS_DIR = os.path.join(ABSOLUTE_BASE_DIR, 'PerModelGraphs')


#Model and training hyperparameters.
#Note these might need to be adjusted based on the availabe hardware (e.g. larger batch sizes need more memory), 
#larger epochs will take longer to train, and larger hidden layers will also take more memory.
LEARNING_RATE = 0.0001
BATCH_SIZE = 4096
EPOCHS = 100
HIDDEN_LAYERS = [256, 128, 64]
VALIDATION_SPLIT = 0.2
RF_ESTIMATORS = 100

#Tab Transformer hyperparameters.
TRANSFORMER_DIM = 32
TRANSFORMER_HEADS = 8
TRANSFORMER_LAYERS = 6
TRANSFORMER_DROPOUT = 0.1

#Training Functions

#Saves the model's current training state, allowing for the session to be resumed at any point later.
def save_checkpoint(state, session_name, is_best, is_interrupt=False):
    """Saves model checkpoint."""
    session_dir = os.path.join(CHECKPOINT_DIR, session_name)
    os.makedirs(session_dir, exist_ok=True)
    filename = 'interrupt.pth.tar' if is_interrupt else 'checkpoint.pth.tar'
    filepath = os.path.join(session_dir, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(session_dir, 'model_best.pth.tar'))
        print("Best model has been saved!")

#Evaluates the current model on the training set and returns the metrics and raw predictions 
#for a better set of data to analyze.
def evaluate_on_validation_set(model, val_loader, device, class_names):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Evaluating on Validation Set", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\n" + "#" * 22 + " Validation Report " + "#" * 22)
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0, digits=4))
    print("#"*59)

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average='weighted', zero_division=0),
        "recall": recall_score(all_labels, all_preds, average='weighted', zero_division=0),
        "f1": f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    }
    return metrics, all_labels, all_preds

#Creates a balanced dataset by reducing the amount of benign classes to ~20% (target_benign_percentage var) instead of ~83% of the data.
def create_balanced_loaders(dataset, target_benign_percentage=0.20):
    print(f"\nBalancing Datasett: Targeting {target_benign_percentage*100:.0f}% for the 'Benign' class in the training set.")

    all_labels = dataset.labels.numpy()
    label_encoder = dataset.label_encoder
    benign_class_name = 'Benign'
    try:
        benign_label_index = np.where(label_encoder.classes_ == benign_class_name)[0][0]
    except IndexError:
        print(f"Error! Class '{benign_class_name}' has NOT been found by the label encoder.")
        return None, None

    all_indices = np.arange(len(all_labels))
    benign_indices = all_indices[all_labels == benign_label_index]
    attack_indices = all_indices[all_labels != benign_label_index]

    n_attack = len(attack_indices)
    n_benign_new = int((target_benign_percentage / (1 - target_benign_percentage)) * n_attack)
    n_benign_new = min(n_benign_new, len(benign_indices))

    print(f"Found {n_attack:,} attack samples.")
    print(f"Found {len(benign_indices):,} original benign samples.")
    print(f"Randomly selecting {n_benign_new:,} benign samples for the training set.")

    #Seed can be changed but the seed used was 42 (meaning of life).
    np.random.seed(42)
    sampled_benign_indices = np.random.choice(benign_indices, size=n_benign_new, replace=False)

    balanced_train_indices = np.concatenate([attack_indices, sampled_benign_indices])
    np.random.shuffle(balanced_train_indices)

    _, val_indices = train_test_split(all_indices, test_size=VALIDATION_SPLIT, stratify=all_labels, random_state=42)

    train_subset = Subset(dataset, balanced_train_indices)
    val_subset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    final_benign_pct = n_benign_new / len(balanced_train_indices) * 100
    print(f"New balanced training set size: {len(train_subset):,} samples ({final_benign_pct:.2f}% benign).")
    print(f"Validation set size: {len(val_subset):,} samples (from original distribution).")
    print("Balanced data loaders are now ready.")

    return train_loader, val_loader

#Creates simplified training loader, this includes removing web attack and infiltration, and also reducing benign to 20% (or whatever target_benign_percentage is set to)
def create_simplified_loaders(dataset, target_benign_percentage=0.20):
    print(f"\nCreating a simplified dataset: Removing Web-Attack/Infiltration and targeting {target_benign_percentage*100:.0f}% 'Benign'.")

    all_labels = dataset.labels.numpy()
    label_encoder = dataset.label_encoder
    class_names = list(label_encoder.classes_)

    # Find indices of classes to remove from the training pool
    classes_to_remove = ['Web-Attack', 'Infiltration']
    indices_to_remove_from_training = []
    for class_name in classes_to_remove:
        if class_name in class_names:
            class_index = class_names.index(class_name)
            indices_to_remove_from_training.extend(np.where(all_labels == class_index)[0])
            print(f"Identified class '{class_name}' for removal from training set.")
        else:
            print(f"Warning: Class '{class_name}' not found in dataset, cannot remove.")

    all_indices = np.arange(len(all_labels))
    # Get the indices that are NOT in the removal list to form our pool of training data
    training_candidate_indices = np.setdiff1d(all_indices, np.unique(indices_to_remove_from_training))
    print(f"Original samples: {len(all_indices):,}. Candidates for training set: {len(training_candidate_indices):,} samples.")

    # Now, perform balancing on this simplified set of indices
    simplified_labels = all_labels[training_candidate_indices]

    benign_class_name = 'Benign'
    try:
        benign_label_index = np.where(label_encoder.classes_ == benign_class_name)[0][0]
    except IndexError:
        print(f"ERROR! Class '{benign_class_name}' has NOT been found by the label encoder.")
        return None, None

    #Seperates the benign and attack indices FROM THE CANDIDATE POOL.
    benign_indices_simplified = training_candidate_indices[simplified_labels == benign_label_index]
    attack_indices_simplified = training_candidate_indices[simplified_labels != benign_label_index]

    n_attack = len(attack_indices_simplified)
    n_benign_new = int((target_benign_percentage / (1 - target_benign_percentage)) * n_attack)
    n_benign_new = min(n_benign_new, len(benign_indices_simplified))

    print(f"Found {n_attack:,} attack samples in the simplified set.")
    print(f"Found {len(benign_indices_simplified):,} benign samples in the simplified set.")
    print(f"Randomly selecting {n_benign_new:,} benign samples for the training set.")

    np.random.seed(42)
    sampled_benign_indices = np.random.choice(benign_indices_simplified, size=n_benign_new, replace=False)

    #The final training indices are the simplified attacks + the down-sampled benigns
    simplified_train_indices = np.concatenate([attack_indices_simplified, sampled_benign_indices])
    np.random.shuffle(simplified_train_indices)

    #The validation set is still a split from the ORIGINAL full dataset for a fair comparison
    _, val_indices = train_test_split(all_indices, test_size=VALIDATION_SPLIT, stratify=all_labels, random_state=42)

    train_subset = Subset(dataset, simplified_train_indices)
    val_subset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    if len(simplified_train_indices) > 0:
        final_benign_pct = n_benign_new / len(simplified_train_indices) * 100
        print(f"New simplified training set size: {len(train_subset):,} samples ({final_benign_pct:.2f}% benign).")
    else:
        print("Warning: New simplified training set is empty.")

    print(f"Validation set size: {len(val_subset):,} samples (from original distribution).")
    print("The simplified data loaders are ready!")

    return train_loader, val_loader

#Creates a SUPER simlpified loader, using the simplified label mappings and the removal of infiltrator class from the training set.
def create_super_simplified_loaders(dataset, target_benign_percentage=0.20):
    print(f"\nCreating Super Simplified dataset: Removing 'Infiltration' and targeting {target_benign_percentage*100:.0f}% 'Benign'.")

    all_labels = dataset.labels.numpy()
    label_encoder = dataset.label_encoder
    class_names = list(label_encoder.classes_)

    #Finds the index of 'Infiltration' in order to remove it from the training pool
    indices_to_remove_from_training = []
    if 'Infiltration' in class_names:
        infiltration_index = class_names.index('Infiltration')
        indices_to_remove_from_training = np.where(all_labels == infiltration_index)[0]
        print(f"Identified and marked '{'Infiltration'}' for removal from the training set ({len(indices_to_remove_from_training):,} samples).")
    else:
        print("Note: 'Infiltration' class not found in dataset, no removal needed.")

    all_indices = np.arange(len(all_labels))
    #Finds the indices that are NOT 'Infiltration' to form the pool of the training data
    training_candidate_indices = np.setdiff1d(all_indices, np.unique(indices_to_remove_from_training))
    print(f"Original samples: {len(all_indices):,}. Candidates for training set: {len(training_candidate_indices):,} samples.")

    filtered_labels = all_labels[training_candidate_indices]

    benign_class_name = 'Benign'
    try:
        benign_label_index = np.where(label_encoder.classes_ == benign_class_name)[0][0]
    except IndexError:
        print(f"ERROR! Class '{benign_class_name}' has NOT been found by the label encoder.")
        return None, None

    #Separates the benign and attack indices from the candidate pool.
    benign_indices_filtered = training_candidate_indices[filtered_labels == benign_label_index]
    attack_indices_filtered = training_candidate_indices[filtered_labels != benign_label_index]

    n_attack = len(attack_indices_filtered)
    n_benign_new = int((target_benign_percentage / (1 - target_benign_percentage)) * n_attack)
    n_benign_new = min(n_benign_new, len(benign_indices_filtered))

    print(f"Found {n_attack:,} attack samples in the filtered set.")
    print(f"Found {len(benign_indices_filtered):,} benign samples in the filtered set.")
    print(f"Balancing: Randomly selecting {n_benign_new:,} benign samples for the training set.")

    np.random.seed(42)
    sampled_benign_indices = np.random.choice(benign_indices_filtered, size=n_benign_new, replace=False)

    #Final training indices are the filtered attacks and the down-sampled benigns
    final_train_indices = np.concatenate([attack_indices_filtered, sampled_benign_indices])
    np.random.shuffle(final_train_indices)

    #The validation set is still a split from the original full dataset to make a good comparison.
    _, val_indices = train_test_split(all_indices, test_size=VALIDATION_SPLIT, stratify=all_labels, random_state=42)

    train_subset = Subset(dataset, final_train_indices)
    val_subset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    if len(final_train_indices) > 0:
        final_benign_pct = n_benign_new / len(final_train_indices) * 100
        print(f"New super simplified training set size: {len(train_subset):,} samples ({final_benign_pct:.2f}% benign).")
    else:
        print("ERROR: The new super simplified training set is empty! ")

    print(f"Validation set size: {len(val_subset):,} samples (from original relabeled distribution).")
    print("Super Simplified data loaders are now ready to be used.")

    return train_loader, val_loader

#Training Code. Adds progress bars, validaiton, and strong checkpoint system to allow for easy usage.
#This version works with both of the preprocessed datasets. (Normal and simplified).
def train_pytorch_model(model, train_loader, val_loader, device, session_name, class_names, start_epoch=0):
    print("\nFinding the class weights for the training dataset...")

    num_classes = len(class_names)
    
    #Handles the different dataset types for weight calculation
    dataset_obj = train_loader.dataset.dataset
    indices = train_loader.dataset.indices
    
    if isinstance(dataset_obj, IntrusionDataset):
        all_labels_in_train_set = dataset_obj.labels[indices].cpu()
    elif isinstance(dataset_obj, TensorDataset):
        #Used for specialist models.
        all_labels_in_train_set = dataset_obj.tensors[1][indices].cpu()
    else:
        #Fallback: Goes through the load if the structure is unknown, but if this happens there's an issue with the 
        #dataset that should be fixed.
        print(f"ERROR: Unable to determin dataset type ({type(dataset_obj)}). Finding weights by iterating loader.")
        all_labels_in_train_set = torch.cat([labels for _, labels in train_loader])

    class_counts = torch.bincount(all_labels_in_train_set, minlength=num_classes)
    class_weights = 1.0 / class_counts.float()
    class_weights[class_counts == 0] = 0 

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * EPOCHS)

    model.to(device)

    print("\n" + "#"*22 + " Training Statistics " + "#"*22)
    print(f"Model: {model.__class__.__name__}")
    print(f"Session: {session_name}")
    print(f"Device: {device.type.upper()}")
    print(f"Loss Function: CrossEntropyLoss (Weighted)")
    print(f"Optimizer: AdamW")
    print(f"Scheduler: CosineAnnealingLR")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("#"*59 + "\n")

    best_f1 = 0.0
    patience_counter = 0
    #Depending on how much training time is available, this can be set high (means models will likely train for significantly longer) 
    #or low if training time is limited and results from diminishing gains are not needed. For this project it was set to 15.
    early_stopping_patience = 15

    try:
        for epoch in range(start_epoch, EPOCHS):
            model.train()
            running_loss = 0.0
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Training]', colour='cyan')

            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                running_loss += loss.item()
                pbar.set_postfix(loss=f'{loss.item():.4f}', lr=f'{optimizer.param_groups[0]["lr"]:.1e}')

            avg_train_loss = running_loss / len(train_loader)

            metrics, all_labels, all_preds = evaluate_on_validation_set(model, val_loader, device, class_names)

            val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
            print(f"Epoch {epoch+1} Summary | Train Loss: {avg_train_loss:.4f} | Macro F1: {val_f1:.4f} | Best Macro F1: {best_f1:.4f}")

            is_best = val_f1 > best_f1
            if is_best:
                best_f1 = val_f1
                patience_counter = 0
            else:
                patience_counter += 1

            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'model_type': model.__class__.__name__,
            }, session_name, is_best)

            if patience_counter >= early_stopping_patience:
                print(f"Early stopping has been triggered after {early_stopping_patience} epochs with no improvement.")
                break

        print(f"\nTraining completed!The Best validation Macro F1-score is: {best_f1:.4f}")
        deploy_model_from_session(session_name)

    except KeyboardInterrupt:
        print("\n\nTraining has been interrupted! Currently saving final state...")
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_f1': best_f1,
            'model_type': model.__class__.__name__,
        }, session_name, is_best=False, is_interrupt=True)
        print("Progress has been saved succesfully, exiting...")
        sys.exit(0)


#Trains the random forest machine learning model.
def train_random_forest(train_loader, val_loader, session_name, class_names):
    print("\n" + "#"*17 + " Random Forest Training " + "="*17)
    print("Data being prepared...")

    X_train = train_loader.dataset.dataset.features[train_loader.dataset.indices].numpy()
    y_train = train_loader.dataset.dataset.labels[train_loader.dataset.indices].numpy()
    X_val = val_loader.dataset.dataset.features[val_loader.dataset.indices].numpy()
    y_val = val_loader.dataset.dataset.labels[val_loader.dataset.indices].numpy()

    print(f"Fitting RandomForestClassifier with the {RF_ESTIMATORS} estimators...")
    model = RandomForestClassifier(n_estimators=RF_ESTIMATORS, random_state=42, n_jobs=-1, verbose=1, max_depth=20, min_samples_leaf=5, class_weight="balanced")
    model.fit(X_train, y_train)

    print("Model fitting has been completed! Evaluating...")
    y_pred = model.predict(X_val)

    print("\n" + "#" * 22 + " Random Forest Results " + "#" * 22)
    print(classification_report(y_val, y_pred, target_names=class_names, zero_division=0, digits=4))
    print("#"*59)

    session_dir = os.path.join(CHECKPOINT_DIR, session_name)
    os.makedirs(session_dir, exist_ok=True)
    checkpoint_path = os.path.join(session_dir, 'RandomForest_best.joblib')
    joblib.dump(model, checkpoint_path)
    print(f"Model has been saved to: {checkpoint_path}")

    deploy_model_from_session(session_name)

#Ensemble model is created using checkpoints from balanced MLP and standard random forest.
def create_ensemble_from_checkpoints(device, mlp_checkpoint_path, rf_model_path):
    print("\n" + "#"*17 + " Ensemble (MLP + RF) Stacking Creation " + "#"*17)

    print("Loading data and preparing validation set...")
    dataset = IntrusionDataset(PROCESSED_DIR)
    _, val_loader = create_balanced_loaders(dataset, target_benign_percentage=0.20)
    if val_loader is None:
        print("Ensemble creation has been stopped due to data loading error!.")
        return
    input_size, num_classes = dataset.get_dims()
    class_names = dataset.get_class_names()

    print(f"\nLoading base model EnhancedMLP from '{os.path.basename(mlp_checkpoint_path)}'...")
    try:
        mlp_model = instantiate_model('EnhancedMLP', input_size, num_classes)
        checkpoint = torch.load(mlp_checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        mlp_model.load_state_dict(state_dict)
        mlp_model.to(device)
        mlp_model.eval()
        print("MLP model has been loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load the MLP model: {e}"); return

    print(f"\nLoading the base model Random Forest from '{os.path.basename(rf_model_path)}'...")
    try:
        rf_model = joblib.load(rf_model_path)
        print("Random Forest model has been loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load RandomForest model: {e}"); return

    print("\nGenerating features from the validation set...")
    X_val = val_loader.dataset.dataset.features[val_loader.dataset.indices].numpy()
    y_val = val_loader.dataset.dataset.labels[val_loader.dataset.indices].numpy()

    all_mlp_preds = []
    with torch.no_grad():
        for inputs, _ in tqdm(val_loader, desc="Getting MLP predictions", leave=False, colour='yellow'):
            inputs = inputs.to(device)
            outputs = mlp_model(inputs)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            all_mlp_preds.append(probabilities)
    mlp_meta_features = np.concatenate(all_mlp_preds)
    rf_meta_features = rf_model.predict_proba(X_val)
    X_meta_train = np.concatenate([mlp_meta_features, rf_meta_features], axis=1)
    print("Features generated succesfully!")

    print("\nTraining learner and deploying the ensemble model...")
    meta_learner = LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1)
    meta_learner.fit(X_meta_train, y_val)
    print("Meta learner has been trained succesfully!")

    session_name = f"EnsembleModelMLPForest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    deploy_filename = f"{session_name}_deployable.joblib"
    deploy_path = os.path.join(DEPLOY_DIR, deploy_filename)

    ensemble_package = {
        'mlp_model_state_dict': mlp_model.state_dict(),
        'rf_model': rf_model,
        'meta_learner': meta_learner,
        'scaler': dataset.scaler,
        'label_encoder': dataset.label_encoder,
        'metadata': dataset.metadata,
        'model_type': 'EnsembleMLPForest'
    }

    joblib.dump(ensemble_package, deploy_path)
    print(f"The ensemble model has been deployed successfully to: {deploy_path}")

    print("\n" + "#"*22 + " Ensemble Validation Report " + "="*22)
    final_predictions = meta_learner.predict(X_meta_train)
    print(classification_report(y_val, final_predictions, target_names=class_names, zero_division=0, digits=4))
    print("#"*71)

#Similar to above ensemble function except uses the simplified dataset
def create_simplified_ensemble_from_checkpoints(device):
    print("\n" + "="*10 + " Simplified Ensemble (MLP + RF) Stacking Creation " + "="*10)

    print("Data loading and preparing validation set...")
    dataset = IntrusionDataset(PROCESSED_DIR)
    _, val_loader = create_simplified_loaders(dataset, target_benign_percentage=0.20)
    if val_loader is None:
        print("Data loading error! Operation stopped.")
        return
    input_size, num_classes = dataset.get_dims()
    class_names = dataset.get_class_names()

    print(f"\nLoading MLP model from a 'SIMPLIFIED' checkpoint...")
    mlp_sessions = sorted([d for d in os.listdir(CHECKPOINT_DIR) if 'EnhancedMLP_SIMPLIFIED' in d and os.path.isdir(os.path.join(CHECKPOINT_DIR, d))])
    if not mlp_sessions:
        print("\n'EnhancedMLP_SIMPLIFIED' sessions not found. Please train one using option 6."); return
    mlp_checkpoint_path = os.path.join(CHECKPOINT_DIR, mlp_sessions[-1], 'model_best.pth.tar') 
    print(f"Using MLP model from session: {mlp_sessions[-1]}")
    if not os.path.exists(mlp_checkpoint_path):
        print(f"The best model has NOT been found in session '{mlp_sessions[-1]}'."); return

    try:
        mlp_model = instantiate_model('EnhancedMLP', input_size, num_classes)
        checkpoint = torch.load(mlp_checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        mlp_model.load_state_dict(state_dict)
        mlp_model.to(device)
        mlp_model.eval()
        print("MLP model has been loaded successfully.")
    except Exception as e:
        print(f"Failed to load MLP model: {e}"); return

    print(f"\nLoading Random Forest model from a 'SIMPLIFIED' checkpoint...")

    #Finds the latest SIMPLIFIED RF model
    rf_sessions = sorted([d for d in os.listdir(CHECKPOINT_DIR) if 'RandomForest_SIMPLIFIED' in d and os.path.isdir(os.path.join(CHECKPOINT_DIR, d))])
    if not rf_sessions:
        print("\n'RandomForest_SIMPLIFIED' has NO sessions found. Please train one using option 6."); return
    rf_model_path = os.path.join(CHECKPOINT_DIR, rf_sessions[-1], 'RandomForest_best.joblib')
    print(f"   - Using RF model from session: {rf_sessions[-1]}")
    if not os.path.exists(rf_model_path):
        print(f"Best model has NOT been found in the session '{rf_sessions[-1]}'."); return

    try:
        rf_model = joblib.load(rf_model_path)
        print("Random Forest model has been loaded successfully.")
    except Exception as e:
        print(f"Unable to load the RandomForest model: {e}"); return

    print("\nGenerating features from the validation set...")
    X_val = val_loader.dataset.dataset.features[val_loader.dataset.indices].numpy()
    y_val = val_loader.dataset.dataset.labels[val_loader.dataset.indices].numpy()

    all_mlp_preds = []
    with torch.no_grad():
        for inputs, _ in tqdm(val_loader, desc="Getting MLP predictions", leave=False, colour='yellow'):
            inputs = inputs.to(device)
            outputs = mlp_model(inputs)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            all_mlp_preds.append(probabilities)
    mlp_meta_features = np.concatenate(all_mlp_preds)
    rf_meta_features = rf_model.predict_proba(X_val)
    X_meta_train = np.concatenate([mlp_meta_features, rf_meta_features], axis=1)
    print("Features generated.")

    print("\nTraining learner and deploying the simplified ensemble model...")
    meta_learner = LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1)
    meta_learner.fit(X_meta_train, y_val)
    print("Meta learner has been trained.")

    session_name = f"SimplifiedEnsembleMLPForest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    deploy_filename = f"{session_name}_deployable.joblib"
    deploy_path = os.path.join(DEPLOY_DIR, deploy_filename)

    ensemble_package = {
        'mlp_model_state_dict': mlp_model.state_dict(),
        'rf_model': rf_model,
        'meta_learner': meta_learner,
        'scaler': dataset.scaler,
        'label_encoder': dataset.label_encoder,
        'metadata': dataset.metadata,
        'model_type': 'EnsembleMLPForest'
    }

    joblib.dump(ensemble_package, deploy_path)
    print(f"The simplified ensemble model has been deployed successfully to: {deploy_path}")

    print("\n" + "#"*22 + " Final Simplified Ensemble Validation Report " + "="*22)
    final_predictions = meta_learner.predict(X_meta_train)
    print(classification_report(y_val, final_predictions, target_names=class_names, zero_division=0, digits=4))
    print("="*78)

#Same function as above except uses the SUPER simple dataset (With different mapping labels)
def create_super_simplified_ensemble_from_checkpoints(device):
    print("\n" + "#"*12 + " Super Simplified Ensemble Stacking Creation " + "="*12)

    print("Data is being loaded and validation set is being prepared...")
    try:
        dataset = IntrusionDataset(PROCESSED_DIR_SIMPLIFIED)
    except FileNotFoundError:
        print(f"Super Simplified processed data has NOT found. Please run option '2' first.")
        return

    #Uses the dedicated loader to help create a consistent validation set.
    _, val_loader = create_super_simplified_loaders(dataset, target_benign_percentage=0.20)
    if val_loader is None:
        print("Data loading FAILED! Ensemble creation aborted.")
        return
    input_size, num_classes = dataset.get_dims()
    class_names = dataset.get_class_names()

    print(f"\nSelecting the 'RELABELEDSIMPLE' model...")
    pytorch_sessions = sorted([d for d in os.listdir(CHECKPOINT_DIR) if 'RELABELEDSIMPLE' in d and 'RandomForest' not in d and os.path.isdir(os.path.join(CHECKPOINT_DIR, d))])
    if not pytorch_sessions:
        print("\nERROR: No 'RELABELEDSIMPLE' sessions found. Please train one using option 7."); return

    for i, s in enumerate(pytorch_sessions): print(f"{i+1}. {s}")
    try:
        selection = int(input(f"Select a PyTorch model session (1-{len(pytorch_sessions)}): ")) - 1
        pytorch_session_name = pytorch_sessions[selection]
    except (ValueError, IndexError):
        print("Invalid selection."); return

    pytorch_model_path = os.path.join(CHECKPOINT_DIR, pytorch_session_name, 'model_best.pth.tar')
    if not os.path.exists(pytorch_model_path):
        print(f"No best model found in session '{pytorch_session_name}'."); return

    try:
        checkpoint = torch.load(pytorch_model_path, map_location=device, weights_only=False)
        model_type = checkpoint['model_type']
        pytorch_model = instantiate_model(model_type, input_size, num_classes)
        pytorch_model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
        pytorch_model.to(device)
        pytorch_model.eval()
        print(f"Model ({model_type}) loaded successfully from '{pytorch_session_name}'.")
    except Exception as e:
        print(f"Unable to load model: {e}"); return

    print(f"\nLoading latest 'RELABELEDSIMPLE' RandomForest model...")
    rf_sessions = sorted([d for d in os.listdir(CHECKPOINT_DIR) if 'RandomForest_RELABELEDSIMPLE' in d and os.path.isdir(os.path.join(CHECKPOINT_DIR, d))])
    if not rf_sessions:
        print("\nNo 'RandomForest_RELABELEDSIMPLE' sessions found. Please train one using option 7."); return
    rf_model_path = os.path.join(CHECKPOINT_DIR, rf_sessions[-1], 'RandomForest_best.joblib')
    print(f"Using RF model from session: {rf_sessions[-1]}")
    if not os.path.exists(rf_model_path):
        print(f"Best model has NOT been found in session '{rf_sessions[-1]}'."); return

    try:
        rf_model = joblib.load(rf_model_path)
        print("Random Forest base model has been loaded successfully.")
    except Exception as e:
        print(f"Unable to load Random Forest model: {e}"); return

    print("\nCreating features from the validation set...")
    X_val = val_loader.dataset.dataset.features[val_loader.dataset.indices].numpy()
    y_val = val_loader.dataset.dataset.labels[val_loader.dataset.indices].numpy()

    all_pytorch_preds = []
    with torch.no_grad():
        for inputs, _ in tqdm(val_loader, desc=f"Getting {model_type} predictions", leave=False, colour='yellow'):
            inputs = inputs.to(device)
            outputs = pytorch_model(inputs)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            all_pytorch_preds.append(probabilities)
    pytorch_meta_features = np.concatenate(all_pytorch_preds)
    rf_meta_features = rf_model.predict_proba(X_val)
    X_meta_train = np.concatenate([pytorch_meta_features, rf_meta_features], axis=1)
    print("Features generated.")

    print("\nTraining learner and deploying the Super Simplified ensemble model...")
    meta_learner = LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1)
    meta_learner.fit(X_meta_train, y_val)
    print("Meta learner has finished training and is ready to use!")

    session_name = f"SuperSimplifiedEnsemble_{model_type}Forest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    deploy_filename = f"{session_name}_deployable.joblib"
    deploy_path = os.path.join(DEPLOY_DIR, deploy_filename)

    ensemble_package = {
        'pytorch_model_state_dict': pytorch_model.state_dict(),
        'pytorch_model_type': model_type, 
        'rf_model': rf_model,
        'meta_learner': meta_learner,
        'scaler': dataset.scaler,
        'label_encoder': dataset.label_encoder,
        'metadata': dataset.metadata,
        'model_type': 'EnsemblePytorchForest' 
    }

    joblib.dump(ensemble_package, deploy_path)
    print(f"Super Simplified Ensemble model deployed successfully to: {deploy_path}")

    print("\n" + "#"*22 + " Final Super Simplified Ensemble Validation Report " + "="*22)
    final_predictions = meta_learner.predict(X_meta_train)
    print(classification_report(y_val, final_predictions, target_names=class_names, zero_division=0, digits=4))
    print("#"*87)

#Deploye the best model from a training session in the DeployModel folder.
def deploy_model_from_session(session_name):
    session_dir = os.path.join(CHECKPOINT_DIR, session_name)
    os.makedirs(DEPLOY_DIR, exist_ok=True)

    best_model_path = None
    if 'RandomForest' in session_name:
        best_model_path = os.path.join(session_dir, 'RandomForest_best.joblib')
    else:
        best_model_path = os.path.join(session_dir, 'model_best.pth.tar')

    if not os.path.exists(best_model_path):
        print(f"Error finding the best model file in '{session_dir}' to deploy.")
        return

    deploy_filename = (f"{session_name}_deployable.joblib"
                   if 'RandomForest' in session_name
                   else f"{session_name}_deployable.pth")
    deploy_path = os.path.join(DEPLOY_DIR, deploy_filename)
    shutil.copy(best_model_path, deploy_path)
    print(f"Model from session '{session_name}' has been deployed successfully to: {deploy_path}")

#Function used to create a model instance from its string name.
def instantiate_model(model_type, input_size, num_classes):
    if model_type == 'MLP' or model_type == 'EnhancedMLP':
        return EnhancedMLP(input_size, HIDDEN_LAYERS, num_classes)
    elif model_type == 'LSTM' or model_type == 'EnhancedLSTM':
        return EnhancedLSTM(input_size, hidden_size=128, num_classes=num_classes)
    elif model_type == 'CNN' or model_type == 'EnhancedCNN':
        try:
            return EnhancedCNN(input_size, num_classes)
        except ValueError as e:
            print(f"CNN Error: {e}")
            return None
    elif model_type == 'TabTransformer':
        return TabTransformer(
            num_features=input_size, num_classes=num_classes, dim=TRANSFORMER_DIM,
            n_heads=TRANSFORMER_HEADS, n_layers=TRANSFORMER_LAYERS, dropout=TRANSFORMER_DROPOUT
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

#Loads a deployed model from (DeployedModel folder) and runs it to get its validation metrics, and generate a report.
#Also a helper function for the graphing function.
def get_validation_metrics(model_path, val_loader, device, dataset):
    model_filename = os.path.basename(model_path)
    input_size, num_classes = dataset.get_dims()
    all_preds, all_labels = [], []
    report_dict = {}

    try:
        if model_path.endswith('.joblib'):
            package = joblib.load(model_path)
            X_val = val_loader.dataset.dataset.features[val_loader.dataset.indices].numpy()
            y_val = val_loader.dataset.dataset.labels[val_loader.dataset.indices].numpy()
            all_labels = y_val

            #Checks to see if the loaded object is a dictionary (an ensemble package)
            if isinstance(package, dict):
                if package.get('model_type') == 'EnsemblePytorchForest':
                    pytorch_model_type = package['pytorch_model_type']
                    pytorch_model = instantiate_model(pytorch_model_type, input_size, num_classes)
                    pytorch_model.load_state_dict(package['pytorch_model_state_dict'])
                    pytorch_model.to(device); pytorch_model.eval()
                    rf_model = package['rf_model']
                    meta_learner = package['meta_learner']
                    
                    all_pytorch_preds_prob = []
                    with torch.no_grad():
                        for inputs, _ in val_loader:
                            inputs = inputs.to(device)
                            outputs = pytorch_model(inputs)
                            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
                            all_pytorch_preds_prob.append(probabilities)
                    pytorch_meta_features = np.concatenate(all_pytorch_preds_prob)
                    rf_meta_features = rf_model.predict_proba(X_val)
                    X_meta_val = np.concatenate([pytorch_meta_features, rf_meta_features], axis=1)
                    all_preds = meta_learner.predict(X_meta_val)

                elif package.get('model_type') == 'EnsembleMLPForest':
                    mlp_model = instantiate_model('EnhancedMLP', input_size, num_classes)
                    mlp_model.load_state_dict(package['mlp_model_state_dict'])
                    mlp_model.to(device); mlp_model.eval()
                    rf_model = package['rf_model']
                    meta_learner = package['meta_learner']
                    
                    all_mlp_preds_prob = []
                    with torch.no_grad():
                        for inputs, _ in val_loader:
                            inputs = inputs.to(device)
                            outputs = mlp_model(inputs)
                            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
                            all_mlp_preds_prob.append(probabilities)
                    mlp_meta_features = np.concatenate(all_mlp_preds_prob)
                    rf_meta_features = rf_model.predict_proba(X_val)
                    X_meta_val = np.concatenate([mlp_meta_features, rf_meta_features], axis=1)
                    all_preds = meta_learner.predict(X_meta_val)
            else:
                model = package
                all_preds = model.predict(X_val)

        elif model_path.endswith(('.pth', '.pth.tar')):
            if model_filename.startswith('Specialist_OvR_'):
                 raise NotImplementedError("Specialist model graphing not supported in this mode.")
            else:
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                model_type = checkpoint.get('model_type')
                if not model_type:
                    if 'MLP' in model_filename: model_type = 'EnhancedMLP'
                    elif 'LSTM' in model_filename: model_type = 'EnhancedLSTM'
                    elif 'CNN' in model_filename: model_type = 'EnhancedCNN'
                    elif 'TabTransformer' in model_filename: model_type = 'TabTransformer'
                    else: raise ValueError("Cannot determine model type")

                model = instantiate_model(model_type, input_size, num_classes)
                if model is None: return None, None
                model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
                model.to(device); model.eval()

                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        all_preds.extend(predicted.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
        else:
            return None, None

        metrics = {
            "model_name": model_filename.replace('_deployable.joblib', '').replace('_deployable.pth', ''),
            "accuracy": accuracy_score(all_labels, all_preds),
            "precision": precision_score(all_labels, all_preds, average='weighted', zero_division=0),
            "recall": recall_score(all_labels, all_preds, average='weighted', zero_division=0),
            "f1": f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        }
        report_dict = classification_report(all_labels, all_preds, target_names=dataset.get_class_names(), zero_division=0, output_dict=True)
        return metrics, report_dict

    except Exception as e:
        print(f"Error! No metrics found for: {model_filename}. Reason: {e}")
        return None, None

#Filters, and then evaluates models to create a comparison and per model graph(s).
def generate_validation_graphs(val_loader, device, dataset, val_choice):
    print("\n" + "#"*22 + " Generating Validation Graphs " + "="*22)
    os.makedirs(GRAPHS_DIR, exist_ok=True)

    all_deployed_models = sorted([f for f in os.listdir(DEPLOY_DIR) if f.endswith(('.pth', '.pth.tar', '.joblib'))])
    
    #Filtering of Models based on the validation dataset choice
    models_to_evaluate = []
    if val_choice == '1': 
        print("Filtering for Standard, Balanced, and Simplified models...")
        for m in all_deployed_models:
            if 'RELABELEDSIMPLE' not in m and 'SuperSimplified' not in m:
                models_to_evaluate.append(m)
    elif val_choice == '2': 
        print("Filtering for Super Simplified models...")
        for m in all_deployed_models:
            if 'RELABELEDSIMPLE' in m or 'SuperSimplified' in m:
                models_to_evaluate.append(m)
    
    if not models_to_evaluate:
        print("ERROR: NO compatible deployed models found for the selected dataset. (Train some!)"); return

    print(f"Found {len(models_to_evaluate)} compatible models to evaluate...")
    all_metrics = []
    for model_file in tqdm(models_to_evaluate, desc="Evaluating models", unit="model"):
        model_path = os.path.join(DEPLOY_DIR, model_file)
        metrics, report_dict = get_validation_metrics(model_path, val_loader, device, dataset)
        if metrics and report_dict:
            all_metrics.append(metrics)
            #Generates a per model graph immediately after successful validation
            plot_per_model_graph(metrics['model_name'], report_dict)

    if not all_metrics:
        print("ERROR: No models could be successfully evaluated. Graph generation has failed!"); return
        
    print(f"\nSuccessfully validated the {len(all_metrics)} models. Individual graphs have been saved to '{PER_MODEL_GRAPHS_DIR}'.")

    #Sorts the models by accuracy in descending order.
    all_metrics.sort(key=lambda x: x['accuracy'], reverse=True)
    
    #Splits the models into models_per_graph chunks in order to keep the graphs clean and legible.
    models_per_graph = 5
    metric_chunks = [all_metrics[i:i + models_per_graph] for i in range(0, len(all_metrics), models_per_graph)]

    for i, chunk in enumerate(metric_chunks):
        model_names = [m['model_name'] for m in chunk]
        accuracy_scores = [m['accuracy'] for m in chunk]
        precision_scores = [m['precision'] for m in chunk]
        recall_scores = [m['recall'] for m in chunk]
        f1_scores = [m['f1'] for m in chunk]

        x = np.arange(len(model_names))
        width = 0.2

        fig, ax = plt.subplots(figsize=(20, 12))
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
    
        rects1 = ax.bar(x - 1.5*width, accuracy_scores, width, label='Accuracy', color='#002060')
        rects2 = ax.bar(x - 0.5*width, precision_scores, width, label='Precision (Weighted)', color='#0070C0')
        rects3 = ax.bar(x + 0.5*width, recall_scores, width, label='Recall (Weighted)', color='#B4C7E7')
        rects4 = ax.bar(x + 1.5*width, f1_scores, width, label='F1-Score (Weighted)', color='#C00000', hatch='..', edgecolor='white')

        for spine in ax.spines.values():
            spine.set_color('white')
        ax.tick_params(axis='x', colors='white', labelsize=14)
        ax.tick_params(axis='y', colors='white', labelsize=16)
        ax.grid(color='white', linestyle='--', linewidth=0.5, axis='y', alpha=0.5)

        ax.set_ylabel('Score', fontsize=21, color='white')
        ax.set_title(f'Model Performance Comparison (Part {i+1})', fontsize=30, color='white', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=25, ha="right")
        ax.set_ylim(0, 1.1)
        
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), facecolor='black', edgecolor='white', fontsize=16, labelcolor='white')

        for rects in [rects1, rects2, rects3, rects4]:
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.3f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', color='white', fontsize=10)

        fig.tight_layout()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        graph_path = os.path.join(GRAPHS_DIR, f"Validation_Comparison_{timestamp}_Part_{i+1}.png")
        plt.savefig(graph_path, facecolor='black', bbox_inches='tight')
        print(f"Comparison graph has been saved to: {graph_path}")
        plt.show()

#Creates and saves a bar chart for a single model, in order to save its validation report metrics
#Including precision, recall, and f1 score per each dataset metric (scuh as Benign or DoS).
def plot_per_model_graph(model_name, report_dict):
    os.makedirs(PER_MODEL_GRAPHS_DIR, exist_ok=True)
    
    class_names = [label for label in report_dict.keys() if label not in ['accuracy', 'macro avg', 'weighted avg']]
    
    precision = [report_dict[label]['precision'] for label in class_names]
    recall = [report_dict[label]['recall'] for label in class_names]
    f1_score = [report_dict[label]['f1-score'] for label in class_names]

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    rects1 = ax.bar(x - width, precision, width, label='Precision', color='#0070C0')
    rects2 = ax.bar(x, recall, width, label='Recall', color='#B4C7E7')
    rects3 = ax.bar(x + width, f1_score, width, label='F1-Score', color='#C00000', hatch='..', edgecolor='white')

    for spine in ax.spines.values():
        spine.set_color('white')
    ax.tick_params(axis='x', colors='white', labelsize=12)
    ax.tick_params(axis='y', colors='white', labelsize=12)
    ax.grid(color='white', linestyle='--', linewidth=0.5, axis='y', alpha=0.5)

    ax.set_ylabel('Score', fontsize=16, color='white')
    ax.set_title(f'Per-Class Performance: {model_name}', fontsize=20, color='white', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=20, ha="right")
    ax.set_ylim(0, 1.1)
    
    legend = ax.legend(facecolor='black', edgecolor='white', fontsize=14, labelcolor='white')

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', color='white', fontsize=9)

    autolabel(rects1); autolabel(rects2); autolabel(rects3)

    fig.tight_layout()
    graph_path = os.path.join(PER_MODEL_GRAPHS_DIR, f"{model_name}.png")
    plt.savefig(graph_path, facecolor='black', bbox_inches='tight')
    plt.close(fig)

#Model is validated to get its performance metrics to see how good it is.
def run_validation_on_deployed_model(model_path, val_loader, device, dataset):
    model_filename = os.path.basename(model_path)
    print("\n" + "#"*22 + f" Validating: {model_filename} " + "#"*22)

    input_size, num_classes = dataset.get_dims()
    class_names = dataset.get_class_names()

    if model_path.endswith('.joblib'):
        try:
            package = joblib.load(model_path)
            if isinstance(package, dict) and package.get('model_type') == 'EnsemblePytorchForest':
                print("Found Super Simplified Ensemble Model package. Validating...")
                pytorch_model_type = package['pytorch_model_type']
                pytorch_model = instantiate_model(pytorch_model_type, input_size, num_classes)
                pytorch_model.load_state_dict(package['pytorch_model_state_dict'])
                pytorch_model.to(device); pytorch_model.eval()
                rf_model = package['rf_model']
                meta_learner = package['meta_learner']
                X_val = val_loader.dataset.dataset.features[val_loader.dataset.indices].numpy()
                y_val = val_loader.dataset.dataset.labels[val_loader.dataset.indices].numpy()

                all_pytorch_preds = []
                with torch.no_grad():
                    for inputs, _ in tqdm(val_loader, desc=f"Getting {pytorch_model_type} predictions", leave=False):
                        inputs = inputs.to(device)
                        outputs = pytorch_model(inputs)
                        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
                        all_pytorch_preds.append(probabilities)
                pytorch_meta_features = np.concatenate(all_pytorch_preds)
                rf_meta_features = rf_model.predict_proba(X_val)
                X_meta_val = np.concatenate([pytorch_meta_features, rf_meta_features], axis=1)
                final_predictions = meta_learner.predict(X_meta_val)

                print("\n" + "="*20 + " Final Ensemble Validation Report " + "="*20)
                print(classification_report(y_val, final_predictions, target_names=class_names, zero_division=0, digits=4))
                print("="*70)

            elif isinstance(package, dict) and package.get('model_type') == 'EnsembleMLPForest':
                print("Found Standard/Simplified Ensemble Model package. Validating...")
                mlp_model = instantiate_model('EnhancedMLP', input_size, num_classes)
                mlp_model.load_state_dict(package['mlp_model_state_dict'])
                mlp_model.to(device); mlp_model.eval()
                rf_model = package['rf_model']
                meta_learner = package['meta_learner']
                X_val = val_loader.dataset.dataset.features[val_loader.dataset.indices].numpy()
                y_val = val_loader.dataset.dataset.labels[val_loader.dataset.indices].numpy()

                all_mlp_preds = []
                with torch.no_grad():
                    for inputs, _ in tqdm(val_loader, desc="Getting MLP predictions", leave=False):
                        inputs = inputs.to(device)
                        outputs = mlp_model(inputs)
                        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
                        all_mlp_preds.append(probabilities)
                mlp_meta_features = np.concatenate(all_mlp_preds)
                rf_meta_features = rf_model.predict_proba(X_val)
                X_meta_val = np.concatenate([mlp_meta_features, rf_meta_features], axis=1)
                final_predictions = meta_learner.predict(X_meta_val)

                print("\n" + "#" * 22 + " Final Ensemble Validation Report " + "#" * 22)
                print(classification_report(y_val, final_predictions, target_names=class_names, zero_division=0, digits=4))
                print("#"*75)
            else:
                model = package
                print(f"Found and loaded the Random Forest model: {model_filename}")
                X_val = val_loader.dataset.dataset.features[val_loader.dataset.indices].numpy()
                y_val = val_loader.dataset.dataset.labels[val_loader.dataset.indices].numpy()
                print("Calcluating validation set...")
                y_pred = model.predict(X_val)
                print("\n" + "#" * 22 + " Final Validation Report " + "#" * 22)
                print(classification_report(y_val, y_pred, target_names=class_names, zero_division=0, digits=4))
                print("#"*67)
        except Exception as e:
            print(f"ERROR, could not load or validate the model: {e}")
    elif model_path.endswith(('.pth', '.pth.tar')):
        try:
            #Handles specialist model (experimental).
            is_specialist = model_filename.startswith('Specialist_OvR_')
            if is_specialist:
                specialist_class_name = model_filename.split('_')[2]
                print(f"Detected Specialist Model for class: '{specialist_class_name}'. Evaluating in binary mode.")
                
                label_encoder = dataset.label_encoder
                positive_class_index = list(label_encoder.classes_).index(specialist_class_name)
                
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                model = instantiate_model(checkpoint['model_type'], input_size, 2) 
                if model is None: return 
                model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
                model.to(device)
                model.eval()

                all_preds, all_labels_binary = [], []
                with torch.no_grad():
                    for inputs, labels_multiclass in tqdm(val_loader, desc="Evaluating Specialist Model", leave=False):
                        inputs = inputs.to(device)
                        labels_binary = (labels_multiclass == positive_class_index).long().to(device)
                        outputs = model(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        all_preds.extend(predicted.cpu().numpy())
                        all_labels_binary.extend(labels_binary.cpu().numpy())
                
                print("\n" + "="*15 + f" Specialist Validation Report ({specialist_class_name} vs. Rest) " + "="*15)
                print(classification_report(all_labels_binary, all_preds, target_names=['Rest', specialist_class_name], zero_division=0, digits=4))
                print("="*75)
                return 

            #Multi class model logic.
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            model_type = checkpoint.get('model_type')
            if not model_type:
                if 'MLP' in model_filename: model_type = 'EnhancedMLP'
                elif 'LSTM' in model_filename: model_type = 'EnhancedLSTM'
                elif 'CNN' in model_filename: model_type = 'EnhancedCNN'
                elif 'TabTransformer' in model_filename: model_type = 'TabTransformer'
                else:
                    print(f"ERROR: Unable to find the model type for '{model_filename}'.")
                    return
            print(f"Instantiating {model_type} architecture...")
            model = instantiate_model(model_type, input_size, num_classes)
            if model is None: return
            model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
            model.to(device)
            print(f"Load successful for model : {model_filename}")
            evaluate_on_validation_set(model, val_loader, device, class_names)
        except Exception as e:
            print(f"Unable to load or validate the model: {e}")
    else:
        print(f"Unknown model file type: {model_filename}. Skipping.")


### Main Menu System ###

#Checks for hardware (Note CUDA was not tested, training was run on Apple Silicon).
def get_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("MPS compatible Apple Silicon GPU has been found and will be used to accelerate training.")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("CUDA compatible NVIDIA GPU has been found available and will be used to accelerate training.")
        return torch.device("cuda")
    else:
        print("No GPU detected. Defaulting to CPU. (This MAY be very slow, unless there's like a 128 Core+ CPU being used)")
        return torch.device("cpu")

def print_main_menu():
    print("\n" + "#" * 22 + " Intrusion Detection System " + "#" * 22)
    print("### Data & Setup ###")
    print(" 1. Pre-process Raw CSV Data (Standard)")
    print(" 2. Pre-process Raw CSV Data (Super Simplified)")
    print(" 3. List Available Raw Datasets")
    print("\n### Standard Model Training ###")
    print(" 4. Train a New Model (Standard Imbalanced Data)")
    print(" 5. Train a New Model (Balanced Data via Down-sampling)")
    print(" 6. Train a New Model (Simplified Data: Balanced & No Web/Infiltration)")
    print("\n### Super Simplified Model Training ###")
    print(" 7. Train a New Model (Super Simplified)")
    print(" 8. Train All Models (Super Simplified)")
    print("\n### Bulk Training & Continuation ###")
    print(" 9. Train All Models (Standard)")
    print(" 10. Train All Models (Balanced)")
    print(" 11. Continue a Paused Training Session")
    print("\n### Advanced & Ensemble ###")
    print(" 12. Create Stacking Ensemble (from balanced MLP+RF)")
    print(" 13. Create Stacking Ensemble (from simplified MLP+RF)")
    print(" 14. Create Stacking Ensemble (from Super Simplified Models)")
    print(" 15. Train a Specialist OvR Ensemble")
    print("\n### Evaluation & Deployment ###")
    print(" 16. Validate Deployed Models")
    print(" 17. Generate Validation Performance Graphs")
    print(" 18. Deploy Best Model from a Session")
    print("\n### Management ###")
    print(" 19. List Available Training Sessions")
    print(" 20. Exit")
    print("#" * 71)

def start_cli():
    device = get_device()
    for path in [DATASET_DIR, PROCESSED_DIR, PROCESSED_DIR_SIMPLIFIED, CHECKPOINT_DIR, DEPLOY_DIR, GRAPHS_DIR]:
        os.makedirs(path, exist_ok=True)

    while True:
        print_main_menu()
        choice = input("Enter your choice: ").strip()

        standard_data_required = ['4', '5', '6', '9', '10', '12', '13']
        simplified_data_required = ['7', '8', '14']
        validation_required = ['16', '17']
        specialist_required = ['15']

        is_standard_data_available = os.path.exists(os.path.join(PROCESSED_DIR, 'features.pt'))
        is_simplified_data_available = os.path.exists(os.path.join(PROCESSED_DIR_SIMPLIFIED, 'features.pt'))

        if choice in standard_data_required and not is_standard_data_available:
            print("\nError: Standard processed data not found. Please run option '1' first.")
            continue
        if choice in simplified_data_required and not is_simplified_data_available:
            print(f"\nError: Super Simplified processed data not found. Please run option '2' first.")
            continue
        if choice in validation_required and not is_standard_data_available and not is_simplified_data_available:
            print("\nError: No processed data found. Please run option '1' or '2' before validating.")
            continue
        if choice in specialist_required and not is_standard_data_available and not is_simplified_data_available:
            print("\nError: No processed data found. Please run option '1' or '2' before training specialists.")
            continue
        if choice == '11': 
             if not is_standard_data_available and not is_simplified_data_available:
                 print("\nError: No processed data found. Please run option '1' or '2' before continuing a session.")
                 continue

        if choice == '1' or choice == '2':
            try:
                all_files = sorted([f for f in os.listdir(DATASET_DIR) if f.endswith('.csv')])
                if not all_files: print(f"No CSV files found in {DATASET_DIR}"); continue
            except FileNotFoundError:
                print(f"Dataset directory not found at: {DATASET_DIR}"); continue

            print("\n### Select Raw CSVs to Pre-process ###")
            for i, f in enumerate(all_files): print(f"{i+1}. {f}")
            print("all. Process all listed files")

            selection = input("Enter selection (e.g., '1,3,5' or 'all'): ").strip().lower()
            if not selection: print("INVALID selection."); continue

            try:
                selected_files = all_files if selection == 'all' else [all_files[int(i)-1] for i in selection.split(',')]
            except (ValueError, IndexError): print("INVALID selection format."); continue

            dataset_paths = [os.path.join(DATASET_DIR, f) for f in selected_files]
            if choice == '1':
                print("\n### Starting Standard Pre-processing ###")
                preprocess_and_save_data(dataset_paths, PROCESSED_DIR, LABEL_MAPPINGS)
            else: 
                print("\n### Starting Super Simplified Pre-processing ###")
                preprocess_and_save_data(dataset_paths, PROCESSED_DIR_SIMPLIFIED, LABEL_MAPPINGS_SIMPLIFIED_)

        elif choice == '3': 
             print("\n" + "_"*5 + " Available Raw CSV Datasets " + "─"*5)
             try:
                files = sorted([f for f in os.listdir(DATASET_DIR) if f.endswith('.csv')])
                if not files: print("No CSV files found.")
                for f in files: print(f" - {f}")
             except FileNotFoundError:
                print(f"Dataset directory not found at: {DATASET_DIR}")

        #Train new models (All types)
        elif choice in ['4', '5', '6', '7']:
            model_map = {'1': 'EnhancedMLP', '2': 'EnhancedLSTM', '3': 'EnhancedCNN', '4': 'RandomForest', '5': 'TabTransformer'}
            
            if choice == '4':
                title, data_dir, loader_fn, session_suffix = "Standard Imbalanced", PROCESSED_DIR, None, ""
            elif choice == '5':
                title, data_dir, loader_fn, session_suffix = "Balanced", PROCESSED_DIR, create_balanced_loaders, "_REDUCED"
            elif choice == '6':
                title, data_dir, loader_fn, session_suffix = "Simplified", PROCESSED_DIR, create_simplified_loaders, "_SIMPLIFIED"
            elif choice == '7':
                title, data_dir, loader_fn, session_suffix = "Super Simplified", PROCESSED_DIR_SIMPLIFIED, create_super_simplified_loaders, "_RELABELEDSIMPLE"

            print(f"\n" + "─"*10 + f" Train New Model ({title}) " + "─"*10)
            print("1. MLP (Multi-Layer Perceptron)")
            print("2. LSTM (Recurrent Neural Network)")
            print("3. CNN (Convolutional Neural Network)")
            print("4. RandomForest (Ensemble Method)")
            print("5. TabTransformer (Advanced Attention Model)")
            model_choice = input("Enter model choice: ").strip()

            if model_choice not in model_map: print("INVALID model choice."); continue
            model_key = model_map[model_choice]

            dataset = IntrusionDataset(data_dir)
            input_size, num_classes = dataset.get_dims()
            class_names = dataset.get_class_names()
            #For balanced, simplified, and super simplified.
            if loader_fn: 
                train_loader, val_loader = loader_fn(dataset, target_benign_percentage=0.20)
                if train_loader is None: continue
            #For standard imbalanced.
            else:
                indices = list(range(len(dataset)))
                train_indices, val_indices = train_test_split(indices, test_size=VALIDATION_SPLIT, stratify=dataset.labels.numpy(), random_state=42)
                train_loader = DataLoader(Subset(dataset, train_indices), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
                val_loader = DataLoader(Subset(dataset, val_indices), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
            
            session_name = f"{model_key}{session_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
             #Else handles instantiation failure (e.g. for CNN)
            if model_key == 'RandomForest':
                train_random_forest(train_loader, val_loader, session_name, class_names)
            else:
                model = instantiate_model(model_key, input_size, num_classes)
                if model is None: continue 
                train_pytorch_model(model, train_loader, val_loader, device, session_name, class_names)

        #Train all the models (may take a while).
        elif choice in ['8', '9', '10']:
            if choice == '8':
                title, data_dir, loader_fn, session_suffix = "Super Simplified", PROCESSED_DIR_SIMPLIFIED, create_super_simplified_loaders, "_RELABELEDSIMPLE"
            elif choice == '9':
                title, data_dir, loader_fn, session_suffix = "Standard Imbalanced", PROCESSED_DIR, None, ""
            elif choice == '10':
                title, data_dir, loader_fn, session_suffix = "Balanced", PROCESSED_DIR, create_balanced_loaders, "_REDUCED"

            print(f"\n" + "─"*10 + f" Train All Models Sequentially ({title}) " + "─"*10)
            models_to_train = ['EnhancedMLP', 'EnhancedLSTM', 'EnhancedCNN', 'RandomForest', 'TabTransformer']
            dataset = IntrusionDataset(data_dir)
            input_size, num_classes = dataset.get_dims()
            class_names = dataset.get_class_names()

            if loader_fn:
                train_loader, val_loader = loader_fn(dataset, target_benign_percentage=0.20)
                if train_loader is None: continue
            else:
                indices = list(range(len(dataset)))
                train_indices, val_indices = train_test_split(indices, test_size=VALIDATION_SPLIT, stratify=dataset.labels.numpy(), random_state=42)
                train_loader = DataLoader(Subset(dataset, train_indices), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
                val_loader = DataLoader(Subset(dataset, val_indices), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
            print("Data loaders are now ready!")

            for i, model_key in enumerate(models_to_train):
                print(f"\n{'#'*62}\nStarting Training for Model {i+1}/{len(models_to_train)}: {model_key}\n{'#'*62}")
                session_name = f"{model_key}{session_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                if model_key == 'RandomForest':
                    train_random_forest(train_loader, val_loader, session_name, class_names)
                else:
                    model = instantiate_model(model_key, input_size, num_classes)
                    if model is None: continue 
                    train_pytorch_model(model, train_loader, val_loader, device, session_name, class_names)
            print(f"\n{ '*' * 72 }\nAll models have been trained successfully! \n{ '*' * 72}")

        elif choice == '11':
            sessions = sorted([d for d in os.listdir(CHECKPOINT_DIR) if os.path.isdir(os.path.join(CHECKPOINT_DIR, d)) and 'RandomForest' not in d])
            if not sessions: print("\nNo useable training sessions found!"); continue

            print("\n### Select a Session to Continue ###")
            for i, s in enumerate(sessions): print(f"{i+1}. {s}")
            try:
                session_name = sessions[int(input(f"Select a session (1-{len(sessions)}): ")) - 1]
            except (ValueError, IndexError): print("Selection is invalid!"); continue

            checkpoint_path = os.path.join(CHECKPOINT_DIR, session_name, 'interrupt.pth.tar')
            if not os.path.exists(checkpoint_path):
                checkpoint_path = os.path.join(CHECKPOINT_DIR, session_name, 'checkpoint.pth.tar')
                if not os.path.exists(checkpoint_path):
                    print(f"NO checkpoint or interrupt file found for session '{session_name}'."); continue

            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            start_epoch = checkpoint['epoch']
            if start_epoch >= EPOCHS: print(f"\nNote: Training for '{session_name}' is already complete!"); continue
            
            #Determines which data loader to used based on the session name.
            if "_RELABELEDSIMPLE" in session_name:
                print("Resuming session with SUPER SIMPLIFIED data loader.")
                dataset = IntrusionDataset(PROCESSED_DIR_SIMPLIFIED)
                loader_fn = create_super_simplified_loaders
            elif "_SIMPLIFIED" in session_name:
                print("Resuming session with SIMPLIFIED data loader.")
                dataset = IntrusionDataset(PROCESSED_DIR)
                loader_fn = create_simplified_loaders
            elif "_REDUCED" in session_name:
                print("Resuming session with REDUCED (balanced) data loader.")
                dataset = IntrusionDataset(PROCESSED_DIR)
                loader_fn = create_balanced_loaders
            elif "Specialist_OvR" in session_name:
                print("UNABLE to continue a Specialist training run directly. Please start a new one.")
                continue
            else:
                print("Resuming session with STANDARD data loader.")
                dataset = IntrusionDataset(PROCESSED_DIR)
                loader_fn = None

            input_size, num_classes = dataset.get_dims()
            class_names = dataset.get_class_names()

            if loader_fn:
                train_loader, val_loader = loader_fn(dataset, target_benign_percentage=0.20)
            else:
                indices = list(range(len(dataset)))
                train_indices, val_indices = train_test_split(indices, test_size=VALIDATION_SPLIT, stratify=dataset.labels.numpy(), random_state=42)
                train_loader = DataLoader(Subset(dataset, train_indices), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
                val_loader = DataLoader(Subset(dataset, val_indices), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

            model = instantiate_model(checkpoint['model_type'], input_size, num_classes)
            if model is None: continue
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"\nTraining resuming for {session_name} from epoch {start_epoch + 1}.")
            train_pytorch_model(model, train_loader, val_loader, device, session_name, class_names, start_epoch=start_epoch)

        #Creates balanced ensemble
        elif choice == '12': 
            print("\n" + "_"*11 + " Create Ensemble Model from Checkpoints " + "_"*11)
            mlp_sessions = sorted([d for d in os.listdir(CHECKPOINT_DIR) if 'EnhancedMLP_REDUCED' in d])
            if not mlp_sessions: print("\nNo 'EnhancedMLP_REDUCED' sessions found. Train one with option 5."); continue
            print("\n### (1/2) Select the EnhancedMLP_REDUCED session ###")
            for i, s in enumerate(mlp_sessions): print(f"{i+1}. {s}")
            try:
                mlp_session = mlp_sessions[int(input(f"Select session (1-{len(mlp_sessions)}): ")) - 1]
                mlp_checkpoint_path = os.path.join(CHECKPOINT_DIR, mlp_session, 'model_best.pth.tar')
                if not os.path.exists(mlp_checkpoint_path): print(f"Best model has NOT been found for '{mlp_session}'."); continue
            except (ValueError, IndexError): print("INVALID selection."); continue

            rf_sessions = sorted([d for d in os.listdir(CHECKPOINT_DIR) if 'RandomForest_REDUCED' in d])
            if not rf_sessions: print("\nNO 'RandomForest_REDUCED' sessions found. Train one with option 5."); continue
            print("\n### (2/2) Select the RandomForest_REDUCED session ###")
            for i, s in enumerate(rf_sessions): print(f"{i+1}. {s}")
            try:
                rf_session = rf_sessions[int(input(f"Select session (1-{len(rf_sessions)}): ")) - 1]
                rf_model_path = os.path.join(CHECKPOINT_DIR, rf_session, 'RandomForest_best.joblib')
                if not os.path.exists(rf_model_path): print(f"Best model not found for '{rf_session}'."); continue
            except (ValueError, IndexError): print("INVALID selection."); continue
            create_ensemble_from_checkpoints(device, mlp_checkpoint_path, rf_model_path)

        elif choice == '13': # Create Simplified Ensemble
            create_simplified_ensemble_from_checkpoints(device)

        elif choice == '14': # Create Super Simplified Ensemble
            create_super_simplified_ensemble_from_checkpoints(device)

        #Experimental sepcialist model comprised of multiple models.
        elif choice == '15':
            print("\n" + "─"*10 + " Train Specialist (One-vs-Rest) Ensemble " + "─"*10)
            print("Select the dataset to use for training the specialists:")
            print("1. Standard Data (Imbalanced)")
            print("2. Balanced/REDUCED Data (Down-sampled Benign)")
            print("3. Super Simplified Data")
            data_choice = input("Enter choice (1-3): ").strip()

            if data_choice == '1':
                title, data_dir, loader_fn, session_suffix = "Standard", PROCESSED_DIR, None, "_Standard"
            elif data_choice == '2':
                title, data_dir, loader_fn, session_suffix = "REDUCED", PROCESSED_DIR, create_balanced_loaders, "_REDUCED"
            elif data_choice == '3':
                title, data_dir, loader_fn, session_suffix = "SuperSimplified", PROCESSED_DIR_SIMPLIFIED, create_super_simplified_loaders, "_SuperSimple"
            else:
                print("ERROR: Invalid data choice!"); continue
            
            try:
                dataset = IntrusionDataset(data_dir)
            except FileNotFoundError:
                print(f"Processed data has NOT been found at {data_dir}. Please run the appropriate preprocessing first."); continue

            all_features_tensor = dataset.features
            all_labels_numpy = dataset.labels.numpy()
            label_encoder = dataset.label_encoder
            class_names = list(label_encoder.classes_)
            input_size, _ = dataset.get_dims()

            print("\nSelect training scope:")
            print("1. Train specialists for ALL classes")
            print("2. Train a specialist for a SINGLE class")
            scope_choice = input("Enter choice (1-2): ").strip()

            classes_to_train = []
            if scope_choice == '1':
                classes_to_train = class_names
            elif scope_choice == '2':
                print("\n### Select a Class to Train a Specialist For ###")
                for i, name in enumerate(class_names): print(f"{i+1}. {name}")
                try:
                    class_idx = int(input(f"Select class (1-{len(class_names)}): ")) - 1
                    classes_to_train.append(class_names[class_idx])
                except (ValueError, IndexError):
                    print("INVALID class selection."); continue
            else:
                print("INVALID scope choice."); continue

            model_map = {'1': 'EnhancedMLP', '2': 'EnhancedLSTM', '3': 'EnhancedCNN', '4': 'TabTransformer'}
            print("\nSelect model architecture for the specialists:")
            print("1. MLP | 2. LSTM | 3. CNN | 4. TabTransformer")
            model_choice = input("Enter model choice: ").strip()
            if model_choice not in model_map: print("INVALID model choice."); continue
            model_key = model_map[model_choice]

            for specialist_class_name in classes_to_train:
                print(f"\n{'#'*20} Training Specialist for: {specialist_class_name} {'#'*20}")

                #Creates a binary dataset for this specialist (1 for the class, 0 for all others)
                specialist_class_index = class_names.index(specialist_class_name)
                binary_labels = (all_labels_numpy == specialist_class_index).astype(int)
                binary_labels_tensor = torch.tensor(binary_labels, dtype=torch.long)
                
                #Creates a temporary dataset for splitting
                binary_dataset = TensorDataset(all_features_tensor, binary_labels_tensor)
                
                #Creates a stratified train/validation split for the binary problem
                train_indices, val_indices = train_test_split(
                    list(range(len(binary_dataset))),
                    test_size=VALIDATION_SPLIT,
                    stratify=binary_labels,
                    random_state=42
                )
                train_loader = DataLoader(Subset(binary_dataset, train_indices), batch_size=BATCH_SIZE, shuffle=True)
                val_loader = DataLoader(Subset(binary_dataset, val_indices), batch_size=BATCH_SIZE, shuffle=False)
                print(f"Created binary dataset: {np.sum(binary_labels)} positive samples, {len(binary_labels) - np.sum(binary_labels)} negative samples.")

                #Instantiates the model with 2 output classes for the binary problem
                specialist_model = instantiate_model(model_key, input_size, num_classes=2)
                if specialist_model is None:
                    print(f"Unable to create {model_key} model, skipping specialist for {specialist_class_name}."); continue

                session_name = f"Specialist_OvR_{specialist_class_name}_{model_key}{session_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                #Trains using the standard training function and to handle classification.
                train_pytorch_model(
                    specialist_model, 
                    train_loader, 
                    val_loader, 
                    device, 
                    session_name, 
                    class_names=['Rest', specialist_class_name] # Pass binary class names
                )

            print("\nAll the selected specialist models have been successfully trained!")

        elif choice == '16': 
            print("\n" + "_" * 11 + " Validate Deployed Model " + "_" * 11)
            try:
                deployed_models = sorted([f for f in os.listdir(DEPLOY_DIR) if f.endswith(('.pth', '.pth.tar', '.joblib'))])
                if not deployed_models: print("No deployed models found."); continue
            except FileNotFoundError: print(f"Deploy directory not found at: {DEPLOY_DIR}"); continue

            print("\n### Select a Deployed Model to Validate ###")
            for i, model_name in enumerate(deployed_models): print(f"{i+1}. {model_name}")
            print("all. Validate all listed models")
            selection = input("Enter selection (e.g., '1,2' or 'all'): ").strip().lower()
            if not selection: print("INVALID selection."); continue

            try:
                selected_models = deployed_models if selection == 'all' else [deployed_models[int(i)-1] for i in selection.split(',')]
            except (ValueError, IndexError): print("INVALID selection format!"); continue

            #Finds which dataset to load for validation.
            print("\nWhich dataset should be used for validation?")
            print("1. Standard Dataset (for standard, balanced, simplified, and specialist models)")
            print("2. Super Simplified Dataset (for relabeled models)")
            val_choice = input("Enter choice (1-2): ").strip()

            if val_choice == '1':
                data_dir = PROCESSED_DIR
                loader_fn = create_balanced_loaders 
            elif val_choice == '2':
                data_dir = PROCESSED_DIR_SIMPLIFIED
                loader_fn = create_super_simplified_loaders
            else:
                print("INVALID choice."); continue
            
            try:
                dataset = IntrusionDataset(data_dir)
            except FileNotFoundError:
                print(f"Processed data has NOT been found at {data_dir}. Please run the appropriate preprocessing command first."); continue

            _, val_loader = loader_fn(dataset, target_benign_percentage=0.20)
            print("Validation data is ready!")

            for model_file in selected_models:
                run_validation_on_deployed_model(os.path.join(DEPLOY_DIR, model_file), val_loader, device, dataset)

        #Generates the validation graphs.
        elif choice == '17':
            print("\nWhich dataset should be used for validation graphs?")
            print("1. Standard Dataset")
            print("2. Super Simplified Dataset")
            val_choice = input("Enter choice (1-2): ").strip()

            if val_choice == '1':
                data_dir = PROCESSED_DIR
                loader_fn = create_balanced_loaders
            elif val_choice == '2':
                data_dir = PROCESSED_DIR_SIMPLIFIED
                loader_fn = create_super_simplified_loaders
            else:
                print("INVALID choice."); continue
            
            try:
                dataset = IntrusionDataset(data_dir)
            except FileNotFoundError:
                print(f"Processed data could not be found at {data_dir}. Please run the preprocessing command first."); continue
            
            _, val_loader = loader_fn(dataset, target_benign_percentage=0.20)
            generate_validation_graphs(val_loader, device, dataset, val_choice)

        #Deploys the model to the DeployModel folder.
        elif choice == '18': 
            sessions = sorted([d for d in os.listdir(CHECKPOINT_DIR) if os.path.isdir(os.path.join(CHECKPOINT_DIR, d))])
            if not sessions: print("\nNo training sessions found."); continue

            print("\n### Select a Session to Deploy Its Best Model ###")
            for i, s in enumerate(sessions): print(f"{i+1}. {s}")
            try:
                session_name = sessions[int(input(f"Select a session (1-{len(sessions)}): ")) - 1]
            except (ValueError, IndexError): print("INVALID selection."); continue
            deploy_model_from_session(session_name)

        elif choice == '19': 
            print("\n" + "_" * 7 + " Available Training Sessions " + "_" * 7)
            try:
                sessions = sorted([d for d in os.listdir(CHECKPOINT_DIR) if os.path.isdir(os.path.join(CHECKPOINT_DIR, d))])
                if not sessions: print(" - No training sessions found.")
                for s in sessions: print(f" - {s}")
            except FileNotFoundError:
                print(f"The checkpoint directory has NOT been found at: {CHECKPOINT_DIR}")

        elif choice == '20':
            print("\nExiting program..."); break

        else:
            print("\nINVALID choice. Please try again.")

if __name__ == '__main__':
    start_cli()