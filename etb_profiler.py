#Authors: Serban Voinea Gabreanu, Gur Rehmat Singh Chahal, Algoma University, COSC5906002 Advanced Topics of Computer Networks (25SP), Final Project.
#This script is responsible for running a website where a user is able to upload csv files from wireshark (Converted from PCAP to CSV with CICFLOWMETER (Python version)
#and also for copy pasting data directly from the IDS 2018 Dataset in order to see if network traffic is benign or not.)
#This script also uses flask to run the website, which has an index.html file that uses some javascript, and also a styles.css file for making the website
#look nicer.

import os
import io
import sys
import torch
import torch.nn as nn
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, flash, url_for
import logging
import warnings
import re
import json
from models import EnhancedMLP, EnhancedLSTM, EnhancedCNN, TabTransformer


#Sets up the logging to track the scripts behaviour better.
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

### Configuration & Setup ###
app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'a-fallback-key-for-development-only')
Force_White_Space = False 

#File paths (using relative path, if its not working an absolute path can be set)
#Each directory is checked for relative location, if not found it will use an absolute path.
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

ABSOLUTE_BASE_DIR = '/Users/serbanvg/Documents/School 2025/2025 Spring Algoma/COSC5906 Networking/Final Project'

DEPLOY_DIR = os.path.join(BASE_DIR, 'DeployModel')
if not os.path.isdir(DEPLOY_DIR):
    DEPLOY_DIR = os.path.join(ABSOLUTE_BASE_DIR, 'DeployModel')

PROCESSED_DIR = os.path.join(BASE_DIR, 'ProcessedDataset')
if not os.path.isdir(PROCESSED_DIR):
    PROCESSED_DIR = os.path.join(ABSOLUTE_BASE_DIR, 'ProcessedDataset')

PROCESSED_DIR_SIMPLIFIED = os.path.join(BASE_DIR, 'ProcessedDatasetSimplified')
if not os.path.isdir(PROCESSED_DIR_SIMPLIFIED):
    PROCESSED_DIR_SIMPLIFIED = os.path.join(ABSOLUTE_BASE_DIR, 'ProcessedDatasetSimplified')

#Global dictionary to hold loaded specialist models. (Experimental).
SPECIALIST_MODELS = {}

def instantiate_pytorch_model(model_type, input_size, num_classes):
    hidden_layers_mlp = [256, 128, 64]
    if model_type == 'MLP' or model_type == 'EnhancedMLP':
        return EnhancedMLP(input_size, hidden_layers_mlp, num_classes)
    elif model_type == 'LSTM' or model_type == 'EnhancedLSTM':
        return EnhancedLSTM(input_size, hidden_size=128, num_classes=num_classes)
    elif model_type == 'CNN' or model_type == 'EnhancedCNN':
        return EnhancedCNN(input_size, num_classes)
    elif model_type == 'TabTransformer':
        return TabTransformer(
            num_features=input_size, num_classes=num_classes,
            dim=32, n_heads=8, n_layers=6, dropout=0.1 
        )
    else:
        raise ValueError(f"Unknown PyTorch model type: {model_type}")

### Preprocessing Data. ###
def preprocess_input_data(df, expected_features):
    logger.info(f"Received {df.shape[0]} row(s) with {df.shape[1]} columns for initial preprocessing.")
    df.columns = [col.strip() for col in df.columns]
    
    #Defines the columns that are never used as features for the model.
    cols_to_drop = ['Timestamp', 'Label', 'Src IP', 'Dst IP', 'Src Port']
    
    df_features = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
    
    #Aligns the data to the model's blueprint.
    df_aligned = df_features.reindex(columns=expected_features, fill_value=0)
    
    for col in df_aligned.columns:
        df_aligned[col] = pd.to_numeric(df_aligned[col], errors='coerce')
    df_aligned.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_aligned.fillna(0, inplace=True)
    
    df_final = df_aligned.astype(np.float32)
    logger.info(f"Preprocessing complete. Data aligned to {df_final.shape[1]} features.")
    return df_final


### Value Prediction Functions ###

def run_specialist_prediction(data_df, specialist_group):
    """
    [DEBUG] New prediction function for the specialist ensemble.
    It queries every specialist in the group and uses the "Highest Confidence Wins" strategy.
    """
    logger.info(f"Running prediction with a specialist ensemble of {len(specialist_group)} models.")
    predictions = {}

    #Gets an "opinion" of every specialist in the group (for the experimental specialist feature).
    with torch.no_grad():
        for model_filename in specialist_group:
            specialist_info = SPECIALIST_MODELS.get(model_filename)
            if not specialist_info:
                logger.warning(f"Could not find loaded specialist model for {model_filename}. Skipping.")
                continue

            model = specialist_info['model']
            scaler = specialist_info['scaler']
            class_name = specialist_info['class_name']

            processed_df = preprocess_input_data(data_df, scaler.feature_names_in_)
            scaled_data = scaler.transform(processed_df)

            data_tensor = torch.tensor(scaled_data, dtype=torch.float32)
            output_logits = model(data_tensor)
            confidence = torch.softmax(output_logits, dim=1)[0][1].item()
            predictions[class_name] = confidence

    if not predictions:
        raise ValueError("No specialist models could be run for this ensemble.")

    final_prediction = max(predictions, key=predictions.get)
    final_confidence = predictions[final_prediction]

    logger.info(f"Specialist Prediction: '{final_prediction}' with confidence {final_confidence:.2%}")
    return final_prediction, final_confidence, predictions

#Runs the prediction for a single model (or non specialist) ensemble.
def run_prediction(model_path, data_df, scaler, label_encoder):
    processed_df = preprocess_input_data(data_df, scaler.feature_names_in_)

    if processed_df.shape[1] != len(scaler.feature_names_in_):
        raise ValueError(f"Error in data processing! Expected {len(scaler.feature_names_in_)} features, but got {processed_df.shape[1]}.")

    scaled_data = scaler.transform(processed_df)

    model_filename = os.path.basename(model_path)
    logger.info(f"Model used for Analysis: {model_filename}")

    if model_path.endswith('.joblib'):
        package = joblib.load(model_path)
        input_size = scaler.n_features_in_
        num_classes = len(label_encoder.classes_)

        if isinstance(package, dict) and package.get('model_type') == 'EnsemblePytorchForest':
            logger.info("Detected EnsemblePytorchForest model. Running stacking prediction.")
            pytorch_model_type = package['pytorch_model_type']
            pytorch_model = instantiate_pytorch_model(pytorch_model_type, input_size, num_classes)
            pytorch_model.load_state_dict(package['pytorch_model_state_dict'])
            pytorch_model.eval()
            data_tensor = torch.tensor(scaled_data, dtype=torch.float32)
            with torch.no_grad():
                pytorch_logits = pytorch_model(data_tensor)
                pytorch_probs = torch.softmax(pytorch_logits, dim=1).numpy()
            rf_model = package['rf_model']
            rf_probs = rf_model.predict_proba(scaled_data)
            meta_features = np.concatenate([pytorch_probs, rf_probs], axis=1)
            meta_learner = package['meta_learner']
            final_predictions = meta_learner.predict(meta_features)
            final_probabilities = meta_learner.predict_proba(meta_features)

        elif isinstance(package, dict) and package.get('model_type') == 'EnsembleMLPForest':
            logger.info("Detected EnsembleMLPForest model. Running stacking prediction.")
            mlp_model = instantiate_pytorch_model('EnhancedMLP', input_size, num_classes)
            mlp_model.load_state_dict(package['mlp_model_state_dict'])
            mlp_model.eval()
            data_tensor = torch.tensor(scaled_data, dtype=torch.float32)
            with torch.no_grad():
                mlp_logits = mlp_model(data_tensor)
                mlp_probs = torch.softmax(mlp_logits, dim=1).numpy()
            rf_model = package['rf_model']
            rf_probs = rf_model.predict_proba(scaled_data)
            meta_features = np.concatenate([mlp_probs, rf_probs], axis=1)
            meta_learner = package['meta_learner']
            final_predictions = meta_learner.predict(meta_features)
            final_probabilities = meta_learner.predict_proba(meta_features)

        else:
            logger.info("Detected standard scikit-learn model (e.g., RandomForest).")
            model = package
            final_predictions = model.predict(scaled_data)
            final_probabilities = model.predict_proba(scaled_data)

        pred_label = label_encoder.inverse_transform(final_predictions)[0]
        confidence = final_probabilities[0].max()
        class_probs = {cls: prob for cls, prob in zip(label_encoder.classes_, final_probabilities[0])}

    elif model_path.endswith(('.pth', '.pth.tar')):
        input_size = scaler.n_features_in_
        num_classes = len(label_encoder.classes_)
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        model_type = checkpoint.get('model_type')
        if not model_type: 
            model_type = model_filename.split('_')[0]
        model = instantiate_pytorch_model(model_type, input_size, num_classes)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
        model.eval()
        data_tensor = torch.tensor(scaled_data, dtype=torch.float32)
        with torch.no_grad():
            output_logits = model(data_tensor)
            softmax_probs = torch.softmax(output_logits, dim=1)
            confidence, pred_index = torch.max(softmax_probs, dim=1)
            pred_label = label_encoder.inverse_transform([pred_index.item()])[0]
            confidence = confidence.item()
            class_probs = {cls: prob.item() for cls, prob in zip(label_encoder.classes_, softmax_probs[0])}
    else:
        raise ValueError(f"Unsupported model file type: {model_filename}")

    logger.info(f"Prediction: '{pred_label}' with confidence {confidence:.2%}")
    return pred_label, confidence, class_probs

### Flask Web Application ###

#Conversion logic to remap columns to IDS format from the CICFLOWMETER format and vise versa.
def remap_cicflowmeter_columns(df):
    logger.info("Remapping columns from CICFlowMeter format to IDS2018 format.")
    
    column_mapping = {
        #Dropped columns.
        'src_ip': 'Src IP', 'dst_ip': 'Dst IP', 'src_port': 'Src Port', 'dst_port': 'Dst Port',
        'protocol': 'Protocol', 'timestamp': 'Timestamp',
        
        #Core Flow Features
        'flow_duration': 'Flow Duration', 'tot_fwd_pkts': 'Tot Fwd Pkts', 'tot_bwd_pkts': 'Tot Bwd Pkts',
        'totlen_fwd_pkts': 'TotLen Fwd Pkts', 'totlen_bwd_pkts': 'TotLen Bwd Pkts',
        'fwd_pkt_len_max': 'Fwd Pkt Len Max', 'fwd_pkt_len_min': 'Fwd Pkt Len Min',
        'fwd_pkt_len_mean': 'Fwd Pkt Len Mean', 'fwd_pkt_len_std': 'Fwd Pkt Len Std',
        'bwd_pkt_len_max': 'Bwd Pkt Len Max', 'bwd_pkt_len_min': 'Bwd Pkt Len Min',
        'bwd_pkt_len_mean': 'Bwd Pkt Len Mean', 'bwd_pkt_len_std': 'Bwd Pkt Len Std',
        'flow_byts_s': 'Flow Byts/s', 'flow_pkts_s': 'Flow Pkts/s',
        
        #Flow IAT
        'flow_iat_mean': 'Flow IAT Mean', 'flow_iat_std': 'Flow IAT Std', 
        'flow_iat_max': 'Flow IAT Max', 'flow_iat_min': 'Flow IAT Min',
        
        #Forward IAT
        'fwd_iat_tot': 'Fwd IAT Tot', 'fwd_iat_mean': 'Fwd IAT Mean', 
        'fwd_iat_std': 'Fwd IAT Std', 'fwd_iat_max': 'Fwd IAT Max', 'fwd_iat_min': 'Fwd IAT Min',
        
        #Backward IAT
        'bwd_iat_tot': 'Bwd IAT Tot', 'bwd_iat_mean': 'Bwd IAT Mean', 
        'bwd_iat_std': 'Bwd IAT Std', 'bwd_iat_max': 'Bwd IAT Max', 'bwd_iat_min': 'Bwd IAT Min',
        
        #Flags
        'fwd_psh_flags': 'Fwd PSH Flags', 'bwd_psh_flags': 'Bwd PSH Flags',
        'fwd_urg_flags': 'Fwd URG Flags', 'bwd_urg_flags': 'Bwd URG Flags',
        'fin_flag_cnt': 'FIN Flag Cnt', 'syn_flag_cnt': 'SYN Flag Cnt',
        'rst_flag_cnt': 'RST Flag Cnt', 'psh_flag_cnt': 'PSH Flag Cnt',
        'ack_flag_cnt': 'ACK Flag Cnt', 'urg_flag_cnt': 'URG Flag Cnt',
        'cwr_flag_count': 'CWE Flag Count', 'ece_flag_cnt': 'ECE Flag Cnt',
        
        #Header and Packet Length
        'fwd_header_len': 'Fwd Header Len', 'bwd_header_len': 'Bwd Header Len',
        'fwd_pkts_s': 'Fwd Pkts/s', 'bwd_pkts_s': 'Bwd Pkts/s',
        'pkt_len_min': 'Pkt Len Min', 'pkt_len_max': 'Pkt Len Max',
        'pkt_len_mean': 'Pkt Len Mean', 'pkt_len_std': 'Pkt Len Std', 'pkt_len_var': 'Pkt Len Var',
        
        #Sizing and Ratios
        'down_up_ratio': 'Down/Up Ratio', 'pkt_size_avg': 'Pkt Size Avg',
        'fwd_seg_size_avg': 'Fwd Seg Size Avg', 'bwd_seg_size_avg': 'Bwd Seg Size Avg',
        'fwd_byts_b_avg': 'Fwd Byts/b Avg', 'fwd_pkts_b_avg': 'Fwd Pkts/b Avg',
        'fwd_blk_rate_avg': 'Fwd Blk Rate Avg', 'bwd_byts_b_avg': 'Bwd Byts/b Avg',
        'bwd_pkts_b_avg': 'Bwd Pkts/b Avg', 'bwd_blk_rate_avg': 'Bwd Blk Rate Avg',
        
        #Subflows
        'subflow_fwd_pkts': 'Subflow Fwd Pkts', 'subflow_fwd_byts': 'Subflow Fwd Byts',
        'subflow_bwd_pkts': 'Subflow Bwd Pkts', 'subflow_bwd_byts': 'Subflow Bwd Byts',
        
        #Window and Segment Size
        'init_fwd_win_byts': 'Init Fwd Win Byts', 'init_bwd_win_byts': 'Init Bwd Win Byts',
        'fwd_act_data_pkts': 'Fwd Act Data Pkts', 'fwd_seg_size_min': 'Fwd Seg Size Min',
        
        #Active/Idle
        'active_mean': 'Active Mean', 'active_std': 'Active Std', 
        'active_max': 'Active Max', 'active_min': 'Active Min',
        'idle_mean': 'Idle Mean', 'idle_std': 'Idle Std', 
        'idle_max': 'Idle Max', 'idle_min': 'Idle Min',
        
        #Label (will be dropped)
        'label': 'Label'
    }
    
    #Makes sure all of the column names in the DataFrame are clean before renaming.
    df.columns = df.columns.str.strip()
    df_renamed = df.rename(columns=column_mapping)
    return df_renamed

#This function is made to handle both uploaded CSV files and direct copy pasting of traffic flow in the textbox on the website.
#It also detects the input structure to correctly parse the information.
def parse_input_data(data_stream, data_format, ids2018_columns):
    raw_text = data_stream.read().decode("UTF8", errors='ignore').strip()
    
    if not raw_text:
        logger.warning("Input data is empty. Returning empty DataFrame.")
        return pd.DataFrame()

    logger.info(f"Attempting to parse content (showing first 500 chars):\n{raw_text[:500]}...")
    
    #Checks for newlines. Uploaded files or multi-line pastes will have them.
    #Single-line pastes should not.
    is_multiline_or_has_header = '\n' in raw_text

    try:
        #Case 1: Uploaded file or multi-line paste. Assume the first row is a header.
        if is_multiline_or_has_header:
            logger.info("Detected multi-line input. Assuming header is present (using pandas header=0).")
            final_stream = io.StringIO(raw_text)
            
            if data_format == 'cicflowmeter':
                logger.info("Processing as 'WireShark CICFlowMeter' format.")
                input_df = pd.read_csv(final_stream, header=0, on_bad_lines='warn', skipinitialspace=True)
                input_df = remap_cicflowmeter_columns(input_df)
            else: #ids2018 format
                logger.info("Processing as 'IDS 2018 Dataset' format.")
                input_df = pd.read_csv(final_stream, header=0, on_bad_lines='warn', skipinitialspace=True)
            
        #Case 2: Pasted single-line data. Assume no header.
        else:
            logger.info("Detected single-line input. Assuming no header is present (assigning names manually).")
            
            #Cleans the raw string to handle copy-paste errors
            values = raw_text.split(',')
            stripped_values = [v.strip() for v in values]
            clean_csv_string = ",".join(stripped_values)
            final_stream = io.StringIO(clean_csv_string)
            
            #List is needed to assign column names to headerless CICFlowMeter data
            cicflowmeter_live_columns = [
                'src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol', 'timestamp', 'flow_duration',
                'flow_byts_s', 'flow_pkts_s', 'fwd_pkts_s', 'bwd_pkts_s', 'tot_fwd_pkts',
                'tot_bwd_pkts', 'totlen_fwd_pkts', 'totlen_bwd_pkts', 'fwd_pkt_len_max',
                'fwd_pkt_len_min', 'fwd_pkt_len_mean', 'fwd_pkt_len_std', 'bwd_pkt_len_max',
                'bwd_pkt_len_min', 'bwd_pkt_len_mean', 'bwd_pkt_len_std', 'pkt_len_max',
                'pkt_len_min', 'pkt_len_mean', 'pkt_len_std', 'pkt_len_var', 'fwd_header_len',
                'bwd_header_len', 'fwd_seg_size_min', 'fwd_act_data_pkts', 'flow_iat_mean',
                'flow_iat_max', 'flow_iat_min', 'flow_iat_std', 'fwd_iat_tot', 'fwd_iat_max',
                'fwd_iat_min', 'fwd_iat_mean', 'fwd_iat_std', 'bwd_iat_tot', 'bwd_iat_max',
                'bwd_iat_min', 'bwd_iat_mean', 'bwd_iat_std', 'fwd_psh_flags', 'bwd_psh_flags',
                'fwd_urg_flags', 'bwd_urg_flags', 'fin_flag_cnt', 'syn_flag_cnt', 'rst_flag_cnt',
                'psh_flag_cnt', 'ack_flag_cnt', 'urg_flag_cnt', 'ece_flag_cnt', 'down_up_ratio',
                'pkt_size_avg', 'init_fwd_win_byts', 'init_bwd_win_byts', 'active_max',
                'active_min', 'active_mean', 'active_std', 'idle_max', 'idle_min', 'idle_mean',
                'idle_std', 'fwd_byts_b_avg', 'fwd_pkts_b_avg', 'bwd_byts_b_avg', 'bwd_pkts_b_avg',
                'fwd_blk_rate_avg', 'bwd_blk_rate_avg', 'fwd_seg_size_avg', 'bwd_seg_size_avg',
                'cwr_flag_count', 'subflow_fwd_pkts', 'subflow_bwd_pkts', 'subflow_fwd_byts',
                'subflow_bwd_byts'
            ]

            if data_format == 'cicflowmeter':
                logger.info("Processing as 'WireShark CICFlowMeter' format.")
                num_values = len(stripped_values)
                col_names = cicflowmeter_live_columns[:num_values]
                input_df = pd.read_csv(final_stream, header=None, names=col_names, on_bad_lines='warn')
                input_df = remap_cicflowmeter_columns(input_df)

            #Ids2018 format
            else: 
                logger.info("Processing as 'IDS 2018 Dataset' format.")
                input_df = pd.read_csv(final_stream, header=None, names=ids2018_columns, on_bad_lines='warn')

        logger.info(f"Successfully parsed {input_df.shape[0]} row(s) with {input_df.shape[1]} columns.")
        return input_df
        
    except Exception as e:
        if "No columns to parse from file" in str(e) or "empty data" in str(e).lower():
            logger.error("Parsing failed. The input might be empty or contain only a header row.", exc_info=False)
            return pd.DataFrame()
        logger.error(f"Pandas parsing failed: {e}", exc_info=True)
        return pd.DataFrame()


@app.route('/', methods=['GET', 'POST'])
def home():
    available_models = []
    specialist_ensembles = {}
    try:
        all_deployed_files = sorted(os.listdir(DEPLOY_DIR))
        for f in all_deployed_files:
            if f.startswith('Specialist_OvR_'):
                if '_SuperSimple' in f:
                    ensemble_type = 'Super Simplified'
                elif '_REDUCED' in f:
                    ensemble_type = 'REDUCED (Balanced)'
                else:
                    ensemble_type = 'Standard'
                
                if ensemble_type not in specialist_ensembles:
                    specialist_ensembles[ensemble_type] = []
                specialist_ensembles[ensemble_type].append(f)
            elif f.endswith(('.pth', '.pth.tar', '.joblib')):
                available_models.append(f)
    except FileNotFoundError:
        flash("Deployment folder not found. Please create the 'DeployModel' folder and add trained models.", "error")

    ensemble_display_names = [f"Specialist Ensemble ({key} Data)" for key in sorted(specialist_ensembles.keys())]

    context = {
        "models": available_models,
        "specialist_ensembles": ensemble_display_names,
        "selected_model": request.form.get('model'),
        "data_format": request.form.get('data_format', 'cicflowmeter'),
        "force_white_space": False
    }

    if request.method == 'POST':
        try:
            selected_model_file = request.form['model']
            pasted_text = request.form.get('pasted_info')
            uploaded_file = request.files.get('file')
            data_format = request.form.get('data_format', 'cicflowmeter')

            context['selected_model'] = selected_model_file
            context['data_format'] = data_format

            data_stream = None

            if uploaded_file and uploaded_file.filename:
                logger.info(f"Processing uploaded file: {uploaded_file.filename}")

                uploaded_file.seek(0)
                
                #Reads the content into a variable.
                file_content = uploaded_file.read()

                #Makes sure the file is not empty before creating a stream.
                if file_content:
                    data_stream = io.BytesIO(file_content)
                else:
                    logger.warning(f"Uploaded file '{uploaded_file.filename}' was empty.")

            #If no data stream was created from a file, it falls back to pasted text.
            #This also handles the case where an empty file was uploaded.
            if data_stream is None and pasted_text and pasted_text.strip():
                logger.info("Processing content from pasted text area.")
                data_stream = io.BytesIO(pasted_text.encode('UTF8'))

            #If no data is available from either source, then inform the user and return.
            if data_stream is None:
                flash("Please upload a non-empty CSV file or paste CSV data to analyze.", "warning")
                return render_template('index.html', **context)
            
            logger.info(f"Received raw content. Format: {data_format}")
            
            is_specialist_run = selected_model_file.startswith('Specialist Ensemble')
            if is_specialist_run:
                ensemble_type_str = selected_model_file.split('(')[1].split(' ')[0]
                data_dir = PROCESSED_DIR_SIMPLIFIED if ensemble_type_str == 'Super' else PROCESSED_DIR
            else:
                data_dir = PROCESSED_DIR_SIMPLIFIED if 'RELABELEDSIMPLE' in selected_model_file or 'SuperSimple' in selected_model_file else PROCESSED_DIR
            
            scaler = joblib.load(os.path.join(data_dir, 'scaler.gz'))
            label_encoder = joblib.load(os.path.join(data_dir, 'label_encoder.gz'))
            metadata = joblib.load(os.path.join(data_dir, 'metadata.gz'))
            
            expected_model_columns = metadata['feature_names']
            
            input_df = parse_input_data(data_stream, data_format, expected_model_columns)

            if input_df.empty:
                flash("The provided data is empty or could not be parsed. Please ensure the data format is correct.", "error")
                return render_template('index.html', **context)

            if is_specialist_run:
                ensemble_type_str = selected_model_file.split('(')[1].split(' ')[0]
                if ensemble_type_str == 'Super':
                    ensemble_key = 'Super Simplified'
                elif ensemble_type_str == 'REDUCED':
                    ensemble_key = 'REDUCED (Balanced)'
                else:
                    ensemble_key = 'Standard'
                
                specialist_group = specialist_ensembles[ensemble_key]
                prediction, confidence, class_probs = run_specialist_prediction(
                    input_df.head(1),
                    specialist_group
                )
            else:
                prediction, confidence, class_probs = run_prediction(
                    os.path.join(DEPLOY_DIR, selected_model_file),
                    input_df.head(1),
                    scaler,
                    label_encoder
                )

            context['prediction'] = prediction
            context['confidence'] = f"{confidence:.2%}"
            context['results'] = sorted(class_probs.items(), key=lambda item: item[1], reverse=True)

            return render_template('index.html', **context)

        except FileNotFoundError as e:
            logger.error(f"Missing a required file: {e}")
            flash(f"Error: A required file is missing. Make sure 'scaler.gz', 'label_encoder.gz', and 'metadata.gz' exist in the correct processed data folder for the selected model.", "error")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}", exc_info=True)
            flash(f"An analysis error occurred: {str(e)}", "error")

    return render_template('index.html', **context)

#Function should be called at startup to load the experimental specialist models (if present).
def load_specialist_ensemble():
    logger.info("Loading Specialist OvR Ensemble models...")
    
    artifacts = {}
    try:
        artifacts['standard'] = {
            'scaler': joblib.load(os.path.join(PROCESSED_DIR, 'scaler.gz')),
            'label_encoder': joblib.load(os.path.join(PROCESSED_DIR, 'label_encoder.gz'))
        }
        logger.info("  - Loaded artifacts for Standard/REDUCED models.")
    except FileNotFoundError:
        logger.warning("Could not find standard processing files. Standard specialist models will not be loaded.")

    try:
        artifacts['simplified'] = {
            'scaler': joblib.load(os.path.join(PROCESSED_DIR_SIMPLIFIED, 'scaler.gz')),
            'label_encoder': joblib.load(os.path.join(PROCESSED_DIR_SIMPLIFIED, 'label_encoder.gz'))
        }
        logger.info("  - Loaded artifacts for Super Simplified models.")
    except FileNotFoundError:
        logger.warning("Could not find super simplified processing files. Super Simplified specialist models will not be loaded.")

    if not os.path.isdir(DEPLOY_DIR):
        return

    deployed_models = os.listdir(DEPLOY_DIR)
    specialist_files = [f for f in deployed_models if f.startswith('Specialist_OvR_')]

    for f in specialist_files:
        try:
            parts = f.replace('_deployable.pth', '').split('_')
            class_name = parts[2]
            model_type = parts[3].split('SuperSimple')[0].split('REDUCED')[0].split('Standard')[0]
            
            if 'SuperSimple' in f:
                artifact_key = 'simplified'
            else:
                artifact_key = 'standard'

            if artifact_key not in artifacts:
                logger.warning(f"Skipping specialist '{f}' because its required artifacts ('{artifact_key}') were not loaded.")
                continue

            scaler = artifacts[artifact_key]['scaler']
            label_encoder = artifacts[artifact_key]['label_encoder']
            input_size = scaler.n_features_in_
            
            model_path = os.path.join(DEPLOY_DIR, f)
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
            model = instantiate_pytorch_model(model_type, input_size, num_classes=2)
            model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
            model.eval()
            
            SPECIALIST_MODELS[f] = {
                'model': model,
                'scaler': scaler,
                'label_encoder': label_encoder,
                'class_name': class_name
            }
            logger.info(f"  - Loaded specialist for: {class_name} (Type: {model_type}, Data: {artifact_key})")
        except Exception as e:
            logger.error(f"Failed to load specialist model {f}. Reason: {e}", exc_info=True)

#Checks to see if at least one processed data file is available, and if the deployment directory exists.
def check_startup_files():
    standard_files_exist = all(os.path.exists(os.path.join(PROCESSED_DIR, f)) for f in ['scaler.gz', 'label_encoder.gz', 'metadata.gz'])
    simplified_files_exist = all(os.path.exists(os.path.join(PROCESSED_DIR_SIMPLIFIED, f)) for f in ['scaler.gz', 'label_encoder.gz', 'metadata.gz'])

    if not standard_files_exist and not simplified_files_exist:
        logger.critical("CRITICAL ERROR: No processed data found!")
        print("\nCould not find required files in 'ProcessedDataset' OR 'ProcessedDatasetSimplified'.")
        print("Please run one of the preprocessing options in the training script first.")
        sys.exit(1)
    
    if standard_files_exist:
        logger.info("Standard processed data files found.")
    if simplified_files_exist:
        logger.info("Super Simplified processed data files found.")

    if not os.path.isdir(DEPLOY_DIR):
        logger.warning("The 'DeployModel' directory does not exist. Creating it now.")
        os.makedirs(DEPLOY_DIR, exist_ok=True)

    logger.info("Required files and directories checked. Loading models...")


if __name__ == '__main__':
    check_startup_files()
    load_specialist_ensemble() 
    app.run(debug=True, port=5001)
