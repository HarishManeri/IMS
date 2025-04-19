import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import io
import base64

# Set page configuration
st.set_page_config(
    page_title="IoT Malware Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for improved UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #0066cc;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #444;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f7ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #0066cc;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #e6ffe6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #00cc66;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #fff9e6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ffcc00;
        margin-bottom: 1rem;
    }
    .danger-box {
        background-color: #ffe6e6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #cc0000;
        margin-bottom: 1rem;
    }
    .stButton button {
        background-color: #0066cc;
        color: white;
        border-radius: 0.3rem;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton button:hover {
        background-color: #004d99;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.markdown("<h1 class='main-header'>Intelligent Malware Detection for IoT Devices</h1>", unsafe_allow_html=True)
st.markdown("<div class='info-box'>This application uses CNN architectures and ensemble machine learning techniques to detect malware in IoT devices. Upload your IoT network traffic or device behavior data to analyze for potential threats.</div>", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.image("https://img.icons8.com/fluency/96/security-shield.png", width=80)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Analysis", "Model Training", "Real-time Detection", "About"])

# Functions for data processing
def preprocess_data(df):
    """Preprocess the uploaded data for model compatibility"""
    # Check for missing values
    if df.isnull().sum().sum() > 0:
        st.warning("Dataset contains missing values. Handling missing values...")
        # Replace missing values with median for numerical columns
        for col in df.select_dtypes(include=np.number).columns:
            df[col] = df[col].fillna(df[col].median())
        
        # Replace missing values with mode for categorical columns
        for col in df.select_dtypes(exclude=np.number).columns:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    # Convert categorical features to numerical if any
    for col in df.select_dtypes(exclude=np.number).columns:
        if col != 'label' and col in df.columns:  # Skip the target column if present
            df[col] = pd.factorize(df[col])[0]
    
    # For demonstration, if 'label' column doesn't exist, create a dummy one
    if 'label' not in df.columns:
        st.info("No 'label' column found. This app assumes binary classification (0: benign, 1: malware).")
    
    return df

def create_cnn_model(input_shape):
    """Create a CNN model for malware detection"""
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=256, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    
    return model

def create_ensemble_model():
    """Create an ensemble of traditional ML models"""
    # Define base models
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    
    # Create voting classifier
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb)],
        voting='soft'
    )
    
    return ensemble

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign', 'Malware'],
                yticklabels=['Benign', 'Malware'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

def plot_roc_curve(y_true, y_pred_proba):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

def get_feature_importance(model, X):
    """Get feature importance from the random forest model"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title('Feature Importances')
        plt.bar(range(X.shape[1]), importances[indices], align='center')
        plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
        plt.tight_layout()
        
        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        return buf
    else:
        return None

# Home page
if page == "Home":
    st.markdown("<h2 class='sub-header'>Welcome to IoT Malware Detection System</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Key Features
        
        - **CNN-based Deep Learning**: Leverages convolutional neural networks to identify complex patterns in IoT device behavior
        - **Ensemble ML Techniques**: Combines multiple machine learning models for higher accuracy and robustness
        - **Interactive Data Analysis**: Visualize and understand your IoT data patterns
        - **Real-time Detection**: Monitor and detect malware threats as they happen
        - **Comprehensive Reporting**: Get detailed insights into detected threats
        
        ### How to Use
        
        1. **Data Analysis**: Upload your IoT device data to visualize and understand patterns
        2. **Model Training**: Train custom ML models on your specific IoT ecosystem
        3. **Real-time Detection**: Deploy the trained models for continuous monitoring
        
        ### Supported Data Formats
        
        - CSV files with network traffic features
        - Device telemetry data
        - System logs and events
        """)
    
    with col2:
        st.image("https://img.icons8.com/color/240/iot-dashboard.png", width=200)
        st.markdown("<div class='warning-box'>⚠️ This application is intended for cybersecurity professionals and researchers. Please ensure you have proper authorization to analyze the IoT devices and networks.</div>", unsafe_allow_html=True)

# Data Analysis page
elif page == "Data Analysis":
    st.markdown("<h2 class='sub-header'>Data Analysis</h2>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload your IoT device data (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        # Load and display the data
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Data loaded successfully! Shape: {df.shape}")
            
            # Data overview tab
            tab1, tab2, tab3, tab4 = st.tabs(["Data Overview", "Feature Analysis", "Correlation Analysis", "Data Preprocessing"])
            
            with tab1:
                st.write("### Data Sample")
                st.dataframe(df.head())
                
                st.write("### Data Information")
                buffer = io.StringIO()
                df.info(buf=buffer)
                s = buffer.getvalue()
                st.text(s)
                
                st.write("### Statistical Summary")
                st.dataframe(df.describe())
                
                # Check for class imbalance if label column exists
                if 'label' in df.columns:
                    st.write("### Class Distribution")
                    class_counts = df['label'].value_counts()
                    fig, ax = plt.subplots(figsize=(8, 4))
                    class_counts.plot(kind='bar', ax=ax)
                    plt.ylabel('Count')
                    plt.title('Class Distribution')
                    st.pyplot(fig)
            
            with tab2:
                st.write("### Feature Distribution")
                
                # Select numerical columns for histograms
                num_cols = df.select_dtypes(include=np.number).columns.tolist()
                if len(num_cols) > 0:
                    selected_cols = st.multiselect("Select features to visualize", num_cols, default=num_cols[:3])
                    
                    if selected_cols:
                        for col in selected_cols:
                            fig, ax = plt.subplots(figsize=(10, 4))
                            sns.histplot(df[col], kde=True, ax=ax)
                            plt.title(f'Distribution of {col}')
                            st.pyplot(fig)
                else:
                    st.warning("No numerical features found in the dataset.")
            
            with tab3:
                st.write("### Correlation Matrix")
                
                # Select numerical columns for correlation
                num_cols = df.select_dtypes(include=np.number).columns.tolist()
                if len(num_cols) > 0:
                    corr = df[num_cols].corr()
                    
                    fig, ax = plt.subplots(figsize=(12, 10))
                    mask = np.triu(np.ones_like(corr, dtype=bool))
                    cmap = sns.diverging_palette(230, 20, as_cmap=True)
                    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                                square=True, linewidths=.5, annot=False, fmt='.2f', ax=ax)
                    plt.title('Correlation Matrix')
                    st.pyplot(fig)
                    
                    st.write("### Top Correlations")
                    # Get the top correlations (excluding self-correlations)
                    corr_pairs = []
                    for i in range(len(num_cols)):
                        for j in range(i+1, len(num_cols)):
                            corr_pairs.append((num_cols[i], num_cols[j], abs(corr.iloc[i, j])))
                    
                    corr_pairs.sort(key=lambda x: x[2], reverse=True)
                    top_corrs = corr_pairs[:10]  # Show top 10 correlations
                    
                    if top_corrs:
                        top_corr_df = pd.DataFrame(top_corrs, columns=['Feature 1', 'Feature 2', 'Correlation (abs)'])
                        st.dataframe(top_corr_df)
                else:
                    st.warning("No numerical features found in the dataset.")
            
            with tab4:
                st.write("### Data Preprocessing")
                st.write("Apply preprocessing steps to prepare your data for model training.")
                
                preprocess_button = st.button("Apply Basic Preprocessing")
                if preprocess_button:
                    processed_df = preprocess_data(df.copy())
                    st.success("Preprocessing completed!")
                    st.dataframe(processed_df.head())
                    
                    # Save preprocessed data to session state
                    st.session_state['processed_data'] = processed_df
                    
                    # Download button for preprocessed data
                    csv = processed_df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="preprocessed_iot_data.csv">Download Preprocessed Data</a>'
                    st.markdown(href, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error loading the file: {e}")
    else:
        st.markdown("<div class='info-box'>📤 Please upload a CSV file containing IoT device data. The file should include features like network traffic patterns, system resource usage, or other relevant IoT device metrics.</div>", unsafe_allow_html=True)
        
        # Sample data info
        st.markdown("""
        ### Expected Data Format
        
        Your CSV file should contain features relevant to IoT device behavior. Common features include:
        
        - Network traffic statistics (packets/sec, bytes/sec)
        - Connection patterns (connection count, unique IPs)
        - System resource usage (CPU, memory, disk I/O)
        - Protocol-specific metrics
        - If labeled data, a 'label' column (0 for benign, 1 for malware)
        
        Don't have a dataset? Use the sample dataset option below:
        """)
        
        if st.button("Load Sample Dataset"):
            # Create a synthetic dataset for demonstration
            np.random.seed(42)
            n_samples = 1000
            
            # Generate synthetic features
            data = {
                'packets_per_sec': np.random.normal(500, 150, n_samples),
                'bytes_per_sec': np.random.normal(5000, 1500, n_samples),
                'conn_count': np.random.poisson(10, n_samples),
                'unique_ips': np.random.poisson(5, n_samples),
                'avg_conn_duration': np.random.exponential(30, n_samples),
                'tcp_percent': np.random.uniform(50, 95, n_samples),
                'udp_percent': np.random.uniform(5, 50, n_samples),
                'dns_queries': np.random.poisson(3, n_samples),
                'cpu_usage': np.random.uniform(10, 90, n_samples),
                'memory_usage': np.random.uniform(20, 80, n_samples)
            }
            
            # Create benign samples (0) and malware samples (1)
            benign_indices = np.random.choice(n_samples, int(n_samples * 0.8), replace=False)
            malware_indices = np.array([i for i in range(n_samples) if i not in benign_indices])
            
            # Set label
            data['label'] = np.zeros(n_samples)
            data['label'][malware_indices] = 1
            
            # For malware samples, adjust some features to create patterns
            for idx in malware_indices:
                # Malware typically has higher network traffic
                data['packets_per_sec'][idx] *= np.random.uniform(1.5, 3.0)
                data['bytes_per_sec'][idx] *= np.random.uniform(1.5, 3.0)
                data['conn_count'][idx] *= np.random.uniform(1.5, 2.5)
                data['unique_ips'][idx] *= np.random.uniform(1.5, 3.0)
                data['dns_queries'][idx] *= np.random.uniform(2.0, 5.0)
                
                # Malware might use more UDP for C&C communications
                data['udp_percent'][idx] = np.random.uniform(60, 90)
                data['tcp_percent'][idx] = 100 - data['udp_percent'][idx]
            
            sample_df = pd.DataFrame(data)
            
            # Save to session state
            st.session_state['processed_data'] = sample_df
            
            st.success("Sample dataset loaded!")
            st.dataframe(sample_df.head())

# Model Training page
elif page == "Model Training":
    st.markdown("<h2 class='sub-header'>Model Training</h2>", unsafe_allow_html=True)
    
    if 'processed_data' not in st.session_state:
        st.warning("No data available for training. Please go to the Data Analysis page first and upload or generate sample data.")
    else:
        df = st.session_state['processed_data']
        
        st.markdown("<div class='info-box'>This section allows you to train both CNN and ensemble machine learning models on your IoT data for malware detection.</div>", unsafe_allow_html=True)
        
        # Model training parameters
        st.write("### Training Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            test_size = st.slider("Test Set Size (%)", min_value=10, max_value=40, value=20, step=5) / 100
            if 'label' in df.columns:
                target_col = 'label'
            else:
                target_col = st.selectbox("Select Target Column", df.columns)
        
        with col2:
            model_type = st.radio("Select Model Type", ["CNN", "Ensemble", "Both"])
            validation = st.checkbox("Use validation set", value=True)
        
        # Model specific parameters
        if model_type == "CNN" or model_type == "Both":
            st.write("### CNN Model Parameters")
            
            cnn_col1, cnn_col2, cnn_col3 = st.columns(3)
            
            with cnn_col1:
                epochs = st.number_input("Epochs", min_value=5, max_value=500, value=50, step=5)
                batch_size = st.number_input("Batch Size", min_value=16, max_value=512, value=64, step=16)
            
            with cnn_col2:
                cnn_optimizer = st.selectbox("Optimizer", ["adam", "rmsprop", "sgd"], index=0)
                early_stopping = st.checkbox("Early Stopping", value=True)
            
            with cnn_col3:
                sequence_length = st.number_input("Sequence Length", min_value=10, max_value=100, value=20, step=5)
        
        if model_type == "Ensemble" or model_type == "Both":
            st.write("### Ensemble Model Parameters")
            
            ens_col1, ens_col2 = st.columns(2)
            
            with ens_col1:
                n_estimators = st.number_input("Number of Estimators", min_value=50, max_value=500, value=100, step=50)
            
            with ens_col2:
                voting = st.radio("Voting Method", ["soft", "hard"])
        
        # Start training
        if st.button("Start Training"):
            with st.spinner("Training models... This may take a while."):
                try:
                    # Split data into features and target
                    X = df.drop(columns=[target_col])
                    y = df[target_col]
                    
                    # Standardize features
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)
                    
                    # Save feature names
                    feature_names = X.columns
                    
                    # Initialize dictionary to store metrics
                    metrics = {}
                    
                    # Train CNN model
                    if model_type == "CNN" or model_type == "Both":
                        # Reshape data for CNN input
                        X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                        X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
                        
                        # Define callbacks
                        callbacks = []
                        if early_stopping:
                            callbacks.append(EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True))
                        
                        # Create and train CNN model
                        cnn_model = create_cnn_model((X_train.shape[1], 1))
                        
                        history = cnn_model.fit(
                            X_train_cnn, y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_split=0.2 if validation else 0.0,
                            callbacks=callbacks,
                            verbose=0
                        )
                        
                        # Evaluate CNN model
                        y_pred_cnn = (cnn_model.predict(X_test_cnn) > 0.5).astype(int)
                        y_pred_proba_cnn = cnn_model.predict(X_test_cnn).ravel()
                        
                        # Calculate metrics
                        metrics['CNN'] = {
                            'accuracy': accuracy_score(y_test, y_pred_cnn),
                            'precision': precision_score(y_test, y_pred_cnn),
                            'recall': recall_score(y_test, y_pred_cnn),
                            'f1': f1_score(y_test, y_pred_cnn),
                            'cm_buf': plot_confusion_matrix(y_test, y_pred_cnn),
                            'roc_buf': plot_roc_curve(y_test, y_pred_proba_cnn),
                            'model': cnn_model,
                            'predictions': y_pred_cnn,
                            'probabilities': y_pred_proba_cnn
                        }
                        
                        # Save training history
                        train_acc = history.history['accuracy']
                        val_acc = history.history['val_accuracy'] if validation else None
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(train_acc, label='Training Accuracy')
                        if val_acc:
                            ax.plot(val_acc, label='Validation Accuracy')
                        ax.set_xlabel('Epoch')
                        ax.set_ylabel('Accuracy')
                        ax.set_title('CNN Training History')
                        ax.legend()
                        
                        buf = io.BytesIO()
                        fig.savefig(buf, format='png')
                        buf.seek(0)
                        metrics['CNN']['history_buf'] = buf
                        
                        # Save model
                        st.session_state['cnn_model'] = cnn_model
                        st.session_state['scaler'] = scaler
                    
                    # Train Ensemble model
                    if model_type == "Ensemble" or model_type == "Both":
                        # Create and train ensemble model
                        ensemble_model = create_ensemble_model()
                        ensemble_model.fit(X_train, y_train)
                        
                        # Evaluate ensemble model
                        y_pred_ens = ensemble_model.predict(X_test)
                        y_pred_proba_ens = ensemble_model.predict_proba(X_test)[:, 1]
                        
                        # Calculate metrics
                        metrics['Ensemble'] = {
                            'accuracy': accuracy_score(y_test, y_pred_ens),
                            'precision': precision_score(y_test, y_pred_ens),
                            'recall': recall_score(y_test, y_pred_ens),
                            'f1': f1_score(y_test, y_pred_ens),
                            'cm_buf': plot_confusion_matrix(y_test, y_pred_ens),
                            'roc_buf': plot_roc_curve(y_test, y_pred_proba_ens),
                            'model': ensemble_model,
                            'predictions': y_pred_ens,
                            'probabilities': y_pred_proba_ens
                        }
                        
                        # Get feature importances
                        feat_imp_buf = get_feature_importance(ensemble_model.named_estimators_['rf'], pd.DataFrame(X_test, columns=feature_names))
                        if feat_imp_buf:
                            metrics['Ensemble']['feat_imp_buf'] = feat_imp_buf
                        
                        # Save model
                        st.session_state['ensemble_model'] = ensemble_model
                    
                    # Display results
                    st.success("Training completed successfully!")
                    
                    # Create tabs for different model results
                    model_tabs = []
                    if model_type == "CNN" or model_type == "Both":
                        model_tabs.append("CNN Results")
                    if model_type == "Ensemble" or model_type == "Both":
                        model_tabs.append("Ensemble Results")
                    if model_type == "Both":
                        model_tabs.append("Models Comparison")
                    
                    tabs = st.tabs(model_tabs)
                    
                    tab_idx = 0
                    if "CNN" in metrics:
                        with tabs[tab_idx]:
                            st.write("### CNN Model Performance")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("#### Metrics")
                                metrics_df = pd.DataFrame({
                                    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                                    'Value': [
                                        f"{metrics['CNN']['accuracy']:.4f}",
                                        f"{metrics['CNN']['precision']:.4f}",
                                        f"{metrics['CNN']['recall']:.4f}",
                                        f"{metrics['CNN']['f1']:.4f}"
                                    ]
                                })
                                st.dataframe(metrics_df)
                                
                                st.write("#### Training History")
                                st.image(metrics['CNN']['history_buf'])
                            
                            with col2:
                                st.write("#### Confusion Matrix")
                                st.image(metrics['CNN']['cm_buf'])
                                
                                st.write("#### ROC Curve")
                                st.image(metrics['CNN']['roc_buf'])
                            
                            st.write("#### Classification Report")
                            report = classification_report(y_test, metrics['CNN']['predictions'], output_dict=True)
                            report_df = pd.DataFrame(report).transpose()
                            st.dataframe(report_df)
                        
                        tab_idx += 1
                    
                    if "Ensemble" in metrics:
                        with tabs[tab_idx]:
                            st.write("### Ensemble Model Performance")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("#### Metrics")
                                metrics_df = pd.DataFrame({
                                    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                                    'Value': [
                                        f"{metrics['Ensemble']['accuracy']:.4f}",
                                        f"{metrics['Ensemble']['precision']:.4f}",
                                        f"{metrics['Ensemble']['recall']:.4f}",
                                        f"{metrics['Ensemble']['f1']:.4f}"
                                    ]
                                })
                                st.dataframe(metrics_df)
                                
                                st.write("#### Feature Importance")
                                if 'feat_imp_buf' in metrics['Ensemble']:
                                    st.image(metrics['Ensemble']['feat_imp_buf'])
                                else:
                                    st.info("Feature importance not available for this model.")
                            
                            with col2:
                                st.write("#### Confusion Matrix")
                                st.image(metrics['Ensemble']['cm_buf'])
                                
                                st.write("#### ROC Curve")
                                st.image(metrics['Ensemble']['roc_buf'])
                            
                            st.write("#### Classification Report")
                            report = classification_report(y_test, metrics['Ensemble']['predictions'], output_dict=True)
                            report_df = pd.DataFrame(report).transpose()
                            st.dataframe(report_df)
                        
                        tab_idx += 1
                    
                    if model_type == "Both":
                        with tabs[tab_idx]:
                            st.write("### Models Comparison")
                            
                            # Compare metrics
                            comparison_df = pd.DataFrame({
                                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                                'CNN': [
                                    f"{metrics['CNN']['accuracy']:.4f}",
                                    f"{metrics['CNN']['precision']:.4f}",
                                    f"{metrics['CNN']['recall']:.4f}",
                                    f"{metrics['CNN']['f1']:.4f}"
                                ],
                                'Ensemble': [
                                    f"{metrics['Ensemble']['accuracy']:.4f}",
                                    f"{metrics['Ensemble']['precision']:.4f}",
                                    f"{metrics['Ensemble']['recall']:.4f}",
                                    f"{metrics['Ensemble']['f1']:.4f}"
                                ]
                            })
                            
                            st.dataframe(comparison_df)
                            
                            # Plot comparison
                            fig, ax = plt.subplots(figsize=(10, 6))
                            x = np.arange(4)
                            width = 0.35
                            
                            cnn_values = [metrics['CNN']['accuracy'], metrics['CNN']['precision'], 
                                          metrics['CNN']['recall'], metrics['CNN']['f1']]
                            ensemble_values = [metrics['Ensemble']['accuracy'], metrics['Ensemble']['precision'], 
                                              metrics['Ensemble']['recall'], metrics['Ensemble']['f1']]
                            
                            rects1 = ax.bar(x - width/2, cnn_values, width, label='CNN')
                            rects2 = ax.bar(x + width/2, ensemble_values, width, label='Ensemble')
                            
                            ax.set_ylabel('Score')
                            ax.set_title('Performance Comparison')
                            ax.set_xticks(x)
                            ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1'])
                            ax.legend()
                            
                            st.pyplot(fig)
                            
                            # Recommendation
                            st.write("### Model Recommendation")
                            
                            # Calculate a weighted score (F1 score has higher weight)
                            cnn_score = 0.2 * metrics['CNN']['accuracy'] + 0.2 * metrics['CNN']['precision'] + \
                                       0.2 * metrics['CNN']['recall'] + 0.4 * metrics['CNN']['f1']
                            
                            ensemble_score = 0.2 * metrics['Ensemble']['accuracy'] + 0.2 * metrics['Ensemble']['precision'] + \
                                            0.2 * metrics['Ensemble']['recall'] + 0.4 * metrics['Ensemble']['f1']
                            
                            if cnn_score > ensemble_score:
                                st.markdown("<div class='success-box'>Based on the performance metrics, we recommend using the <strong>CNN model</strong> for your IoT malware detection task.</div>", unsafe_allow_html=True)
                            else:
                                st.markdown("<div class='success-box'>Based on the performance metrics, we recommend using the <strong>Ensemble model</strong> for your IoT malware detection task.</div>", unsafe_allow_html=True)
                            
                            # Save the best model
                            if cnn_score > ensemble_score:
                                st.session_state['best_model_type'] = 'CNN'
                                st.session_state['best_model'] = metrics['CNN']['model']
                            else:
                                st.session_state['best_model_type'] = 'Ensemble'
                                st.session_state['best_model'] = metrics['Ensemble']['model']
                    
                    # Add option to save models
                    st.write("### Save Models")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if "CNN" in metrics:
                            if st.button("Save CNN Model"):
                                # Save CNN model to pickle
                                model_bytes = io.BytesIO()
                                tf.keras.models.save_model(metrics['CNN']['model'], model_bytes, save_format='h5')
                                model_bytes.seek(0)
                                
                                b64_model = base64.b64encode(model_bytes.read()).decode()
                                href = f'<a href="data:application/octet-stream;base64,{b64_model}" download="cnn_malware_model.h5">Download CNN Model</a>'
                                st.markdown(href, unsafe_allow_html=True)
                    
                    with col2:
                        if "Ensemble" in metrics:
                            if st.button("Save Ensemble Model"):
                                # Save Ensemble model to pickle
                                model_bytes = io.BytesIO()
                                pickle.dump(metrics['Ensemble']['model'], model_bytes)
                                model_bytes.seek(0)
                                
                                b64_model = base64.b64encode(model_bytes.read()).decode()
                                href = f'<a href="data:application/octet-stream;base64,{b64_model}" download="ensemble_malware_model.pkl">Download Ensemble Model</a>'
                                st.markdown(href, unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"Error during model training: {e}")

# Real-time Detection page
elif page == "Real-time Detection":
    st.markdown("<h2 class='sub-header'>Real-time Malware Detection</h2>", unsafe_allow_html=True)
    
    if 'best_model' not in st.session_state and 'cnn_model' not in st.session_state and 'ensemble_model' not in st.session_state:
        st.warning("No trained models available. Please go to the Model Training page first.")
    else:
        st.markdown("<div class='info-box'>This section allows you to perform real-time malware detection on new IoT device data.</div>", unsafe_allow_html=True)
        
        # Determine available models
        available_models = []
        if 'cnn_model' in st.session_state:
            available_models.append("CNN")
        if 'ensemble_model' in st.session_state:
            available_models.append("Ensemble")
        if 'best_model' in st.session_state:
            available_models.append("Best Model")
        
        # Select model to use
        selected_model = st.selectbox("Select Model for Detection", available_models)
        
        # Choose input method
        input_method = st.radio("Input Method", ["Upload CSV File", "Enter Values Manually"])
        
        if input_method == "Upload CSV File":
            uploaded_file = st.file_uploader("Upload IoT data for detection", type=["csv"])
            
            if uploaded_file is not None:
                try:
                    # Load the data
                    df = pd.read_csv(uploaded_file)
                    st.success(f"Data loaded successfully! Shape: {df.shape}")
                    
                    # Display sample
                    st.write("### Data Sample")
                    st.dataframe(df.head())
                    
                    # Preprocess data
                    st.write("### Data Preprocessing")
                    
                    if st.button("Preprocess and Detect"):
                        with st.spinner("Processing data and running detection..."):
                            # Check if label column exists and remove it
                            if 'label' in df.columns:
                                X = df.drop(columns=['label'])
                            else:
                                X = df
                            
                            # Standardize features
                            if 'scaler' in st.session_state:
                                X_scaled = st.session_state['scaler'].transform(X)
                            else:
                                scaler = StandardScaler()
                                X_scaled = scaler.fit_transform(X)
                            
                            # Make predictions
                            if selected_model == "CNN":
                                model = st.session_state['cnn_model']
                                # Reshape for CNN
                                X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
                                y_pred_proba = model.predict(X_reshaped)
                                y_pred = (y_pred_proba > 0.5).astype(int)
                            elif selected_model == "Ensemble":
                                model = st.session_state['ensemble_model']
                                y_pred = model.predict(X_scaled)
                                y_pred_proba = model.predict_proba(X_scaled)[:, 1]
                            else:  # Best Model
                                if st.session_state['best_model_type'] == 'CNN':
                                    model = st.session_state['best_model']
                                    # Reshape for CNN
                                    X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
                                    y_pred_proba = model.predict(X_reshaped)
                                    y_pred = (y_pred_proba > 0.5).astype(int)
                                else:
                                    model = st.session_state['best_model']
                                    y_pred = model.predict(X_scaled)
                                    y_pred_proba = model.predict_proba(X_scaled)[:, 1]
                            
                            # Add predictions to dataframe
                            results_df = df.copy()
                            results_df['prediction'] = y_pred
                            if isinstance(y_pred_proba, np.ndarray) and len(y_pred_proba.shape) == 1:
                                results_df['malware_probability'] = y_pred_proba
                            else:
                                results_df['malware_probability'] = y_pred_proba.flatten()
                            
                            # Display results
                            st.write("### Detection Results")
                            
                            # Summary
                            malware_count = y_pred.sum()
                            benign_count = len(y_pred) - malware_count
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Total Samples", len(y_pred))
                            
                            with col2:
                                st.metric("Malware Detected", malware_count)
                            
                            with col3:
                                st.metric("Benign Samples", benign_count)
                            
                            # Results table
                            st.write("#### Detailed Results")
                            st.dataframe(results_df)
                            
                            # Visualizations
                            st.write("#### Detection Visualizations")
                            
                            viz_col1, viz_col2 = st.columns(2)
                            
                            with viz_col1:
                                # Malware probability distribution
                                fig, ax = plt.subplots(figsize=(10, 6))
                                sns.histplot(results_df['malware_probability'], kde=True, ax=ax)
                                plt.title('Malware Probability Distribution')
                                plt.xlabel('Malware Probability')
                                plt.ylabel('Count')
                                st.pyplot(fig)
                            
                            with viz_col2:
                                # Pie chart of predictions
                                fig, ax = plt.subplots(figsize=(8, 8))
                                ax.pie([benign_count, malware_count], 
                                       labels=['Benign', 'Malware'], 
                                       autopct='%1.1f%%',
                                       colors=['#4CAF50', '#F44336'],
                                       explode=(0, 0.1))
                                plt.title('Detection Results')
                                st.pyplot(fig)
                            
                            # Download results
                            csv = results_df.to_csv(index=False)
                            b64 = base64.b64encode(csv.encode()).decode()
                            href = f'<a href="data:file/csv;base64,{b64}" download="malware_detection_results.csv">Download Results</a>'
                            st.markdown(href, unsafe_allow_html=True)
                            
                            # Alerts for high-confidence malware
                            high_confidence_malware = results_df[(results_df['prediction'] == 1) & 
                                                               (results_df['malware_probability'] > 0.9)]
                            
                            if len(high_confidence_malware) > 0:
                                st.markdown("<div class='danger-box'>⚠️ <strong>High-confidence malware detected!</strong> The system identified samples with over 90% probability of being malware.</div>", unsafe_allow_html=True)
                                st.dataframe(high_confidence_malware)
                
                except Exception as e:
                    st.error(f"Error during detection: {e}")
        
        else:  # Manual Input
            st.write("### Enter IoT Device Features")
            st.markdown("<div class='warning-box'>Please enter values for all features below. Use reasonable values based on your IoT device behavior.</div>", unsafe_allow_html=True)
            
            # Create input fields for common IoT features
            col1, col2 = st.columns(2)
            
            with col1:
                packets_per_sec = st.number_input("Packets per second", min_value=0.0, max_value=10000.0, value=500.0)
                bytes_per_sec = st.number_input("Bytes per second", min_value=0.0, max_value=100000.0, value=5000.0)
                conn_count = st.number_input("Connection count", min_value=0, max_value=1000, value=10)
                unique_ips = st.number_input("Unique IPs", min_value=0, max_value=500, value=5)
                avg_conn_duration = st.number_input("Average connection duration (s)", min_value=0.0, max_value=1000.0, value=30.0)
            
            with col2:
                tcp_percent = st.number_input("TCP traffic percentage", min_value=0.0, max_value=100.0, value=70.0)
                udp_percent = st.number_input("UDP traffic percentage", min_value=0.0, max_value=100.0, value=30.0)
                dns_queries = st.number_input("DNS queries", min_value=0, max_value=500, value=3)
                cpu_usage = st.number_input("CPU usage (%)", min_value=0.0, max_value=100.0, value=50.0)
                memory_usage = st.number_input("Memory usage (%)", min_value=0.0, max_value=100.0, value=60.0)
            
            # Detection button
            if st.button("Run Detection"):
                with st.spinner("Running detection..."):
                    try:
                        # Create a dataframe from the input values
                        input_data = pd.DataFrame({
                            'packets_per_sec': [packets_per_sec],
                            'bytes_per_sec': [bytes_per_sec],
                            'conn_count': [conn_count],
                            'unique_ips': [unique_ips],
                            'avg_conn_duration': [avg_conn_duration],
                            'tcp_percent': [tcp_percent],
                            'udp_percent': [udp_percent],
                            'dns_queries': [dns_queries],
                            'cpu_usage': [cpu_usage],
                            'memory_usage': [memory_usage]
                        })
                        
                        # Standardize features
                        if 'scaler' in st.session_state:
                            X_scaled = st.session_state['scaler'].transform(input_data)
                        else:
                            scaler = StandardScaler()
                            X_scaled = scaler.fit_transform(input_data)
                        
                        # Make predictions
                        if selected_model == "CNN":
                            model = st.session_state['cnn_model']
                            # Reshape for CNN
                            X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
                            y_pred_proba = model.predict(X_reshaped)[0][0]
                            y_pred = 1 if y_pred_proba > 0.5 else 0
                        elif selected_model == "Ensemble":
                            model = st.session_state['ensemble_model']
                            y_pred = model.predict(X_scaled)[0]
                            y_pred_proba = model.predict_proba(X_scaled)[0][1]
                        else:  # Best Model
                            if st.session_state['best_model_type'] == 'CNN':
                                model = st.session_state['best_model']
                                # Reshape for CNN
                                X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
                                y_pred_proba = model.predict(X_reshaped)[0][0]
                                y_pred = 1 if y_pred_proba > 0.5 else 0
                            else:
                                model = st.session_state['best_model']
                                y_pred = model.predict(X_scaled)[0]
                                y_pred_proba = model.predict_proba(X_scaled)[0][1]
                        
                        # Display result
                        st.write("### Detection Result")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if y_pred == 1:
                                st.markdown("<div class='danger-box'><h3>⚠️ Malware Detected!</h3></div>", unsafe_allow_html=True)
                            else:
                                st.markdown("<div class='success-box'><h3>✅ Benign Activity</h3></div>", unsafe_allow_html=True)
                        
                        with col2:
                            # Confidence gauge
                            fig, ax = plt.subplots(figsize=(8, 3))
                            
                            if y_pred == 1:
                                confidence = y_pred_proba
                                color = 'red'
                                label = f"Malware Probability: {confidence:.2%}"
                            else:
                                confidence = 1 - y_pred_proba
                                color = 'green'
                                label = f"Benign Probability: {confidence:.2%}"
                            
                            ax.barh(0, confidence, color=color, height=0.5)
                            ax.barh(0, 1, color='#DDDDDD', height=0.5, alpha=0.3)
                            ax.set_xlim(0, 1)
                            ax.set_yticks([])
                            ax.set_xlabel('Confidence')
                            ax.set_title(label)
                            
                            st.pyplot(fig)
                        
                        # Feature analysis
                        st.write("### Feature Analysis")
                        st.write("Most important features contributing to the detection result:")
                        
                        if selected_model == "Ensemble" or (selected_model == "Best Model" and st.session_state['best_model_type'] == 'Ensemble'):
                            # For ensemble models, use feature importance
                            if selected_model == "Ensemble":
                                importances = st.session_state['ensemble_model'].named_estimators_['rf'].feature_importances_
                            else:
                                importances = st.session_state['best_model'].named_estimators_['rf'].feature_importances_
                            
                            features = list(input_data.columns)
                            
                            # Sort features by importance
                            sorted_idx = np.argsort(importances)[::-1]
                            sorted_features = [features[i] for i in sorted_idx]
                            sorted_importances = importances[sorted_idx]
                            
                            # Plot top 5 important features
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.barh(range(5), sorted_importances[:5], align='center')
                            ax.set_yticks(range(5))
                            ax.set_yticklabels([sorted_features[i] for i in range(5)])
                            ax.set_xlabel('Feature Importance')
                            ax.set_title('Top 5 Important Features')
                            
                            st.pyplot(fig)
                        else:
                            # For CNN models, provide a general analysis
                            st.write("CNN models evaluate complex patterns across all features, but here are some notable observations:")
                            
                            # Check for common malware patterns
                            potential_flags = []
                            
                            if packets_per_sec > 1000:
                                potential_flags.append("High network traffic (packets per second)")
                            
                            if bytes_per_sec > 10000:
                                potential_flags.append("High data transfer rate")
                            
                            if conn_count > 20:
                                potential_flags.append("High number of connections")
                            
                            if unique_ips > 15:
                                potential_flags.append("Connecting to many unique IPs")
                            
                            if udp_percent > 60:
                                potential_flags.append("Unusually high UDP traffic")
                            
                            if dns_queries > 10:
                                potential_flags.append("High number of DNS queries")
                            
                            if potential_flags:
                                st.markdown("<div class='warning-box'>Potential indicators of suspicious activity:</div>", unsafe_allow_html=True)
                                for flag in potential_flags:
                                    st.write(f"- {flag}")
                            else:
                                st.write("No individual feature shows particularly suspicious values.")
                        
                        # Recommendations
                        st.write("### Recommendations")
                        
                        if y_pred == 1:
                            st.markdown("<div class='danger-box'><strong>Recommended Actions:</strong></div>", unsafe_allow_html=True)
                            st.write("1. Isolate the device from the network immediately")
                            st.write("2. Capture and preserve network traffic for forensic analysis")
                            st.write("3. Initiate a full system scan")
                            st.write("4. Check for unauthorized configuration changes")
                            st.write("5. Update firmware and apply security patches")
                        else:
                            st.markdown("<div class='success-box'><strong>Recommended Actions:</strong></div>", unsafe_allow_html=True)
                            st.write("1. Continue monitoring the device behavior")
                            st.write("2. Ensure regular security updates are applied")
                            st.write("3. Maintain baseline metrics for future comparison")
                    
                    except Exception as e:
                        st.error(f"Error during detection: {e}")

# About page
else:  # About page
    st.markdown("<h2 class='sub-header'>About</h2>", unsafe_allow_html=True)
    
    st.write("### Intelligent Malware Detection for IoT Devices")
    
    st.markdown("""
    This application leverages state-of-the-art deep learning and machine learning techniques to detect malware in IoT devices.

    #### Technologies Used

    - **Deep Learning**: Convolutional Neural Networks (CNNs) for complex pattern recognition
    - **Ensemble Learning**: Combining multiple machine learning models for improved accuracy
    - **Streamlit**: Interactive web application framework
    - **TensorFlow/Keras**: Deep learning framework
    - **scikit-learn**: Traditional machine learning algorithms

    #### How It Works

    1. **Data Collection**: The system analyzes network traffic patterns, system resource usage, and other behavioral indicators from IoT devices.
    
    2. **Feature Engineering**: Raw data is transformed into meaningful features that can be used to distinguish between normal and malicious behavior.
    
    3. **Model Training**: Both CNN and ensemble machine learning models are trained on labeled data to learn patterns of malware.
    
    4. **Real-time Detection**: Trained models are used to analyze new data and detect potential malware threats.
    
    5. **Visualization and Reporting**: Results are presented with interactive visualizations and actionable recommendations.

    #### About IoT Malware

    IoT devices are increasingly targeted by malware due to their often limited security features, widespread deployment, and constant connectivity. Common types of IoT malware include:
    
    - **Botnets**: Networks of infected devices controlled remotely for DDoS attacks
    - **Cryptominers**: Malware that uses device resources to mine cryptocurrency
    - **Data Exfiltration Tools**: Malware designed to steal sensitive data
    - **Ransomware**: Malware that locks device functionality until a ransom is paid

    #### Research Sources

    This application is based on research in IoT security, machine learning, and network forensics. For more information, see:
    
    - IEEE papers on IoT security
    - Recent developments in deep learning for malware detection
    - Industry best practices for IoT device monitoring
    """)
    
    st.markdown("<div class='info-box'><strong>Note:</strong> This application is intended for educational and research purposes. Always follow proper security protocols and guidelines when handling potential malware.</div>", unsafe_allow_html=True)
    
    # References and additional resources
    st.write("### Additional Resources")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Security Best Practices**
        
        - Regular firmware updates
        - Network segmentation for IoT devices
        - Strong, unique passwords
        - Disable unnecessary services
        - Use encrypted communications
        """)
    
    with col2:
        st.markdown("""
        **Further Reading**
        
        - NIST Guidelines for IoT Security
        - OWASP IoT Security Project
        - ENISA Baseline Security Recommendations for IoT
        - IoT Security Foundation resources
        """)
