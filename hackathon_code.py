#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install opencv-python')


# In[2]:


import os
import re
import nltk
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PIL import Image

import tensorflow as tf
from nltk.corpus import stopwords
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder


# In[3]:


#EDA process
#load csv file
df_projection=pd.read_csv('indiana_projections.csv')
df_report=pd.read_csv('indiana_reports.csv.zip')
#merge them on uid column and we use inner join to keep both text and image in one row
merge_df=pd.merge(df_projection,df_report,on='uid',how='inner')

#clean the data
print(merge_df.isnull().sum().sum())
print(merge_df.isnull().sum().sort_values(ascending=1))
clean_df=merge_df.dropna(subset=['comparison','findings','indication','impression']) #removing the nulls using dropna()
clean_df=clean_df.reset_index(drop=True)                                             #reseting the index
print(f'orginal samples{len(merge_df)}')
print(f'cleaned samples{len(clean_df)}')

# Extract view from the projection data (usually in a column named 'projection' or similar)
# If not explicitly there, you can sometimes find it in the filename or 'indication'
clean_df['is_frontal'] = clean_df['indication'].str.contains('frontal|pa|ap', case=False).astype(int)

#Named Entity Recognition (NER) for Body Parts
def extract_anatomy(text):
    anatomy = ['lung', 'heart', 'cardiac', 'pleural', 'diaphragm', 'spine', 'rib']
    found = [part for part in anatomy if part in text.lower()]
    return ", ".join(found) if found else "General"

clean_df['affected_anatomy'] = clean_df['findings'].apply(extract_anatomy)
# Binary features for common conditions
clean_df['has_cardiomegaly'] = clean_df['findings'].str.contains('cardiomegaly|enlarged heart', case=False).astype(int)
clean_df['has_effusion'] = clean_df['findings'].str.contains('effusion|fluid', case=False).astype(int)

# Create a Risk Level based on keywords
def categorize_risk(text):
    high_risk_words = ['pneumonia', 'edema', 'effusion', 'cardiomegaly', 'nodule', 'mass']
    text = str(text).lower()
    if any(word in text for word in high_risk_words):
        return 'High Risk'
    return 'Normal/Routine'

#word count
clean_df['report_word_count'] = clean_df['findings'].apply(lambda x: len(str(x).split()))
clean_df['risk_level'] = clean_df['impression'].apply(categorize_risk)

# Remove rows where text is just whitespace or too short to be a real report
clean_df=clean_df[clean_df['findings'].str.strip().str.len()>5]
clean_df.duplicated().sum()

#Visualize the Risk Distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=clean_df, x='risk_level', hue='risk_level', palette='viridis', legend=False)
plt.title('Distribution of Patient Risk Levels')
plt.xlabel('Diagnosis Category')
plt.ylabel('Number of Patients')
plt.savefig('risk_distribution.png') 
plt.show()

# Check for Keyword Frequency 
conditions = ['has_cardiomegaly', 'has_effusion']
counts = clean_df[conditions].sum()

plt.figure(figsize=(8, 5))
counts.plot(kind='bar', color='skyblue')
plt.title('Frequency of Specific Medical Findings')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.savefig('finding_counts.png')
plt.show()
clean_df.to_csv('cleaned_multimodel_dat.csv',index=False)

# Path Integrity Check
image_folder = r"C:\Users\SPOORTHI\Downloads\New folder\images_normalized"
clean_df['full_path'] = clean_df['filename'].apply(lambda x: os.path.join(image_folder, x.strip()))
exists = clean_df['full_path'].apply(lambda x: os.path.exists(x))
clean_df = clean_df[exists].reset_index(drop=True) 
print(f"Verified Dataset Size: {len(clean_df)}")



# In[5]:


#NLP
stop_words=set(stopwords.words('english'))
stop_words
def get_medical_cleaning_tools():
    """
    Customizes the stopword list for medical reports.
    Keeps negation words like 'no', 'not', 'without'.
    """
    stop_words = set(stopwords.words('english'))
    # Removing negation words from the stopword list
    # In radiology, "no pneumonia" is the opposite of "pneumonia"
    negations = {'no', 'not', 'none', 'neither', 'never', 'without', 'negative'}
    medical_stop_words = stop_words - negations
    return medical_stop_words
def clean_medical_text(text,medical_stop_words):
    text=str(text) #Ensure text is a string
    text=text.lower()   #converting all string to lower case
    # Removing Punctuation
    # This keeps alphanumeric characters and removes symbols like ! . , ; :
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    # Removing (Medical-Safe) Stop Words
    words = text.split()
    cleaned_words = [w for w in words if w not in medical_stop_words]
    # Join back and remove extra whitespaces
    return " ".join(cleaned_words).strip()   
med_stops = get_medical_cleaning_tools()

# Apply to Master Table
# dataframe is called 'clean_df'
clean_df['findings_proc'] = clean_df['findings'].apply(lambda x: clean_medical_text(x, med_stops))
clean_df['impression_proc'] = clean_df['impression'].apply(lambda x: clean_medical_text(x, med_stops))

#Doctors always want to know if a patient is getting better or worse.
def detect_trend(text):
    if 'worsening' in text or 'increased' in text:
        return 'Worsening'
    if 'improving' in text or 'resolved' in text or 'decreased' in text:
        return 'Improving'
    return 'Stable'

clean_df['condition_trend'] = clean_df['findings_proc'].apply(detect_trend)

# Create a quick Urgency Score
def estimate_urgency(text):
    urgent_terms = ['acute', 'emergency', 'severe', 'worsening', 'critical', 'immediate']
    score = sum(1 for word in urgent_terms if word in text.lower())
    return score

clean_df['urgency_score'] = clean_df['impression_proc'].apply(estimate_urgency)

# Initialize the Vectorizer
# max_features=1000 keeps the top 1000 most important words (prevents memory crashes)
tfidf_vec = TfidfVectorizer(max_features=1000,ngram_range=(1, 2))

# Fit and Transform the Findings
# This creates a 'Sparse Matrix' of numbers
text_features = tfidf_vec.fit_transform(clean_df['findings_proc'])

# Convert to a regular array for fusion later
text_vectors = text_features.toarray()

print(f"Text Feature Shape: {text_vectors.shape}") # Should be (Number of samples, 1000)


# In[6]:


#CNN
# Loading pre-trained model without the final classification layer
base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))

# Create a function to load and process images (TensorFlow compatible)
def load_and_preprocess(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    return preprocess_input(img)

# Build the "Fast Pipeline"
# This prepares images while the CPU/GPU is busy
path_ds = tf.data.Dataset.from_tensor_slices(clean_df['full_path'].values)
image_ds = path_ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
batch_ds = image_ds.batch(64).prefetch(tf.data.AUTOTUNE)

#The Speed Boost(batch processing)
base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
print("Starting High-Speed Extraction...")
image_vectors = base_model.predict(batch_ds)

# Calculate entropy or simple max probability to determine uncertainty
# (This is especially useful once you add your final classification layer)
def get_confidence_status(prediction_prob):
    if prediction_prob > 0.85:
        return "High Confidence"
    elif prediction_prob > 0.60:
        return "Moderate Confidence"
    else:
        return "Low Confidence / Consult Radiologist"

print(f"Extraction complete! Shape: {image_vectors.shape}")

# Save immediately so we NEVER want to run this again
np.save('image_features_final.npy', image_vectors)
print(f"Image Feature Shape: {image_vectors.shape}")



# In[9]:


#Grad-CAM
def make_gradcam_heatmap(img_path, model):
    # Identify the last 4D (convolutional) layer
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if len(layer.output.shape) == 4: 
            last_conv_layer_name = layer.name
            break

    # Create the Grad-CAM Model
    # We use model.input directly without brackets
    grad_model = tf.keras.models.Model(
        inputs=model.input, 
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Preprocess the image
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Record gradients
    with tf.GradientTape() as tape:
        # Pass the img_array DIRECTLY (no brackets) to match 'keras_tensor_155'
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[:, :] 

    # Gradient Calculation
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Compute and Normalize Heatmap
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)

    return heatmap.numpy()
import cv2
def apply_gradcam_overlay(img_path, heatmap, alpha=0.5):
    """
    Final optimized version for clinical dashboards.
    Ensures RGB sync and handles ROI bounding box.
    """
    # 1. Load RAW original for background (ensures standard grayscale look)
    raw_img = cv2.imread(img_path)
    raw_img = cv2.resize(raw_img, (224, 224))
    raw_img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

    # 2. Process Heatmap (Bilinear interpolation is default for smooth look)
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_rescaled = np.uint8(255 * heatmap_resized)

    # 3. Apply JET Colormap and convert to RGB
    jet_heatmap = cv2.applyColorMap(heatmap_rescaled, cv2.COLORMAP_JET)
    jet_heatmap_rgb = cv2.cvtColor(jet_heatmap, cv2.COLOR_BGR2RGB)

    # 4. Superimpose using Weighted Addition (Prevents neon/washed-out colors)
    superimposed_img = cv2.addWeighted(raw_img_rgb, 1.0, jet_heatmap_rgb, alpha, 0)

    # 5. Draw the Bounding Box (ROI)
    # We find where the AI is > 70% focused
    hot_points = np.argwhere(heatmap_resized > 0.5)
    if len(hot_points) > 0:
        y_min, x_min = hot_points.min(axis=0)
        y_max, x_max = hot_points.max(axis=0)
        # Red Box (RGB: 255, 0, 0) with thickness 2
        cv2.rectangle(superimposed_img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

    return superimposed_img

def get_comparison_plot(img_path, heatmap):
# Load original
    orig = cv2.imread(img_path)
    orig = cv2.resize(orig, (224, 224))
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

    # Get Overlay
    overlay = apply_gradcam_overlay(img_path, heatmap)

    # Create a 1x2 or 1x3 montage
    # This makes the dashboard look very professional
    return orig, overlay
def get_hotspot_coords(heatmap, threshold=0.5):
    # Find coordinates where the AI is most focused
    hot_points = np.argwhere(heatmap > threshold)
    if len(hot_points) > 0:
        y_min, x_min = hot_points.min(axis=0)
        y_max, x_max = hot_points.max(axis=0)
    return (x_min, y_min, x_max, y_max)


# In[10]:


#fusion
#assuming image_vectors and text_vectors have the same number of rows
# We used hstack (horizontal stack) to fuse them
fused_features = np.hstack((image_vectors, text_vectors))

print(f"Fused Vector Shape: {fused_features.shape}") 

# elbow plot
distortions = []
K_range = range(1, 10)
for k in K_range:
    kmeds = KMeans(n_clusters=k, random_state=42,n_init=10).fit(fused_features)
    distortions.append(kmeds.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(K_range,distortions, 'bo-')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Error)')
plt.show()

# Creating 5 clusters (Patient Profiles)

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters = kmeans.fit_predict(fused_features)
# Add cluster labels back to your dataframe for analysis
clean_df['patient_cluster'] =clusters

print("Cluster Distribution:")
print(clean_df['patient_cluster'].value_counts())

def calculate_visual_text_match(heatmap, nlp_anatomy):
    # If heatmap is strong in the center (lungs) and text says 'lung'
    # Return a "High Correlation" status
    # This proves the model is not hallucinating
    avg_intensity = np.mean(heatmap)
    return "High Correlation" if avg_intensity > 0.4 else "Review Required"

#CLUSTER VISUALIZATION (t-SNE)

#Standardize the data (This is the most important step for 'clumsy' clusters)
scaler = StandardScaler()
fused_scaled = scaler.fit_transform(fused_features)
print(" Generating t-SNE Plot (This may take a minute)...")

# Use t-SNE  (t-SNE is better at separating overlapping groups)
# Perplexity 30-50 is usually perfect for this dataset size
tsne = TSNE(n_components=2, perplexity=40, random_state=42, init='pca', learning_rate='auto')
tsne_results = tsne.fit_transform(fused_scaled)

# Plot the cleaner clusters
# Instead of clustering by ID, color by the Risk Level you created in EDA
# This proves your AI 'sees' the difference between Normal and High Risk
plt.figure(figsize=(10, 7))
sns.scatterplot(
    x=tsne_results[:, 0], 
    y=tsne_results[:, 1],
    hue=clean_df['risk_level'],  # Use the feature from your EDA!
    palette='RdYlGn_r',          # Red for High Risk, Green for Normal
    alpha=0.7
)
plt.title('Patient Decision Space: How the AI Groups Pathology')
plt.show()


# In[16]:


# ML MODEL(RANDOM FOREST)
#Create a simple label: 1 if "normal" is NOT in the text, 0 if it is
y = clean_df['findings_proc'].apply(lambda x: 0 if 'normal' in x else 1)

# Map the risk levels to numbers: Normal=0, High Risk=1 (or add Mid Risk if available)
# We use LabelEncoder to make it compatible with Random Forest
le = LabelEncoder()
clean_df['risk_label_num'] = le.fit_transform(clean_df['risk_level'])

# Define your target variable 'y' using the refined multi-class labels
y = clean_df['risk_label_num']

# Store the class names so you can show them on the dashboard later
class_names = le.classes_ # e.g., ['High Risk', 'Normal/Routine']

# Spliting the fused data
X_train, X_test, y_train, y_test = train_test_split(fused_features, y, test_size=0.2, random_state=42)

#Initialize with 'balanced' class weights
# This mathematically prioritizes the minority 'Normal' class 
# without creating any synthetic/fake data points.
# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=42) # This is the "Medical-Safe" way to fix bias 
rf_model.fit(X_train, y_train)

# Get the importance of text vs image features
# Assuming first 1280 are image (MobileNet) and rest are text (TF-IDF)
importances = rf_model.feature_importances_
image_feat_importance = np.sum(importances[:1280])
text_feat_importance = np.sum(importances[1280:])

print(f"--- AI Reasoning Breakdown ---")
print(f"Visual Evidence Weight: {image_feat_importance:.2%}")
print(f"Clinical Text Weight:   {text_feat_importance:.2%}")

# Evaluate
# Instead of y_pred = rf_model.predict(X_test)
y_probs = rf_model.predict_proba(X_test)[:, 1] # Probability of being 'Abnormal'
# Raise the threshold to 0.5 (50% confidence required for 'Abnormal')
tuned_predictions = (y_probs >= 0.5).astype(int)

# This will immediately turn those 146 False Positives back into 'Normal'
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, tuned_predictions))

y_pred = rf_model.predict(X_test)


# In[12]:


def test_new_patient_pro(index_num):
    # Retrieve Data
    test_image_path = clean_df['full_path'].iloc[index_num]
    test_report = clean_df['findings'].iloc[index_num]
    cluster_id = clean_df['patient_cluster'].iloc[index_num]

    # Get Risk Probability
    feat = fused_features[index_num].reshape(1, -1)
    # [Normal Probability, Abnormal Probability]
    prob_score = rf_model.predict_proba(feat)[0][1] * 100 

    # Set Cluster Names based on Word Cloud analysis
    cluster_names = {
        0: "Cardiopulmonary Pathology (Effusion/Pneumothorax)", # Based on 'pleural effusion', 'pneumothorax' in Word Cloud
        1: "Normal / Unremarkable Clinical Findings",           # Relocate 'normal'/'clear' here if they dominate another cluster
        2: "Chronic Degenerative / Osseous Changes",            # For spine/bone-related findings
        3: "Acute Infiltrates & Consolidation",                 # For pneumonia-like patterns
        4: "Post-Surgical / Medical Hardware"                   # For stents, pacemakers, or sutures
    }
    assigned_name = cluster_names.get(cluster_id, "Mixed Clinical Findings")

    test_image_path = clean_df['full_path'].iloc[index_num]

    # 1. Generate Heatmap and Overlay
    heatmap = make_gradcam_heatmap(test_image_path, base_model)
    overlay_img = apply_gradcam_overlay(test_image_path, heatmap, alpha=0.75)

    # 2. NEW: Identify and Draw Hotspot Bounding Box
    # We find where the AI is >70% confident
    coords = get_hotspot_coords(heatmap, threshold=0.7) 

    # Draw the box on the overlay image using OpenCV
    if coords:
        x_min, y_min, x_max, y_max = coords
        # Red Box (RGB: 255, 0, 0) with thickness 2
        cv2.rectangle(overlay_img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

    # 3. Visualization
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Original
    original_img_bgr = cv2.imread(test_image_path)
    original_img_rgb = cv2.cvtColor(cv2.resize(original_img_bgr, (224, 224)), cv2.COLOR_BGR2RGB)
    ax[0].imshow(original_img_rgb)
    ax[0].set_title("Input X-Ray")
    ax[0].axis('off')

    # Right: Overlay + Box
    ax[1].imshow(overlay_img)
    ax[1].set_title("AI Localization & ROI Detection")
    ax[1].axis('off')

    plt.show()


    feat = fused_features[index_num].reshape(1, -1)
    pred_label = rf_model.predict(feat)[0]
    cluster_num = clean_df['patient_cluster'].iloc[index_num]

    # Clinical Printout
    print("-" * 50)
    print(f"AI DIAGNOSTIC REPORT")
    print("-" * 50)
    print(f"AI Assigned Cluster: {cluster_num}")
    print(f"Assigned Cluster   : {assigned_name}")
    print(f"AI Predicted the patient is : {'Abnormal' if pred_label == 1 else 'Normal'}")
    print(f"Abnormality Risk   : {prob_score:.2f}%")
    print(f"Original Findings  : {test_report[:250]}...") # Truncated for display
    print("-" * 50)

    # Retrieve pre-calculated NLP features

    urgency = clean_df['urgency_score'].iloc[index_num]
    anatomy = clean_df['affected_anatomy'].iloc[index_num]
    trend = clean_df['condition_trend'].iloc[index_num]

    # Clinical Printout Improvement
    print(f"Primary Anatomy    : {anatomy}")
    print(f"Clinical Trend     : {trend}")
    print(f"Urgency Level      : {urgency}/1")
    print("-" * 50)

    if prob_score > 50:
        print("ACTION REQUIRED: Flagged for high-priority radiologist review.")
    else:
        print("ACTION REQUIRED:  No urgent findings detected.")

# Execute for a sample patient
test_new_patient_pro(25)


# In[ ]:





# In[14]:


# --- 2. CONFUSION MATRIX HEATMAP (Model Performance) ---
print(" Generating Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Abnormal'], 
            yticklabels=['Normal', 'Abnormal'])
plt.title('Confusion Matrix: Prediction Results')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()


# In[37]:


#WORD CLOUD (Cluster Interpretation)
print("📊 Generating Word Clouds...")
def show_cluster_cloud(cluster_num):
    text = " ".join(clean_df[clean_df['patient_cluster'] == cluster_num]['findings_proc'])
    wordcloud = WordCloud(background_color='black',width=800,height=400).generate(text)
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud,interpolation='bilinear')
    plt.title(f'Word Cloud for Cluster {cluster_num}')
    plt.axis('off')
    plt.show()

show_cluster_cloud(0) 


# In[38]:


#CLUSTER VISUALIZATION (t-SNE)
#Standardize the data (This is the most important step for 'clumsy' clusters)
scaler = StandardScaler()
fused_scaled = scaler.fit_transform(fused_features)
print(" Generating t-SNE Plot (This may take a minute)...")

# Use t-SNE  (t-SNE is better at separating overlapping groups)
# Perplexity 30-50 is usually perfect for this dataset size
tsne = TSNE(n_components=2, perplexity=40, random_state=42, init='pca', learning_rate='auto')
tsne_results = tsne.fit_transform(fused_scaled)

# Plot the cleaner clusters
plt.figure(figsize=(10, 7))
scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], 
            c=clean_df['patient_cluster'], 
            cmap='Spectral', # Brighter colors for better separation
            alpha=0.6, 
            edgecolors='w', 
            s=40)

plt.title('Advanced Patient Profile Visualization (t-SNE)', fontsize=15)
plt.colorbar(scatter, label='Cluster ID')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()


# In[ ]:





# In[13]:


from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score
# 1. Generate the detailed Classification Report
# This gives you Precision, Recall, and F1-Score for both Normal (0) and Abnormal (1)
report = classification_report(y_test, y_pred, target_names=['Normal', 'Abnormal'])
print("Detailed Classification Report:")
print(report)

# 2. Creating a clean Summary Table for your Evaluator
metrics_data = {
    "Metric": ["Accuracy", "Precision (Abnormal)", "Recall (Abnormal)", "F1-Score (Abnormal)"],
    "Score (%)": [
        round(accuracy_score(y_test, y_pred) * 100, 2),
        round(precision_score(y_test, y_pred) * 100, 2),
        round(recall_score(y_test, y_pred) * 100, 2),
        round(f1_score(y_test, y_pred) * 100, 2)
    ]
}

metrics_df = pd.DataFrame(metrics_data)
print("\n--- Performance Summary Table ---")
print(metrics_df.to_string(index=False))


# In[ ]:


import joblib

# 1. Save the "Brain" (The Random Forest Model)
joblib.dump(rf_model, 'medical_rf_model.pkl')

# 2. Save the "Translator" (The Label Encoder)
joblib.dump(le, 'label_encoder.pkl')

# 3. Save the "Feature Scaler" (If you used one for t-SNE)
joblib.dump(scaler, 'scaler.pkl')

print("Success! Your AI is now saved in a file and ready for the Dashboard.")


# In[ ]:


import streamlit as st
import joblib
import numpy as np

# --- STEP A: LOAD THE SAVED BRAIN ---
# (Put this at the very top of app.py)
rf_model = joblib.load('medical_rf_model.pkl')
le = joblib.load('label_encoder.pkl')
class_names = le.classes_

# --- STEP B: THE USER INTERFACE ---
st.title("AI Clinical Decision Support")
# ... (Code for uploading image and entering report text) ...

# --- STEP C: THE DASHBOARD INTEGRATION LOGIC ---
# (Paste this inside the 'if st.button("Analyze")' section)
if st.button("Run Diagnostic Analysis"):

    # 1. Calculate the features (Fusion logic)
    # ... (Your code to get 'current_patient_features') ...

    # 2. Run the Model (The logic you asked about)
    pred_class_index = rf_model.predict(current_patient_features)[0]
    pred_probs = rf_model.predict_proba(current_patient_features)[0]

    risk_category = class_names[pred_class_index]
    confidence = pred_probs[pred_class_index] * 100

    # 3. Show the Results (The high-impact UI logic)
    st.subheader("📊 AI Diagnostic Result")

    if risk_category == 'High Risk':
        st.error(f"Status: {risk_category}")
        st.progress(int(confidence)) # Visual bar
        st.write(f"Radiologist Priority: HIGH ({confidence:.2f}%)")
    else:
        st.success(f"Status: {risk_category}")
        st.write(f"Radiologist Priority: NORMAL ({confidence:.2f}%)")

