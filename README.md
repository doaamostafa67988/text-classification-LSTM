# 🎭 Text Classification: Emotion Recognition Pipeline

An end-to-end Natural Language Processing (NLP) pipeline built with **TensorFlow** and **Keras** to categorize text sequences into human emotional profiles. The model is trained to process unformatted conversational data and predict one of six core underlying human emotions.

---

## 🎯 Project Objective
The main goal of this project is to develop an automated classification pipeline capable of analyzing text strings (such as social media entries, customer feedback, or chatbot user queries) and mapping them accurately to behavioral emotional profiles.

---

## 📊 Dataset Structure
The system ingests text files separated by delimiters (`train.txt`, `val.txt`, `test.txt`) containing structured mappings of raw text strings paired with their matching labels.

### 🏷️ Target Classes
The models learn to handle multi-class textual boundaries distributed across **6 primary emotion nodes**:
* **😊 Joy** (Highest represented class in dataset)
* **😢 Sadness**
* **😠 Anger**
* **😨 Fear**
* **❤️ Love**
* **😮 Surprise**

---

## 🛠️ Technology Stack & Dependencies

* **Runtime Environment:** Google Colab / Kaggle Notebook with TPU acceleration enablement.
* **Core Framework:** TensorFlow Engine (`v2.13.0`).
* **High-level Neural Network API:** Keras.
* **Data Structures & Processing:** Pandas, NumPy.
* **Data Visualization Matrix:** Matplotlib, Seaborn.

---

## 🧠 Approach & Model Pipeline

### 1. Data Ingestion & Extraction
* Unpacking raw localized compressed assets (`data.rar`) into modular text directories.
* Structured assembly of DataFrames mapping structural feature items (`sentence`, `label`) across standardized dataset splits.

### 2. Exploratory Data Analysis (EDA)
* Automated distribution counts tracking target categorical variables.
* Pie chart representation using HUSL color configurations tracking training balance across emotional splits.

### 3. Text Preprocessing & Tokenization *(In Progress)*
* Converting raw sentences into tokenized arrays ready for semantic encoding.
* Sequence padding alignment ensuring unified tensor size entry across data lanes.

### 4. Neural Network Core Architecture *(In Progress)*
* Embedding layers optimizing textual contextual semantic relationships.
* Sequential deep network structures compiling classifications maps across high-frequency categories.

---

## 🚀 Getting Started

### Prerequisites
Ensure your localized shell environment or cloud runner runs standard Python 3 and has the correct version of TensorFlow configured:
```bash
pip install tensorflow==2.13.0 pandas numpy matplotlib seaborn
```
