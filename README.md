# ILFRA - Indian Litigation Finance Risk Assessor

**ILFRA** (LitFin Risk Assessor) is a machine learning-based advisory tool that evaluates risks associated with civil and commercial litigation in India. It is designed to assist litigation funders, lawyers, and businesses by predicting key case trajectories based on historical patterns extracted from public Indian government sources such as the National Judicial Data Grid (NJDG), IBBI CIRP Data, and eCourts judgments.

## What Does It Predict?
The tool evaluates a case and generates three main predictive outputs:
1. **Expected Duration:** The probable length of time the case proceedings will take, complete with confidence intervals (Optimistic, Median, and Pessimistic timelines).
2. **Probability of Favourable Outcome:** A percentage likelihood of a positive result (e.g., winning the lawsuit or reaching a settlement).
3. **Recovery / Realisation Range:** The expected percentage of the claim amount that might be recovered (specifically relevant for Money Recovery and IBC/Insolvency cases).

## System Architecture

The project is structured efficiently into key modules:
- **`src/data_ingestion.py`**: Fetches raw data from NJDG, IBBI folders, and eCourts or synthesizes realistic mock data arrays (approx. 5000 records) to emulate civil/commercial litigations in India for development and testing. 
- **`src/feature_engineering.py`**: Refines the incoming raw structural and proxy data into manageable numerical and categorical features suitable for ML modeling.
- **`src/train.py`**: Trains three internal models sequentially: LightGBM Regressor (Duration), LightGBM Classifier (Favourable Outcome Classification), and LightGBM Regressor (Recovery Quantile percentage).
- **`app/streamlit_app.py`**: Houses the dynamic and responsive frontend built in **Streamlit**. It processes user inputs (such as Case Age, Lawyer Win Rate, Num of Adjournments, Court Type, Sector, and Interim Order status) to output the finalized Risk Assessment dashboard.

---

## How to Properly Run the Tool

To get the app fully functional, follow these simple steps sequentially via the command line or terminal inside the project directory (`c:\Users\baibh\OneDrive\Desktop\ILFRA-Indian-Litigation-Finance-Risk-Assessor`):

### 1. Setup the Environment
Ensure you have the required dependencies downloaded in your Python environment. You can install all dependencies mapped in the requirements file:
```bash
pip install -r requirements.txt
```

### 2. Generate the Raw Data
Run the ingestion script. This generates the `data/raw/` CSV files (like `njdg_synthetic.csv` and `ibbi_synthetic.csv`) used to formulate the training dataset:
```bash
python src/data_ingestion.py
```

### 3. Engineer Features
Transform the raw datasets into ML-readable formats:
```bash
python src/feature_engineering.py
```

### 4. Train the ML Models
Initialize model building. This will output serialized LightGBM models and essential `.csv` configuration insights into your `models/` directory:
```bash
python src/train.py
```

### 5. Launch the Streamlit User Interface
Once the models populate your file structure, boot up the assessor web application:
```bash
streamlit run app/streamlit_app.py
```

The web dashboard should automatically launch in your default browser at `http://localhost:8501`. Navigate through **Case Assessment**, **Model Insights**, and **How It Works** tabs to interact with your predictions!
