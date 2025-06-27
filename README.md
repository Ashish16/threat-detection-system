# AI-Driven Threat Detection System

This project is an AI-based network threat detection tool that uses machine learning (Isolation Forest) to analyze network traffic and identify anomalies. It demonstrates how real-time threat detection can be built using synthetic traffic data and anomaly detection algorithms.

---

## 🚀 Features

- Anomaly detection using Isolation Forest
- Synthetic network traffic data generation
- Visual interface for evaluation
- Modular components for future enhancement

---

## 📁 Project Structure
ai-threat-detector/
│
├── app/                       # Core app logic and routes
│   ├── evaluation_app.py      # Evaluation logic for model
│   ├── threatdetect.py        # Main logic 
│   ├── about.py               # About the project
│   └── templates/
│       └── about.html         # HTML templates 
│
├── data/                      # Input / generated data
│   ├── data.csv
│   ├── synthetic_network_data.csv
│
├── scripts/                   # helper scripts
│   ├── generate_syn_data.py
│   ├── generate_syn_traffic.py
│   └── save_data.py
│
├── main.py                    # Streamlit app entry
├── requirements.txt           # Python dependencies
├── README.md                  # Project overview and instructions

## Install the dependencies
pip install -r requirements.txt

## Run the project
streamlit run app.py