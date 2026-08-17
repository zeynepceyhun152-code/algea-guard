# 🌊 AlgaeGuard — HAB Early Warning System

> **24-hour-ahead Cyanobacteria (Harmful Algal Bloom) forecasts powered by real-time EPA buoy telemetry.**

---

## 📌 Overview
Harmful Algal Blooms (HABs) can contaminate municipal drinking water and force sudden closures of public beaches within 24 to 48 hours[cite: 1]. While EPA and regional water telemetry buoys continuously stream real-time water quality metrics, raw data only reflects existing conditions[cite: 1]. 

**AlgaeGuard** transforms raw environmental streams into predictive next-day intelligence[cite: 1]. By assessing cyanobacteria trends and 24-hour bloom risks in advance, AlgaeGuard gives water managers and public health authorities actionable lead time to adjust treatment processes or issue public advisories before peak toxicity occurs[cite: 1].

---

## ✨ Features
* **Predictive Risk Scoring:** Forecasts next-day phycocyanin pigment levels (RFU) and estimates bloom alert probabilities[cite: 1].
* **Interactive What-If Simulator:** Allows operators to input live or hypothetical sensor parameters to model tomorrow's water quality risk[cite: 1].
* **Time-Series Telemetry Dashboard:** Displays multi-parameter historical trends for monitored waterbodies[cite: 1].
* **Honest Model Validation:** Evaluated using strict time-ordered holdout splits rather than randomized shuffles to prevent future-data leakage[cite: 3].

---

## 🛠️ Tech Stack
* **Frontend:** HTML5, Modern CSS, JavaScript (ES6+), Chart.js[cite: 1]
* **Backend:** Python, FastAPI, Uvicorn[cite: 2]
* **Machine Learning:** Scikit-Learn (RandomForestRegressor & RandomForestClassifier), Pandas, NumPy, Joblib[cite: 2, 3]

---

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have Python installed on your machine.

### 2. Installation & Setup
Clone the repository and install the required dependencies:

```bash
# Clone the repository
git clone [https://github.com/zeynepceyhun152-code/algea-guard.git](https://github.com/zeynepceyhun152-code/algea-guard.git)
cd algae-guard

# Install Python packages
pip install fastapi uvicorn pandas scikit-learn joblib
