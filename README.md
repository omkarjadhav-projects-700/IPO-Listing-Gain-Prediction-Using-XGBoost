# 🚀 IPO Listing Gain Prediction Using XGBoost

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-green?logo=xgboost)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-teal?logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Supported-blue?logo=docker)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📖 Overview

This project is a comprehensive end-to-end machine learning pipeline designed to predict the **listing gain** of Initial Public Offerings (IPOs) on Indian stock exchanges (NSE/BSE). The system uses **XGBoost Regressor** to forecast percentage gains based on a curated dataset encompassing subscription figures, financial ratios, issue structure, and market indicators.

The project demonstrates expertise in:
- Data preprocessing and feature engineering
- Machine learning model training and evaluation
- REST API development with FastAPI
- Containerization with Docker
- Production-ready model deployment

## 🎯 Project Motivation

The Indian IPO market has experienced significant growth. For retail investors, identifying IPOs with high listing gain potential is valuable yet complex. This project aims to:
- Analyze factors driving IPO performance in the Indian context
- Build a data-driven prediction tool for estimated short-term returns
- Provide a scalable, containerized solution for real-world deployment

## 📁 Project Structure

```
.
├── api/                          # FastAPI application
│   └── app.py                   # REST API endpoints
├── src/                         # Core ML pipeline
│   ├── config.py                # Configuration settings
│   ├── preprocess.py            # Data preprocessing functions
│   ├── train.py                 # Model training script
│   ├── evaluate.py              # Model evaluation metrics
│   └── inference.py             # Prediction interface
├── models/                      # Trained model artifacts
│   └── xgb_model.json          # Serialized XGBoost model
├── data/                        # Data directory
│   └── ipo_data_output.csv     # IPO dataset
├── Data Creation/               # Data collection & preparation
│   ├── Sample Dataset.csv       # Sample data
│   ├── Imported from Kaggle/    # External datasets
│   ├── IPO TEXTS/              # IPO prospectus texts
│   ├── phase-1 - (2025-2020)/  # Historical phase data
│   └── Raw Finalized Data/     # Finalized raw data
├── requirements.txt             # Python dependencies
├── dockerfile                   # Docker container specification
├── docker-compose.yml           # Docker Compose configuration
├── main.py                      # Main entry point
├── Rough.py & Rough.ipynb      # Exploratory notebooks
└── README.md                    # This file
```

## 🗃️ Dataset & Features

The dataset includes IPO data from multiple sources with features categorized as:

- **Issue Details:** `issue_size`, `price_band`, `lot_size`, `fresh_issue_shares`, `offer_for_sale`
- **Subscription Data:** `QIB_subscription`, `NII_subscription`, `RII_subscription`, `Total_subscription`
- **Financial Metrics:** `offer_price`, `list_price`, `EPS`, `P/E Ratio`, `ROE`, `ROCE`, `Debt/Equity`
- **Target Variable:** `listing_gains(%)` - Percentage gain/loss from issue price to listing price

**Data Sources:**
- [Chittorgarh](https://www.chittorgarh.com/)
- [Moneycontrol](https://www.moneycontrol.com/)
- [NSE India](https://www.nseindia.com/)
- Kaggle datasets

## ⚙️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.10+ |
| **ML Framework** | XGBoost |
| **Data Processing** | Pandas, NumPy, Scikit-learn |
| **API** | FastAPI, Uvicorn |
| **Containerization** | Docker, Docker Compose |
| **Visualization** | Matplotlib, Seaborn |
| **Version Control** | Git, GitHub |

## 🔄 Pipeline Architecture

### 1. Data Preprocessing (`src/preprocess.py`)
- Load and clean IPO dataset
- Handle missing values and outliers
- Standardize numerical features using `StandardScaler`
- Remove unnecessary columns (e.g., dates, company names)
- Train-test split (80-20)

### 2. Model Training (`src/train.py`)
- **Algorithm:** XGBoost Regressor
- **Hyperparameters:**
  - `n_estimators`: 200
  - `learning_rate`: 0.05
  - `max_depth`: 6
  - `subsample`: 0.9
  - `colsample_bytree`: 0.8
- Model saved as `xgb_model.json`

### 3. Model Evaluation (`src/evaluate.py`)
- **Metrics:**
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - R² Score
- K-Fold cross-validation support

### 4. Inference (`src/inference.py`)
- `Prediction_Model` class for making predictions
- Auto-handling of missing features (zero-fill)
- Returns integer prediction value

### 5. REST API (`api/app.py`)
- FastAPI application
- Endpoint-based prediction service
- Ready for deployment

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- pip or conda
- Docker & Docker Compose (optional, for containerized deployment)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/omkarjadhav-projects-700/IPO-Listing-Gain-Prediction-Using-XGBoost.git
   cd IPO-Listing-Gain-Prediction-Using-XGBoost
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### Option 1: Train Model Locally
```bash
python src/train.py
```

#### Option 2: Evaluate Model
```bash
python src/evaluate.py
```

#### Option 3: Run FastAPI Server
```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

#### Option 4: Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up --build

# Or manually with Docker
docker build -t ipo-prediction:latest .
docker run -p 8000:8000 ipo-prediction:latest
```

## 📊 API Endpoints

Once the FastAPI server is running, access:

- **Root Endpoint:** `GET /` - Returns welcome message
- **Health Check:** `GET /` - Verify server is running
- **Prediction:** `POST /predict` - Submit IPO features for prediction

### Example Request:
```json
{
  "issue_size": 500,
  "QIB_subscription": 2.5,
  "NII_subscription": 1.8,
  "RII_subscription": 3.2,
  "Total_subscription": 2.8,
  "offer_price": 100,
  "list_price": 125
}
```

## 🔍 Model Performance

The XGBoost model achieves strong predictive performance on the test set:
- **MAE:** [To be updated after training]
- **RMSE:** [To be updated after training]
- **R² Score:** [To be updated after training]

## 📚 References

- [Analytics Vidhya - Ultimate Guide to Boosting Algorithms](https://www.analyticsvidhya.com/blog/2022/12/ultimate-guide-to-boosting-algorithms/)
- [Analytics Vidhya - XGBoost Math Guide](https://www.analyticsvidhya.com/blog/2018/09/an-end-to-end-guide-to-understand-the-math-behind-xgboost/)
- [Analytics Vidhya - XGBoost Parameter Tuning](https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

## 🛠️ Development Notes

- **Exploratory Analysis:** See `Rough.ipynb` and `Rough.py` for data exploration
- **Data Management:** Raw data in `Data Creation/` directory; processed data in `data/`
- **Model Artifacts:** Trained model stored in `models/xgb_model.json`
- **Configuration:** Update `src/config.py` for model and data paths

## 👨‍💻 Developer

**Omkar Jadhav**
- GitHub: [@omkarjadhav-projects-700](https://github.com/omkarjadhav-projects-700)
- LinkedIn: [Omkar Jadhav](https://www.linkedin.com/in/omkar-jadhav-637807278/)
- Email: omkarjadhav3540@gmail.com

## 📜 License

This project is licensed under the **MIT License** - see LICENSE file for details.

## 🙏 Acknowledgments

- Indian stock market data sourced from NSE, Moneycontrol, and Chittorgarh
- XGBoost library and community
- FastAPI framework for rapid API development
- Research on IPO underpricing and market efficiency

---

**⭐ If you find this project helpful, please consider starring it on GitHub!**
