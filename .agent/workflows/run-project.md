---
description: How to run the StellarBasin ML Model Visualizer project
---

# Running StellarBasin ML Model Visualizer

This project has two parts that need to run simultaneously:

## 1. Backend (Flask Server)

First, activate the Python virtual environment and start the Flask backend:

```bash
cd backend
..\venv\Scripts\Activate.ps1
python app.py
```

The backend will run on `http://localhost:5000`

## 2. Frontend (React App)

In a separate terminal, start the React development server:

```bash
cd frontend
npm start
```

The frontend will run on `http://localhost:3000` and automatically open in your browser.

## What the app does:
- Upload trained ML models (.pkl files)
- Visualizes Decision Trees, KNN, Linear Regression, and Logistic Regression models
- Displays interactive Plotly charts or Matplotlib images
- Shows model reports and summaries

## Troubleshooting:
- If Python dependencies are missing, run: `pip install -r requirements.txt` in the backend folder
- If Node modules are missing, run: `npm install` in the frontend folder
- Make sure both servers are running simultaneously for the app to work
