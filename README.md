# AI-Based Queue & Crowd Load Prediction System

## Abstract
This project presents an AI-Based Queue & Crowd Load Prediction System designed to monitor, analyze, and predict crowd density and queue lengths in real time. The system utilizes computer vision techniques to detect and count people from video feeds and applies machine learning models to forecast future crowd conditions. The objective is to improve crowd management, reduce waiting times, and enhance safety in public spaces such as railway stations, malls, and events. The methodology combines image processing, data analysis, and time-series forecasting. Results demonstrate that the system can effectively estimate crowd size and provide early warnings of overcrowding. This solution can be implemented in smart city infrastructure for efficient resource allocation and improved user experience.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Literature Review](#literature-review)
3. [Methodology](#methodology)
4. [Implementation & Tech Stack](#implementation--tech-stack)
5. [Setup & Installation](#setup--installation)
6. [Model Training](#model-training)
7. [Running the Application](#running-the-application)
8. [Deployment Guide](#deployment-guide)
9. [Results & Discussion](#results-and-discussion)
10. [Limitations & Future Scope](#limitation--future-scope)
11. [Conclusion](#conculusion)
12. [References](#references)

---

## Introduction
Crowd management is a critical challenge in public places such as transportation hubs, shopping centers, and large events. Uncontrolled crowd growth can lead to long queues, inefficiency, and even safety hazards. Traditional methods rely on manual monitoring, which is inefficient and prone to error.

The motivation behind this project is to develop an intelligent system that can automatically monitor and predict crowd behavior using artificial intelligence. The objective is to provide real-time crowd analysis and future predictions to help authorities take proactive decisions such as opening additional counters or redirecting people.

## Literature Review
Existing systems for crowd monitoring mainly rely on CCTV surveillance and manual observation. Recent research uses computer vision techniques like object detection and tracking to estimate crowd density. Algorithms such as YOLO (You Only Look Once) are widely used for real-time person detection.

In addition, time-series forecasting models such as ARIMA and LSTM have been applied to predict crowd trends based on historical data. However, many systems lack integration between real-time detection and predictive analytics. This project aims to combine both approaches into a unified system.

## Methodology
The system captures video input through cameras and processes it using computer vision techniques to detect and count people. The extracted data is stored over time and used to train machine learning models. A Random Forest prediction model forecasts future crowd levels based on historical patterns. The results are displayed on a dynamic dashboard with alerts for overcrowding situations.

---

## Implementation & Tech Stack
- **Programming Language**: Python 3.13+
- **Machine Learning**: Scikit-learn, Joblib
- **Data Engineering**: Pandas, NumPy
- **Visualizations**: Matplotlib, Seaborn
- **Dashboard UI**: Streamlit (Premium dark-theme layout)

---

## Setup & Installation

### 1. Clone the repository
Navigate to your workspace directory.

### 2. Install Dependencies
Install the required packages listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## Model Training
A dedicated model training pipeline is provided in `train.py`. The model trains a `RandomForestRegressor` using the historical dataset `crowd_data.csv`.

To train/retrain the predictive model:
```bash
python train.py
```
This script will output model validation metrics (MAE, RMSE, R² score) and save the model to `queue_model.pkl` along with metadata in `model_metrics.pkl`.

---

## Running the Application
To launch the interactive dashboard locally:
```bash
streamlit run streamlit_app.py
```
Open the URL printed in your terminal (usually `http://localhost:8501`) in your browser to interact with the system.

---

## Deployment Guide

### Option 1: Streamlit Community Cloud (Recommended & Easiest)
1. Push your repository to GitHub.
2. Sign in to [Streamlit Share](https://share.streamlit.io/).
3. Click **New app**, select your repository, branch, and specify `streamlit_app.py` as the entrypoint.
4. Click **Deploy**. Streamlit will automatically read `requirements.txt` and launch the application!

### Option 2: Hugging Face Spaces (Gradio/Streamlit)
1. Create a new Space on [Hugging Face](https://huggingface.co/spaces) and choose the **Streamlit** SDK.
2. Git clone the space repo, copy all files (including `crowd_data.csv` and `queue_model.pkl`), and push them to Hugging Face.
3. Hugging Face will build and serve your app.

### Option 3: Docker Deployment
A sample `Dockerfile` for containerization:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```
Build and run:
```bash
docker build -t crowd-prediction-system .
docker run -p 8501:8501 crowd-prediction-system
```

---

## Results and Discussion
The system successfully detects and counts people in video frames with good accuracy. Graphs of crowd count versus time are generated for visualization. The prediction model provides an estimate of future crowd levels, helping in identifying peak hours.

The results show that the system can be used for real-time monitoring and early warning of overcrowding. Accuracy depends on video quality and model training data.

## Limitation
- Accuracy may decrease in highly dense crowds.
- Requires good quality camera input.
- Privacy concerns due to surveillance.
- Prediction accuracy depends on data availability.

## Future Scope
- Integration with multiple cameras.
- Use of advanced deep learning models for better accuracy.
- Mobile app for real-time alerts.
- Integration with smart city infrastructure.
- Crowd heatmap visualization.

## Conculusion  
The AI-Based Queue & Crowd Load Prediction System provides an efficient solution for real-time crowd monitoring and prediction. By combining computer vision and machine learning techniques, the system helps improve safety, reduce waiting time, and optimize resource management. It has strong potential for real-world applications in smart cities and public infrastructure.

## References
[1] Redmon, J., "YOLO: Real-Time Object Detection," 2016.  
[2] Hochreiter, S., "Long Short-Term Memory," Neural Computation, 1997.  
[3] https://opencv.org  
[4] https://tensorflow.org  
