<div align="center">

# 🔮 Time-Series Fault Prediction with LSTM
### Chapter 10: Deep Learning for Predictive Maintenance

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange?logo=tensorflow&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Sklearn-Data_Science-yellow?logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Educational_Demo-green)

[View Architecture](#-neural-network-architecture) • [Understanding LSTMs](#-why-lstm-for-time-series)

</div>

---

## 📖 Overview

This project demonstrates how to detect evolving faults in machinery using **Deep Learning**. Unlike traditional threshold-based systems that only react when a value gets "too high," this model uses **Long Short-Term Memory (LSTM)** networks to analyze the *shape* and *trend* of data over time.

The script generates synthetic sensor data simulating two states:
1.  **🟢 Normal Operation:** Stable oscillation (sine wave) with random sensor noise.
2.  **🔴 Fault Progression:** The same oscillation but with a subtle, drifting trend indicating wear.

The goal is to predict the probability of failure based on a rolling window of the last 20 timestamps.

---

## ⚡ Key Features

* **🌊 Synthetic Data Engine:** Automatically generates realistic sensor data with noise and drifting trends.
* **🧠 Recurrent Neural Network:** Implements an LSTM layer to capture temporal dependencies (history) in the data.
* **📉 Statistical Preprocessing:** Uses `StandardScaler` to normalize time-series sequences for optimal neural network convergence.
* **📊 Visualization Suite:**
    * **Fault Progression:** Visualizes how the model's "risk score" increases as the fault develops.
    * **Confusion Matrix:** Evaluates classification accuracy.
    * **Probability Distribution:** Shows the model's confidence in its predictions.

---

## 🏗 Neural Network Architecture

The model is built using TensorFlow/Keras with a specific focus on sequence processing:

```mermaid
graph LR
    A[Input Sequence<br/>(20 Time Steps)] --> B[LSTM Layer<br/>(50 Units, ReLU)]
    B --> C[Dense Layer<br/>(25 Units, ReLU)]
    C --> D[Output Layer<br/>(Sigmoid)]
    D --> E[Fault Probability<br/>(0.0 to 1.0)]
