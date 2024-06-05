# 📈 Gold Price Prediction Project

Welcome to the Gold Price Prediction project! This repository contains the code and data used to predict the future adjusted closing price of Gold ETF. Using machine learning techniques, we aim to create an accurate model that helps in forecasting gold prices. 

Check out my [GitHub profile](https://github.com/Armanx200) for more cool projects!

![Gold Price Prediction Plot](https://github.com/Armanx200/Gold-Price-Prediction/blob/main/Figure.png)

## 🚀 Project Overview

This project uses a RandomForestRegressor model to predict the adjusted closing price of Gold ETF based on various features including stock prices and trends from different markets.

### 📝 Dataset

The dataset includes the following columns:
- **Date**: The date of the record.
- **Open, High, Low, Close, Adj Close**: Gold ETF price metrics.
- **Volume**: Trading volume.
- **SP, DJ, EG, EU, OF, OS, SF, USB, PLT, PLD, RHO, USDI, GDX, USO**: Various market indices and their respective metrics.

### 🔍 Data Preprocessing

1. Convert the 'Date' column to datetime format.
2. Drop the 'Date' and 'Adj Close' columns from the features.
3. Standardize the features using `StandardScaler`.

### 🛠️ Model Training and Evaluation

We use a `RandomForestRegressor` model to predict the adjusted closing price of Gold ETF. The model is evaluated using the following metrics:

- **Mean Squared Error (MSE)**
- **Mean Absolute Error (MAE)**
- **R-squared (R²)**

#### 📊 Performance Metrics

| Metric       | Training Set | Testing Set |
|--------------|--------------|-------------|
| **MSE**      | 0.0048       | 0.0769      |
| **MAE**      | 0.0257       | 0.0732      |
| **R²**       | 0.99998      | 0.99975     |

### 📈 Visualization

Here's a visualization of the actual vs predicted values for both the training and testing sets:

![Plot](https://github.com/Armanx200/Gold-Price-Prediction/blob/main/Figure.png)

### 🛠️ How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Armanx200/Gold-Price-Prediction.git
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the code:
   ```bash
   python main.py
   ```

### 🤝 Contributing

Contributions are welcome! Please fork this repository and submit a pull request for any changes you'd like to make.

### 📄 License

This project is licensed under the MIT License.

### 📧 Contact

For any questions or suggestions, feel free to open an issue or contact me at [Armanx200](https://github.com/Armanx200).

---

Happy predicting! 🎉
