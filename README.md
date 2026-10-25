# Advanced House Price Predictor (ML) 🏠🤖

An interactive machine learning system built inside WSL2 (Ubuntu) using Python and Scikit-Learn. The application utilizes a Linear Regression model trained on residential property metrics to dynamically estimate house valuations based on custom user inputs.

## Features ✨
- **Supervised ML Pipeline:** Implements Scikit-Learn's `Linear Regression` for multi-variable continuous data tracking.
- **Dynamic User Inputs:** Prompts users directly in the terminal for real-time inference testing (Size, Bedrooms, and Age).
- **Automated Data Visualization:** Generates and exports a continuous mathematical prediction trend graph (`house_price_chart.png`) matching features directly against pricing indices.

## Directory Structure 📁
All assets execute safely out of an independent storage environment (e.g., partitioned virtual mount setups):
- `predict.py` - Core machine learning logic and user interaction interface.
- `house_price_chart.png` - Automatically compiled Matplotlib scatter-plot tracking price scales.
- `ml_env/` - Isolated production dependencies context.

## Local Configuration & Deployment 🏃‍♂️

1. **Clone down this repository:**
   ```bash
   git clone https://github.com
   cd house-price-predictor-ml
   ```

2. **Initialize and trigger your clean pipeline wrapper:**
   ```bash
   python3 -m venv ml_env
   source ml_env/bin/activate
   pip install numpy pandas scikit-learn matplotlib
   ```

3. **Run the predictor:**
   ```bash
   python3 predict.py
   ```
