# Water Pump Maintenance Prediction

This feature demonstrates a machine learning workflow for predicting water pump maintenance needs using sensor data. It leverages Python, pandas, scikit-learn, and imbalanced-learn for data processing, modeling, and evaluation.

## Data Understanding 

- **Motor Casing Vibration:** sensor_00, sensor_13, sensor_18  
- **Motor Frequency (Hz):** sensor_01 – sensor_03  
- **Motor Current (A):** sensor_05  
- **Motor Active Power (kW):** sensor_06  
- **Motor Apparent Power (kVA):** sensor_07  
- **Motor Reactive Power (kVAR):** sensor_08  
- **Phase Currents (A/B/C):** sensor_10 – sensor_12  
- **Phase Voltages (AB, BC, CA):** sensor_14, sensor_16, sensor_17  
- *(sensor_15 likely phase average but not recorded)*  
- Remaining sensors (~19–51) may capture gearbox and pump values (temperature, vibration, flow, etc.).

## Workflow Overview

![alt text](workflow/water_pump.svg)
*Figure 1 Workflow for Water Pump Maintenance*

1. **Data Loading & Exploration**
	- Loads sensor data from `data/sensor.csv`.
	- Explores data shape, missing values, and basic info.

2. **Preprocessing**
	- Drops unnecessary columns.
	- Identifies numeric features.
	- Handles missing values and scales features.

3. **Target Preparation**
	- Extracts target variable for prediction.
	- Handles class imbalance using SMOTE oversampling.

4. **Model Training**
	- Splits data into train/test sets.
	- Trains a Random Forest classifier.

5. **Evaluation**
	- Prints classification report and accuracy metrics.

6. **Model Saving**
	- Saves the trained model as `rf_model.pkl` for future use.

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- imbalanced-learn
- matplotlib
- seaborn
- joblib

Install dependencies using pip or your preferred environment manager.

## Usage

1. Place your sensor data in `data/sensor.csv`.
2. Run the notebook `water_pump_maint.ipynb` step by step.
3. The trained model will be saved as `rf_model.pkl`.

## File Structure

- `water_pump_maint.ipynb`: Main notebook for data analysis and modeling.
- `data/sensor.csv`: Input sensor data.
- `rf_model.pkl`: Output trained model.

## Result

               precision    recall  f1-score  support

    BROKEN       1.00      1.00      1.00     51460
    NORMAL       1.00      1.00      1.00     51459
    RECOVERING   1.00      1.00      1.00     3619

    accuracy                         1.00     106538
    macro avg    1.00      1.00      1.00     106538
    weighted avg 1.00      1.00      1.00     106538

References:

https://www.kaggle.com/code/winternguyen/water-pump-maintenance-shutdown-prediction