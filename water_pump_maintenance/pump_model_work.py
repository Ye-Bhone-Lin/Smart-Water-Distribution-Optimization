import pandas as pd
import pickle
import numpy as np

class PumpMaintenanceModel:
    def __init__(self, model_path: str):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        self.class_labels = self.model.classes_

    def predict_full(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            raise ValueError("No numeric columns found in uploaded CSV!")

        X_array = numeric_df.to_numpy()
        predictions = self.model.predict(X_array)
        probas = self.model.predict_proba(X_array)

        df_copy = df.copy()
        df_copy["Maintenance_Status"] = predictions
        for i, cls in enumerate(self.class_labels):
            df_copy[f"Prob_{cls}"] = probas[:, i]
        return df_copy
