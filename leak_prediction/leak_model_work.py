import pickle 
import pandas as pd 

class Leak_Model:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.load_model()

    def load_model(self):
        with open(self.model_path, "rb") as f:
            return pickle.load(f)

    def preprocess_df(self, df: pd.DataFrame) -> pd.DataFrame:
        FEATURES = [
            "Q-E", "Z-V", "Q-V", "N-V", "P-V", "D-E", "O-E", "DB-E", "S-E", "SS-E", "D-S",
            "PH-E", "DB-S", "SS-S", "D-D", "PH-D", "DB-D", "SS-D", "D-R", "PH-R", "DB-R", "SS-R",
            "D-I", "E-I", "S-I", "SS-I", "D-O", "PH-O", "DB-O", "SS-O", "P", "D", "PH", "DB",
            "SS", "RD-DB", "RD-SS"
        ]
        df = df.reindex(columns=FEATURES)
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].astype("category").cat.codes
        return df

    def predict_proba(self, df: pd.DataFrame):
        df = self.preprocess_df(df)
        return self.model.predict_proba(df)[:, 1]
