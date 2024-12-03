from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pickle
import numpy as np
import pandas as pd
from fastapi.encoders import jsonable_encoder


with open('ridge_model.pkl', 'rb') as f:
    model_params = pickle.load(f)

model = model_params['model']
scaler = model_params['scaler']
columns_for_ohe = model_params['columns_for_ohe']
ohe = model_params['ohe']

app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


def pydantic_model_to_df(model_instance):
    return pd.DataFrame([jsonable_encoder(model_instance)])

def extract_torque(torque_string):
    if pd.isna(torque_string):
        return pd.NA, pd.NA
    torque_string = torque_string.lower().replace(' ', '')

    try:
        if '@' in torque_string:
            delimiter = '@'
        elif 'at' in torque_string:
            delimiter = 'at'
        else:
            return pd.NA, pd.NA
        
        if 'kgm' in torque_string:
            # convert to Nm
            torque_val = float(torque_string.split('kgm')[0].split(delimiter)[0])
            torque_val *= 9.80665
        else:
            torque_val = float(torque_string.split('nm')[0])
        
        if '-' in torque_string:
            rpm = float(torque_string.split(delimiter)[1].split('-')[0].replace('rpm', ''))
        else:
            rpm = float(torque_string.split(delimiter)[1].replace('rpm', '').replace('(', '').replace(')', ''))
    except:
        return pd.NA, pd.NA
    return torque_val, rpm


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df['torque_nm'], df['max_torque_rpm'] = extract_torque(df['torque'])
    df = df.drop(columns=['torque'])
    df['name'] = df['name'].apply(lambda x: x.split(' ')[0])
    df['torque_nm'] = pd.to_numeric(df['torque_nm'], errors='coerce')
    df['max_torque_rpm'] = pd.to_numeric(df['max_torque_rpm'], errors='coerce')
    df['engine'] = df['engine'].astype(int)
    df['seats'] = df['seats'].astype(int)
    df_ohe = ohe.transform(df[columns_for_ohe])
    X = np.concatenate([df_ohe, df.drop(columns=columns_for_ohe).values], axis=1)
    X_scaled = scaler.transform(X)
    return X_scaled


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    df_instance = pydantic_model_to_df(item)
    preprocessed_data = preprocess_data(df_instance)
    prediction = model.predict(preprocessed_data).tolist()[0]

    response = item.model_dump(by_alias=True)
    response.update({'Prediction': prediction})
    return response


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    df_instances = pd.DataFrame([jsonable_encoder(item) for item in items])
    preprocessed_data = preprocess_data(df_instances)
    predictions = model.predict(preprocessed_data).tolist()
    return [{**item.model_dump(by_alias=True), 'Prediction': prediction} for item, prediction in zip(items, predictions)]

