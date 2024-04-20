from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel
import numpy as np
import mlflow

class PredictRequest(BaseModel):
    model_name : str
    model_version : str | int
    data : list
    predict_function : str = 'predict'
    model_flavor : str = 'pyfunc'

app = FastAPI()

@app.post('/predict')
def predict(body : PredictRequest):
    try:
        if body.model_flavor == 'pyfunc':
            model = mlflow.pyfunc.load_model(f'models:/{body.model_name}/{body.model_version}')
        elif body.model_flavor == 'sklearn':
            model = mlflow.sklearn.load_model(f'models:/{body.model_name}/{body.model_version}')
        else:
            raise HTTPException(
                400,
                f'Only "pyfunc" and "sklearn" model flavors are currently supported, got {body.model_flavor}'
            )
    except Exception:
        raise HTTPException(
            404,
            'Model ID not found'
        )
    
    try:
        to_predict = np.array(body.data)
    except Exception:
        raise HTTPException(
            400,
            'Data malformed and could not be processed'
        )
    
    if body.predict_function == 'predict':
        try:
            results = model.predict(to_predict).tolist()
        except Exception:
            try:
                results = model.predict(to_predict.reshape(1, -1)).tolist()
            except Exception:
                raise HTTPException(400, 'There was an issue running `predict` with the provided data')
    
    elif body.predict_function == 'predict_proba':
        try:
            results = model.predict_proba(to_predict).tolist()
        except Exception:
            try:
                results = model.predict_proba(to_predict.reshape(1, -1)).tolist()
            except Exception:
                raise HTTPException(400, 'There was an issue running `predict_proba` with the provided data')
            
    else:
        raise HTTPException(
            400,
            f'Only `predict` and `predict_proba` are supported predict functions, got {body.predict_function}'
        )
    
    return {
        'prediction' : results
    }
