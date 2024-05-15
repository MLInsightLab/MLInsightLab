from fastapi.responses import RedirectResponse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import mlflow

class PredictRequest(BaseModel):
    data : list
    predict_function : str = 'predict'
    model_flavor : str = 'pyfunc'
    dtype : str = None
    params : dict = None

# Load_model function that allows to load model from either alias or version

# Predict model function that allows to predict

app = FastAPI()

@app.get('/', include_in_schema = False)
def redirect_docs():
    return RedirectResponse(url = '/inference/docs')

@app.post('/{model_name}/version/{model_version}')
def predict(model_name : str, model_version : str | int, body : PredictRequest):

    # Load the model
    # NOTE: "transformer" should also be supported here, but there are unknowns with running inference directly
    try:
        if body.model_flavor == 'pyfunc':
            model = mlflow.pyfunc.load_model(f'models:/{model_name}/{model_version}')
        elif body.model_flavor == 'sklearn':
            model = mlflow.sklearn.load_model(f'models:/{model_name}/{model_version}')
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
    
    # Grab the data to predict on from the input body
    try:
        to_predict = np.array(body.data)
        if body.dtype:
            to_predict = to_predict.astype(body.dtype)
    except Exception:
        raise HTTPException(
            400,
            'Data malformed and could not be processed'
        )
    
    # If predict_function is "predict"
    if body.predict_function == 'predict':
        try:
            if body.model_flavor != 'sklearn':
                results = model.predict(to_predict, params = body.params)
            else:
                results = model.predict(to_predict)
        except Exception:
            try:
                results = model.predict(to_predict.reshape(1, -1))
            except Exception:
                raise HTTPException(400, 'There was an issue running `predict` with the provided data')
    
    # Else if the predict function is "predict_proba"
    elif body.predict_function == 'predict_proba':
        try:
            results = model.predict_proba(to_predict)
        except Exception:
            try:
                results = model.predict_proba(to_predict.reshape(1, -1))
            except Exception:
                raise HTTPException(400, 'There was an issue running `predict_proba` with the provided data')
            
    else:
        raise HTTPException(
            400,
            f'Only `predict` and `predict_proba` are supported predict functions, got {body.predict_function}'
        )
    
    # Convert results to list if they are an array
    if isinstance(results, np.ndarray):
        results = results.tolist()

    return {
        'prediction' : results
    }

@app.post('/{model_name}/alias/{model_alias}')
def predict_alias(model_name : str, model_alias : str | int, body : PredictRequest):

    # Load the model
    # NOTE: "transformer" should also be supported here, but there are unknowns with running inference directly
    try:
        if body.model_flavor == 'pyfunc':
            model = mlflow.pyfunc.load_model(f'models:/{model_name}@{model_alias}')
        elif body.model_flavor == 'sklearn':
            model = mlflow.sklearn.load_model(f'models:/{model_name}@{model_alias}')
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
    
    # Grab the data to predict on from the input body
    try:
        to_predict = np.array(body.data)
        if body.dtype:
            to_predict = to_predict.astype(body.dtype)
    except Exception:
        raise HTTPException(
            400,
            'Data malformed and could not be processed'
        )
    
    # If predict_function is "predict"
    if body.predict_function == 'predict':
        try:
            if body.model_flavor != 'sklearn':
                results = model.predict(to_predict, params = body.params)
            else:
                results = model.predict(to_predict)
        except Exception:
            try:
                results = model.predict(to_predict.reshape(1, -1))
            except Exception:
                raise HTTPException(400, 'There was an issue running `predict` with the provided data')
    
    # Else if the predict function is "predict_proba"
    elif body.predict_function == 'predict_proba':
        try:
            results = model.predict_proba(to_predict)
        except Exception:
            try:
                results = model.predict_proba(to_predict.reshape(1, -1))
            except Exception:
                raise HTTPException(400, 'There was an issue running `predict_proba` with the provided data')
            
    else:
        raise HTTPException(
            400,
            f'Only `predict` and `predict_proba` are supported predict functions, got {body.predict_function}'
        )
    
    # Convert results to list if they are an array
    if isinstance(results, np.ndarray):
        results = results.tolist()

    return {
        'prediction' : results
    }
