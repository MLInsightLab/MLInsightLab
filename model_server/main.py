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
def load_model(model_name, model_flavor, model_version = None, model_alias = None):
    
    if not (model_version or model_alias):
        raise ValueError('Model version or model alias must be provided')
    
    # NOTE: "transformer" should also be supported here, but there are unknowns with running inference directly
    if model_flavor not in ['pyfunc', 'sklearn']:
        raise ValueError(f'Only "pyfunc" and "sklearn" model flavors supported, got {model_flavor}')
    
    try:
        
        if model_flavor == 'pyfunc':
            if model_version:
                model = mlflow.pyfunc.load_model(f'models:/{model_name}/{model_version}')
            elif model_alias:
                model = mlflow.pyfunc.load_model(f'models:/{model_name}@{model_alias}')

        elif model_flavor == 'sklearn':
            if model_version:
                model = mlflow.sklearn.load_model(f'models:/{model_name}/{model_version}')
            elif model_alias:
                model = mlflow.sklearn.load_model(f'models:/{model_name}@{model_alias}')
        
        return model
    
    except Exception as e:
        raise mlflow.MlflowException('Could not load model')

# Predict_model function that runs prediction
def predict_model(model, to_predict, model_flavor, predict_function, params):

    if predict_function == 'predict':
        try:
            if model_flavor != 'sklearn':
                results = model.predict(to_predict, params = params)
            else:
                results = model.predict(to_predict)
        except Exception:
            try:
                results = model.predict(to_predict.reshape(-1, 1))
            except Exception:
                raise ValueError('There was an issue running `predict`')
    
    elif predict_function == 'predict_proba':
        try:
            results = model.predict_proba(to_predict)
        except Exception:
            try:
                results = model.predict_proba(to_predict.reshape(-1, 1))
            except Exception:
                raise ValueError('There was an issue running `predict_proba`')
    
    else:
        raise ValueError('Only `predict` and `predict_proba` are supported predict functions')
    
    if isinstance(results, np.ndarray):
        results = results.tolist()

    return {
        'prediction' : results
    }


app = FastAPI()

@app.get('/', include_in_schema = False)
def redirect_docs():
    return RedirectResponse(url = '/inference/docs')

@app.post('/{model_name}/version/{model_version}')
def predict(model_name : str, model_version : str | int, body : PredictRequest):

    # Load the model
    try:
        model = load_model(model_name, body.model_flavor, model_version)
    except Exception as e:
        if isinstance(e, mlflow.MlflowException):
            raise HTTPException(404, e.message)
        else:
            raise HTTPException(400, e.message)
    
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
    
    try:
        return predict_model(
            model,
            to_predict,
            body.model_flavor,
            body.predict_function,
            body.params
        )
    except Exception as e:
        raise HTTPException(400, e.message)

@app.post('/{model_name}/alias/{model_alias}')
def predict_alias(model_name : str, model_alias : str | int, body : PredictRequest):

    # Load the model
    model = load_model(model_name, body.model_flavor, model_alias = model_alias)
    
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
    
    # Run prediction
    try:
        return predict_model(
            model,
            to_predict,
            body.model_flavor,
            body.predict_function,
            body.params
        )
    except Exception as e:
        raise HTTPException(400, e.message)