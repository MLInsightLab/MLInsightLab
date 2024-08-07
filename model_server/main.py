from fastapi import FastAPI, HTTPException, Depends, Body, BackgroundTasks
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import numpy as np
import subprocess
import mlflow
import json

from db_utils import setup_database, validate_user_key, validate_user_password, fcreate_user, fdelete_user, fissue_new_api_key, fissue_new_password, fget_user_role, fupdate_user_role, flist_users, SERVED_MODEL_CACHE_FILE

# Set up the database
setup_database()

# Global variables for model flavors
ALLOWED_MODEL_FLAVORS = [
    'pyfunc',
    'sklearn',
    'transformers'
]
PYFUNC_FLAVOR = ALLOWED_MODEL_FLAVORS[0]
SKLEARN_FLAVOR = ALLOWED_MODEL_FLAVORS[1]
TRANSFORMERS_FLAVOR = ALLOWED_MODEL_FLAVORS[2]

# Global variables for prediction functions
ALLOWED_PREDICT_FUNCTIONS = [
    'predict',
    'predict_proba'
]
PREDICT = ALLOWED_PREDICT_FUNCTIONS[0]
PREDICT_PROBA = ALLOWED_PREDICT_FUNCTIONS[1]

# Load_model function that allows to load model from either alias or version


def fload_model(
    model_name: str,
    model_flavor: str,
    model_version: str | int | None = None,
    model_alias: str | None = None
):
    f"""
    Load a model from the MLFlow server

    Parameters
    ----------
    model_name : str
        The name of the model
    model_flavor : str
        The flavor of the model, must be one of {ALLOWED_MODEL_FLAVORS}
    model_version : int or None (default None)
        The version of the model
    model_alias : str or None (default None)
        The alias of the model, without the `@` character

    Notes
    -----
    - One of either `model_version` or `model_alias` must be provided

    Returns
    -------
    model : mlflow Model
        The model, in the flavor specified

    Raises
    ------
    - MlflowException, when the model cannot be loaded
    """

    if not (model_version or model_alias):
        raise ValueError('Model version or model alias must be provided')

    if model_flavor not in ALLOWED_MODEL_FLAVORS:
        raise ValueError(
            f'Only "pyfunc" and "sklearn" model flavors supported, got {model_flavor}')

    try:

        if model_version:
            model_uri = f'models:/{model_name}/{model_version}'
        elif model_alias:
            model_uri = f'models:/{model_name}@{model_alias}'

        # Install dependencies for the model
        subprocess.run(
            [
                'pip',
                'install',
                '-r',
                mlflow.pyfunc.get_model_dependencies(model_uri)
            ]
        )

        # Load the model if it is requested to be a pyfunc model
        if model_flavor == PYFUNC_FLAVOR:
            model = mlflow.pyfunc.load_model(model_uri)

        # Load the model if it is requested to be a sklearn model
        elif model_flavor == SKLEARN_FLAVOR:
            model = mlflow.sklearn.load_model(model_uri)

        # Load the model if it is requested to be a transformers model
        elif model_flavor == TRANSFORMERS_FLAVOR:
            model = mlflow.transformers.load_model(model_uri)

        return model

    except Exception:
        raise mlflow.MlflowException('Could not load model')

# Function to load models from cache
def load_models_from_cache():
    try:
        with open(SERVED_MODEL_CACHE_FILE, 'r') as f:
            return json.load(f)
    except:
        return None

# Load all models from cache
try:
    models_to_load = load_models_from_cache()
    LOADED_MODELS = {}
    
    for model_info in models_to_load:
        model_name = model_info['model_name']
        model_flavor = model_info['model_flavor']
        model_version_or_alias = model_info['model_version_or_alias']

        try:
            model = fload_model(
                model_name,
                model_flavor,
                model_version_or_alias
            )
            if not LOADED_MODELS.get(model_name):
                LOADED_MODELS[model_name] = {
                    model_flavor : {
                        model_version_or_alias : model
                    }
                }
            elif not LOADED_MODELS[model_name].get(model_flavor):
                LOADED_MODELS[model_name][model_flavor] = {
                    model_version_or_alias: model
                }
            elif not LOADED_MODELS[model_flavor].get(model_version_or_alias):
                LOADED_MODELS[model_name][model_flavor][model_version_or_alias] = model
        
        except Exception:
            try:
                model = fload_model(
                    model_name,
                    model_flavor,
                    model_alias = model_version_or_alias
                )
                if not LOADED_MODELS.get(model_name):
                    LOADED_MODELS[model_name] = {
                        model_flavor : {
                            model_version_or_alias : model
                        }
                    }
                elif not LOADED_MODELS[model_name].get(model_flavor):
                    LOADED_MODELS[model_name][model_flavor] = {
                        model_version_or_alias: model
                    }
                elif not LOADED_MODELS[model_flavor].get(model_version_or_alias):
                    LOADED_MODELS[model_name][model_flavor][model_version_or_alias] = model
            except Exception:
                raise ValueError('Model not able to be loaded')

except:
    LOADED_MODELS = {}


# Function to save models to cache
def save_models_to_cache():
    to_save = []
    if LOADED_MODELS != {}:
        for model_name in LOADED_MODELS.keys():
            for model_flavor in LOADED_MODELS[model_name]:
                for model_version_or_alias in LOADED_MODELS[model_name][model_flavor].keys():
                    to_save.append(
                        dict(
                            model_name = model_name,
                            model_flavor = model_flavor,
                            model_version_or_alias = model_version_or_alias
                        )
                    )
    with open(SERVED_MODEL_CACHE_FILE, 'w') as f:
        json.dump(to_save, f)

class PredictRequest(BaseModel):
    data: list
    predict_function: str = 'predict'
    dtype: str = None
    params: dict = None


class UserInfo(BaseModel):
    username: str
    role: str
    api_key: str | None = None
    password: str | None = None

class VerifyPasswordInfo(BaseModel):
    username: str
    password: str

# Function to load a model in the background
def load_model_background(model_name : str, model_flavor : str, model_version_or_alias : str|int):
    try:
        model = fload_model(
            model_name,
            model_flavor,
            model_version = model_version_or_alias
        )
    except Exception:
        try:
            model = fload_model(
                model_name,
                model_flavor,
                model_alias = model_version_or_alias
            )
        except Exception as e:
            raise ValueError('Model not able to be loaded')
        
    if not LOADED_MODELS.get(model_name):
        LOADED_MODELS[model_name] = {
            model_flavor : {
                model_version_or_alias : model
            }
        }
    elif not LOADED_MODELS[model_name].get(model_flavor):
        LOADED_MODELS[model_name][model_flavor] = {
            model_version_or_alias: model
        }
    elif not LOADED_MODELS[model_name][model_flavor].get(model_version_or_alias):
        LOADED_MODELS[model_name][model_flavor][model_version_or_alias] = model

    save_models_to_cache()

    return True

# Predict_model function that runs prediction


def predict_model(
    model: mlflow.models.Model,
    to_predict: np.ndarray,
    model_flavor: str,
    predict_function: str,
    params: dict
):
    f"""
    Make predictions with a model

    Parameters
    ----------
    model : mlflow.models.Model
        The model to run prediction on
    to_predict : np.ndarray or array-like
        The data to predict on
    model_flavor : str
        The flavor of the model, must be one of {ALLOWED_MODEL_FLAVORS}
    predict_function : str
        The predict function to run, must be one of {ALLOWED_PREDICT_FUNCTIONS}
    params : dict
        Parameters to run prediction with
    """
    if predict_function == 'predict':
        try:
            if model_flavor == PYFUNC_FLAVOR:
                results = model.predict(to_predict, params=params)
            elif model_flavor == TRANSFORMERS_FLAVOR:
                if params:
                    results = model(to_predict, **params)
                else:
                    results = model(to_predict)
            elif model_flavor == SKLEARN_FLAVOR:
                results = model.predict(to_predict)
        except Exception:
            try:
                if isinstance(to_predict, np.ndarray):
                    to_predict = to_predict.reshape(-1, 1)
                if model_flavor == PYFUNC_FLAVOR:
                    results = model.predict(to_predict, params=params)
                elif model_flavor == TRANSFORMERS_FLAVOR:
                    results = model(to_predict, **params)
                elif model_flavor == SKLEARN_FLAVOR:
                    results = model.predict(to_predict)
            except Exception as e:
                raise ValueError(
                    f'There was an issue running `predict`: {str(e)}')

    elif predict_function == 'predict_proba':
        try:
            results = model.predict_proba(to_predict)
        except Exception:
            try:
                results = model.predict_proba(to_predict.reshape(-1, 1))
            except Exception:
                raise ValueError('There was an issue running `predict_proba`')

    else:
        raise ValueError(
            'Only `predict` and `predict_proba` are supported predict functions')

    if isinstance(results, np.ndarray):
        results = results.tolist()

    return {
        'prediction': results
    }


# Initialize the app and Basic Auth
app = FastAPI()
security = HTTPBasic()

# Function to verify user credentials


def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    """
    Verify a user's API key credentials
    """
    try:
        role = validate_user_key(
            credentials.username,
            credentials.password
        )
        return {
            'username': credentials.username,
            'role': role
        }
    except ValueError as e:
        raise HTTPException(
            401,
            str(e)
        )

# Function to verify user credentials using password


def verify_credentials_password(credentials: HTTPBasicCredentials = Depends(security)):
    """
    Verify a user's Username/Password credentials
    """
    try:
        role = validate_user_password(
            credentials.username,
            credentials.password
        )
        return {
            'username': credentials.username,
            'role': role
        }
    except ValueError as e:
        raise HTTPException(
            401,
            str(e)
        )

# Verify a user's password

@app.post('/password/verify')
def verify_password(body: VerifyPasswordInfo, user_properties: dict = Depends(verify_credentials)):
    """
    Verify a password

    Parameters
    ----------
    username : str
        The user's username
    password : str
        The user's password
    """
    if user_properties['role'] != 'admin':
        raise HTTPException(
            403,
            'User does not have permission'
        )
    try:
        role = validate_user_password(body.username, body.password)
        return role
    except Exception:
        raise HTTPException(401, 'Incorrect credentials')

# Redirect to docs for the landing page


@app.get('/', include_in_schema=False)
def redirect_docs():
    return RedirectResponse(url='/backend/docs')

@app.get('/models/load/{model_name}/{model_flavor}/{model_version_or_alias}')
def load_model(model_name: str, model_flavor: str, model_version_or_alias: str | int, background_tasks : BackgroundTasks, user_properties: dict = Depends(verify_credentials)):
    """
    Load a model into local memory

    Parameters
    ----------
    model_name : str
        The name of the model
    model_flavor : str
        The flavor of the model
    model_version_or_alias : str or int
        The version or alias of the model
    """

    background_tasks.add_task(
        load_model_background,
        model_name,
        model_flavor,
        model_version_or_alias
    )

    return {
        'Processing': True
    }

# See loaded models


@app.get('/models/list')
def list_models(user_properties: dict = Depends(verify_credentials)):
    """
    List loaded models
    """
    if LOADED_MODELS == {}:
        return []
    else:
        to_return = []
        for model_name in LOADED_MODELS.keys():
            for model_flavor in LOADED_MODELS[model_name]:
                for model_version_or_alias in LOADED_MODELS[model_name][model_flavor].keys():
                    to_return.append(
                        dict(
                            model_name=model_name,
                            model_flavor=model_flavor,
                            model_version_or_alias=model_version_or_alias
                        )
                    )
        return to_return

# Delete a loaded model

@app.delete('/models/unload/{model_name}/{model_flavor}/{model_version_or_alias}')
def unload_model(model_name: str, model_flavor: str, model_version_or_alias: str | int, user_properties: dict = Depends(verify_credentials)):
    """
    Unload a model from memory

    Parameters
    ----------
    model_name : str
        The name of the model
    model_flavor : str
        The flavor of the model
    model_version_or_alias : str or int
        The version or alias of the model
    """
    try:
        del LOADED_MODELS[model_name][model_flavor][model_version_or_alias]
        
        save_models_to_cache()
        
        return {
            'success': True
        }
    except Exception:
        raise HTTPException(404, 'Model not found')    

# Predict using a model version or alias


@app.post('/models/predict/{model_name}/{model_flavor}/{model_version_or_alias}')
def predict(model_name: str, model_flavor: str, model_version_or_alias: str | int, body: PredictRequest, user_properties: dict = Depends(verify_credentials)):
    """
    Run prediction

    Parameters
    ----------
    model_name : str
        The name of the model
    model_flavor : str
        The flavor of the model
    model_version_or_alias : str or int
        The version or alias of the model
    """

    # Try to load the model, assuming it has already been loaded
    try:
        model = LOADED_MODELS[model_name][model_flavor][model_version_or_alias]
    except Exception:

        # Model needs to be loaded
        raise HTTPException(
            404, 'That model is not loaded. Please load the model by calling the /model/load endpoint first'
        )

    # Grab the data to predict on from the input body
    try:
        if model_flavor != TRANSFORMERS_FLAVOR:
            to_predict = np.array(body.data)
            if body.dtype:
                to_predict = to_predict.astype(body.dtype)
        else:
            to_predict = body.data
    except Exception:
        raise HTTPException(
            400,
            'Data malformed and could not be processed'
        )

    try:
        return predict_model(
            model,
            to_predict,
            model_flavor,
            body.predict_function,
            body.params
        )
    except Exception as e:
        raise HTTPException(400, str(e))

# Create User
# Need to create prototype for this, and verify that the user has admin access


@app.post('/users/create')
def create_user(user_info: UserInfo, user_properties: dict = Depends(verify_credentials)):
    """
    Create a user

    Parameters
    ----------
    user_info : UserInfo
        Properties of the user
    """
    if user_properties['role'] != 'admin':
        raise HTTPException(
            403,
            'User does not have permissions'
        )
    else:
        return fcreate_user(
            user_info.username,
            user_info.role,
            user_info.api_key,
            user_info.password
        )

# Delete User


@app.delete('/users/delete/{username}')
def delete_user(username, user_properties: dict = Depends(verify_credentials)):
    """
    Delete a user

    Parameters
    ----------
    username : str
        The username of the user to delete
    """
    if user_properties['role'] != 'admin':
        raise HTTPException(
            403,
            'User does not have permissions'
        )
    else:
        return fdelete_user(
            username
        )

# Issue new API key for user


@app.put('/users/api_key/issue/{username}')
def issue_new_api_key(username, user_properties: dict = Depends(verify_credentials_password)):
    """
    Issue a new API key for a user

    Parameters
    ----------
    username : str
        The username of the user
    """
    if user_properties['role'] != 'admin' or username != user_properties['username']:
        raise HTTPException(
            403,
            'User does not have permissions'
        )
    else:
        return fissue_new_api_key(
            username
        )

# Issue new password for user


@app.put('/users/password/issue/{username}')
def issue_new_password(username, new_password: str = Body(embed=True), user_properties: dict = Depends(verify_credentials)):
    """
    Issue a new password for a user

    Parameters
    ----------
    username : str
        The username of the user
    new_password : str
        The new password for the user
    """
    if user_properties['role'] != 'admin' or username != user_properties['username']:
        raise HTTPException(
            403,
            'User does not have permissions'
        )
    else:
        return fissue_new_password(
            username,
            new_password
        )

# Get user role


@app.get('/users/role/{username}')
def get_user_role(username: str, user_properties: dict = Depends(verify_credentials)):
    """
    Get a user's role

    Parameters
    ----------
    username : str
        The username of the user
    """
    if user_properties['role'] not in ['admin', 'data_scientist']:
        raise HTTPException(
            403,
            'User does not have permissions'
        )

    return fget_user_role(username)

# Update user role


@app.put('/users/role/{username}')
def update_user_role(username: str, new_role=Body(embed=True), user_properties: dict = Depends(verify_credentials)):
    """
    Update a user's role

    Parameters
    ----------
    username : str
        The username for the user
    new_role : str
        The new role for the user
    """
    if user_properties['role'] != 'admin':
        raise HTTPException(
            403,
            'User does not have permissions'
        )
    return fupdate_user_role(
        username,
        new_role
    )

# List users


@app.get('/users/list')
def list_users(user_properties: dict = Depends(verify_credentials)):
    """
    List all users
    """
    if user_properties['role'] not in ['admin', 'data_scientist']:
        raise HTTPException(
            403,
            'User does not have permissions'
        )
    return flist_users()
