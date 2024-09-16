from fastapi import FastAPI, HTTPException, Depends, Body, BackgroundTasks
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from transformers import pipeline, BitsAndBytesConfig
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import numpy as np
import subprocess
import mlflow
import base64
import signal
import json
import os

from db_utils import setup_database, validate_user_key, validate_user_password, fcreate_user, fdelete_user, fissue_new_api_key, fissue_new_password, fget_user_role, fupdate_user_role, flist_users, SERVED_MODEL_CACHE_FILE

# Set up the database
setup_database()

# Global variables for model flavors
ALLOWED_MODEL_FLAVORS = [
    'pyfunc',
    'sklearn',
    'transformers',
    'hfhub'
]
PYFUNC_FLAVOR = ALLOWED_MODEL_FLAVORS[0]
SKLEARN_FLAVOR = ALLOWED_MODEL_FLAVORS[1]
TRANSFORMERS_FLAVOR = ALLOWED_MODEL_FLAVORS[2]
HUGGINGFACE_FLAVOR = ALLOWED_MODEL_FLAVORS[3]

# Global variables for prediction functions
ALLOWED_PREDICT_FUNCTIONS = [
    'predict',
    'predict_proba'
]
PREDICT = ALLOWED_PREDICT_FUNCTIONS[0]
PREDICT_PROBA = ALLOWED_PREDICT_FUNCTIONS[1]

DATA_DIRECTORY = '/data/'

VARIABLE_STORE_DIRECTORY = '/variable_store/'
VARIABLE_STORE_FILE = os.path.join(
    VARIABLE_STORE_DIRECTORY, 'variable_store.json')

# Load the variable store
try:
    with open(VARIABLE_STORE_FILE, 'r') as f:
        variable_store = json.load(f)
except Exception:
    variable_store = {}

# Load_model function that allows to load model from either alias or version


def fload_model(
    model_name: str,
    model_flavor: str,
    model_version: str | int | None = None,
    model_alias: str | None = None,
    requirements: str | None = None,
    quantization_kwargs: dict | None = None,
    **kwargs
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
    requirements : str or None (default None)
        Any pip requirements for loading the model
    quantization_kwargs : dict or None (default None)
        Quantization keyword arguments. NOTE: Only applies for hfhub models
    **kwargs : additional keyword arguments
        Additional keyword arguments. NOTE: Only applies to hfhub models

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

    if not (model_version or model_alias) and model_flavor != HUGGINGFACE_FLAVOR:
        raise ValueError('Model version or model alias must be provided')

    if model_flavor not in ALLOWED_MODEL_FLAVORS:
        raise ValueError(
            f'Only "pyfunc", "sklearn", "transformers", and "hfhub" model flavors supported, got {model_flavor}')

    try:

        # If the model is not a huggingface model, then format the model uri
        if model_flavor != HUGGINGFACE_FLAVOR:
            if model_version:
                model_uri = f'models:/{model_name}/{model_version}'
            elif model_alias:
                model_uri = f'models:/{model_name}@{model_alias}'

            # Install dependencies for the model from mlflow
            subprocess.run(
                [
                    'pip',
                    'install',
                    '-r',
                    mlflow.pyfunc.get_model_dependencies(model_uri)
                ]
            )

        # Install requirements for the model if it's a huggingface model
        else:
            if requirements:
                subprocess.run(
                    [
                        'pip',
                        'install',
                        requirements
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
            if mlflow.transformers.is_gpu_available():
                # NOTE: This loads the model to GPU automatically
                # TODO: Change this so that it can be done more intelligently
                model = mlflow.transformers.load_model(
                    model_uri,
                    kwargs={
                        'device_map': 'auto'
                    }
                )
            else:
                model = mlflow.transformers.load_model(model_uri)

        # Load the model if it is a huggingface model
        elif model_flavor == HUGGINGFACE_FLAVOR:
            if quantization_kwargs:
                bnb_config = BitsAndBytesConfig(**quantization_kwargs)
                if not kwargs.get('model_kwargs'):
                    kwargs['model_kwargs'] = {}
                kwargs['model_kwargs']['quantization_config'] = bnb_config

            model = pipeline(**kwargs)

        return model

    except Exception:
        raise mlflow.MlflowException('Could not load model')

# Function to load models from cache


def load_models_from_cache():
    """
    Load models from the cache directory
    """
    try:
        with open(SERVED_MODEL_CACHE_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return None


# Load all models from cache
try:
    models_to_load = load_models_from_cache()
    LOADED_MODELS = {}

    for model_info in models_to_load:
        model_name = model_info['model_name']
        model_flavor = model_info['model_flavor']
        model_version_or_alias = model_info['model_version_or_alias']

        requirements = model_info.get('requirements')
        quantization_kwargs = model_info.get('quantization_kwargs')
        kwargs = model_info.get('kwargs')

        try:
            model = fload_model(
                model_name,
                model_flavor,
                model_version_or_alias,
                requirements=requirements,
                quantization_kwargs=quantization_kwargs,
                **kwargs
            )
            if not LOADED_MODELS.get(model_name):
                LOADED_MODELS[model_name] = {
                    model_flavor: {
                        model_version_or_alias: {
                            'model': model,
                            'requirements': requirements,
                            'quantization_kwargs': quantization_kwargs,
                            'kwargs': kwargs
                        }
                    }
                }
            elif not LOADED_MODELS[model_name].get(model_flavor):
                LOADED_MODELS[model_name][model_flavor] = {
                    model_version_or_alias: {
                        'model': model,
                        'requirements': requirements,
                        'quantization_kwargs': quantization_kwargs,
                        'kwargs': kwargs
                    }
                }
            elif not LOADED_MODELS[model_flavor].get(model_version_or_alias):
                LOADED_MODELS[model_name][model_flavor][model_version_or_alias] = {
                    'model': model,
                    'requirements': requirements,
                    'quantization_kwargs': quantization_kwargs,
                    'kwargs': kwargs
                }

        except Exception:
            try:
                model = fload_model(
                    model_name,
                    model_flavor,
                    model_alias=model_version_or_alias,
                    requirements=requirements,
                    quantization_kwargs=quantization_kwargs,
                    **kwargs
                )
                if not LOADED_MODELS.get(model_name):
                    LOADED_MODELS[model_name] = {
                        model_flavor: {
                            model_version_or_alias: {
                                'model': model,
                                'requirements': requirements,
                                'quantization_kwargs': quantization_kwargs,
                                'kwargs': kwargs
                            }
                        }
                    }
                elif not LOADED_MODELS[model_name].get(model_flavor):
                    LOADED_MODELS[model_name][model_flavor] = {
                        model_version_or_alias: {
                            'model': model,
                            'requirements': requirements,
                            'quantization_kwargs': quantization_kwargs,
                            'kwargs': kwargs
                        }
                    }
                elif not LOADED_MODELS[model_flavor].get(model_version_or_alias):
                    LOADED_MODELS[model_name][model_flavor][model_version_or_alias] = {
                        'model': model,
                        'requirements': requirements,
                        'quantization_kwargs': quantization_kwargs,
                        'kwargs': kwargs
                    }
            except Exception:
                raise ValueError('Model not able to be loaded')

except Exception:
    LOADED_MODELS = {}


# Function to save models to cache
def save_models_to_cache():
    """
    Save models to the cache directory
    """
    to_save = []
    if LOADED_MODELS != {}:
        for model_name in LOADED_MODELS.keys():
            for model_flavor in LOADED_MODELS[model_name]:
                for model_version_or_alias in LOADED_MODELS[model_name][model_flavor].keys():
                    requirements = LOADED_MODELS[model_name][model_flavor][model_version_or_alias]['requirements']
                    quantization_kwargs = LOADED_MODELS[model_name][model_flavor][
                        model_version_or_alias]['quantization_kwargs']
                    kwargs = LOADED_MODELS[model_name][model_flavor][model_version_or_alias]['kwargs']
                    to_save.append(
                        dict(
                            model_name=model_name,
                            model_flavor=model_flavor,
                            model_version_or_alias=model_version_or_alias,
                            requirements=requirements,
                            quantization_kwargs=quantization_kwargs,
                            kwargs=kwargs
                        )
                    )
    with open(SERVED_MODEL_CACHE_FILE, 'w') as f:
        json.dump(to_save, f)


def upload_data_to_fs(
        filename: str,
        file_text: str | None = None,
        file_bytes: str | None = None,
        overwrite: bool = False
):
    """
    Upload data to the data store

    Parameters:
    -----------
    filename : str
        The name of the file, either with or without /data prepended
    file_text : str or None (default None)
        The text of the file, if it is a text file
    file_bytes : str or None (default None)
        The bytes of the file, encoded base64 and then to utf-8, if a binary file
    overwrite : bool (default False)
        Whether to overwrite the file if it already exists

    Returns
    -------
    filename : str
        The final filename of the file, on disk
    """

    # Determine if text or bytes is to be used (only one can be specified)
    if file_text and file_bytes:
        raise ValueError('Only file_text or file_bytes should be specified')

    # Ensure that the data directory leads
    if not filename.startswith(DATA_DIRECTORY):
        filename = os.path.join(
            DATA_DIRECTORY,
            filename.lstrip('/').strip()
        )

    # If the file exists and overwrite False, then raise an Exception
    if os.path.exists(filename) and not overwrite:
        raise FileExistsError(
            'Data file already exists and overwrite was not set to True')

    # Create any intermediate directories if needed
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Determine the content of the file
    if file_text:
        file_content = file_text
        with open(filename, 'w') as f:
            f.write(file_content)
    else:
        file_content = base64.b64decode(
            file_bytes.encode('utf-8')
        )
        with open(filename, 'wb') as f:
            f.write(file_content)

    return filename


def download_data_from_fs(
        filename: str,
        as_bytes: bool = False
):
    """
    Download a file from the file system

    Parameters
    ----------
    filename : str
        The name of the file
    as_bytes : bool (default False)
        Whether the file is a binary file or not

    Returns
    -------
    content : str
        The content of the file, as a string (needs to be coded to bytes if a binary file)
    """
    if not filename.startswith(DATA_DIRECTORY):
        filename = os.path.join(
            DATA_DIRECTORY,
            filename.lstrip('/').strip()
        )

    if not os.path.exists(filename):
        raise FileNotFoundError('File does not exist')

    if as_bytes:
        with open(filename, 'rb') as f:
            content = f.read()
        content = base64.b64encode(content).decode('utf-8')
    else:
        with open(filename, 'r') as f:
            content = f.read()

    return content


class PredictRequest(BaseModel):
    data: list
    predict_function: str = 'predict'
    dtype: str = None
    params: dict = None


class LoadRequest(BaseModel):
    requirements: str | None = None
    quantization_kwargs: dict | None = None
    kwargs: dict | None = None


class UserInfo(BaseModel):
    username: str
    role: str
    api_key: str | None = None
    password: str | None = None


class DataUploadRequest(BaseModel):
    filename: str
    file_text: str | None = None
    file_bytes: str | None = None
    overwrite: bool = False


class DataDownloadRequest(BaseModel):
    filename: str
    as_bytes: bool = False


class VariableSetRequest(BaseModel):
    value: str | int | float | bool | dict | list
    overwrite: bool = False


class VerifyPasswordInfo(BaseModel):
    username: str
    password: str

# Function to load a model in the background


def load_model_background(
    model_name: str,
    model_flavor: str,
    model_version_or_alias: str | int,
    requirements: str | None,
    quantization_kwargs: dict | None,
    **kwargs
):
    """
    Load a model in the background
    """
    try:
        model = fload_model(
            model_name,
            model_flavor,
            model_version=model_version_or_alias,
            requirements=requirements,
            quantization_kwargs=quantization_kwargs,
            **kwargs
        )
    except Exception:
        try:
            model = fload_model(
                model_name,
                model_flavor,
                model_alias=model_version_or_alias,
                requirements=requirements,
                quantization_kwargs=quantization_kwargs,
                **kwargs
            )
        except Exception:
            raise ValueError('Model not able to be loaded')

    if not LOADED_MODELS.get(model_name):
        LOADED_MODELS[model_name] = {
            model_flavor: {
                model_version_or_alias: {
                    'model': model,
                    'requirements': requirements,
                    'quantization_kwargs': quantization_kwargs,
                    'kwargs': kwargs
                }
            }
        }
    elif not LOADED_MODELS[model_name].get(model_flavor):
        LOADED_MODELS[model_name][model_flavor] = {
            model_version_or_alias: {
                'model': model,
                'requirements': requirements,
                'quantization_kwargs': quantization_kwargs,
                'kwargs': kwargs
            }
        }
    elif not LOADED_MODELS[model_name][model_flavor].get(model_version_or_alias):
        LOADED_MODELS[model_name][model_flavor][model_version_or_alias] = {
            'model': model,
            'requirements': requirements,
            'quantization_kwargs': quantization_kwargs,
            'kwargs': kwargs
        }

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
            elif model_flavor in [TRANSFORMERS_FLAVOR, HUGGINGFACE_FLAVOR]:
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
                elif model_flavor in [TRANSFORMERS_FLAVOR, HUGGINGFACE_FLAVOR]:
                    if params:
                        results = model(to_predict, **params)
                    else:
                        results = model(to_predict)
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
    except Exception as e:
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
    except Exception as e:
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
    """
    Redirect the main page to the docs site
    """
    return RedirectResponse(url='/api/docs')


@app.post('/models/load/{model_name}/{model_flavor}/{model_version_or_alias}')
def load_model(model_name: str, model_flavor: str, model_version_or_alias: str | int, body: LoadRequest, background_tasks: BackgroundTasks, user_properties: dict = Depends(verify_credentials)):
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
    body : LoadRequest
        Additional parameters to load the model
    """

    try:
        background_tasks.add_task(
            load_model_background,
            model_name,
            model_flavor,
            model_version_or_alias,
            body.requirements,
            body.quantization_kwargs,
            **body.kwargs
        )
    except Exception:
        background_tasks.add_task(
            load_model_background,
            model_name,
            model_flavor,
            model_version_or_alias,
            body.requirements,
            body.quantization_kwargs
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
    try:
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
    except Exception:
        raise HTTPException(500, 'An unknown error occurred')

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
        model = LOADED_MODELS[model_name][model_flavor][model_version_or_alias]['model']
    except Exception:

        # Model needs to be loaded
        raise HTTPException(
            404, 'That model is not loaded. Please load the model by calling the /models/load endpoint first'
        )

    # Grab the data to predict on from the input body
    try:
        if model_flavor not in [TRANSFORMERS_FLAVOR, HUGGINGFACE_FLAVOR]:
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
        try:
            return fcreate_user(
                user_info.username,
                user_info.role,
                user_info.api_key,
                user_info.password
            )
        except Exception as e:
            raise HTTPException(500, f'The following error occurred: {str(e)}')

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
        try:
            return fdelete_user(
                username
            )
        except Exception:
            raise HTTPException(500, 'An unknown error occurred')

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
        try:
            return fissue_new_api_key(
                username
            )
        except Exception as e:
            raise HTTPException(
                400,
                str(e)
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
        try:
            return fissue_new_password(
                username,
                new_password
            )
        except Exception as e:
            raise HTTPException(
                400,
                str(e)
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

    try:
        return fget_user_role(username)
    except Exception:
        raise HTTPException(500, 'An unknown error occurred')

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

    try:
        return fupdate_user_role(
            username,
            new_role
        )
    except Exception:
        raise HTTPException(500, 'An unknown error occurred')

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

    try:
        return flist_users()
    except Exception:
        raise HTTPException(500, 'An unknown error occurred')


@app.get('/reset')
def reset(user_properties: dict = Depends(verify_credentials)):
    """
    Reset the API, redeploying all models
    """
    if user_properties['role'] != 'admin':
        raise HTTPException(
            403,
            'User does not have permissions'
        )

    os.kill(os.getpid(), signal.SIGTERM)
    return {
        'success': True
    }


@app.get('/system/resource-usage')
def get_usage(user_properties: dict = Depends(verify_credentials)):
    """
    Get system resource usage, in terms of free CPU and GPU memory (if GPU-enabled)
    """

    if user_properties['role'] != 'admin':
        raise HTTPException(
            403,
            'User does not have permissions'
        )

    try:
        cpu_memory_output = subprocess.run(
            ['free', '-h'], check=True, capture_output=True)
        cpu_memory_output = cpu_memory_output.stdout.decode('utf-8')
    except Exception:
        raise HTTPException(
            500,
            'An unknown error occurred'
        )

    try:
        gpu_memory_output = subprocess.run(
            ['nvidia-smi'], check=True, capture_output=True)
        gpu_memory_output = gpu_memory_output.stdout.decode('utf-8')
    except Exception:
        gpu_memory_output = 'No GPU status detected'

    return {
        'cpu_memory_usage': cpu_memory_output,
        'gpu_memory_usage': gpu_memory_output
    }


@app.post('/data/upload')
def upload_file(body: DataUploadRequest, user_properties: dict = Depends(verify_credentials)):
    """
    Upload a file to the data store

    Parameters
    ----------
    body : DataUploadRequest
        Properties of the file to upload

    Returns
    -------
    filename : str
        The full path to the file on disk, in the data directory
    """
    if user_properties['role'] not in ['admin', 'data_scientist']:
        raise HTTPException(
            403,
            'User does not have permissions'
        )

    try:
        filename = upload_data_to_fs(
            body.filename,
            body.file_text,
            body.file_bytes,
            body.overwrite
        )
        return filename
    except Exception as e:
        raise HTTPException(
            400,
            f'The following error occurred: {str(e)}'
        )


@app.post('/data/download')
def download_file(body: DataDownloadRequest, user_properties: dict = Depends(verify_credentials)):
    """
    Download a file from the data drive

    Parameters
    ----------
    body : DataDownloadRequest
        The information about the file to download

    Returns
    -------
    content : str
        The content of the file, as a string
    """
    if user_properties['role'] not in ['admin', 'data_scientist']:
        raise HTTPException(
            403,
            'User does not have permissions'
        )

    try:
        content = download_data_from_fs(
            body.filename,
            body.as_bytes
        )
        return content

    except Exception as e:
        raise HTTPException(
            400,
            f'The following error occurred: {str(e)}'
        )


@app.get('/variable-store/get/{variable}')
def get_variable(variable, user_properties: dict = Depends(verify_credentials)):
    """
    Retrieve a variable from the variable store

    Parameters
    ----------
    variable : str
        The identifier of the variable
    """
    if user_properties['role'] not in ['admin', 'data_scientist']:
        raise HTTPException(
            403,
            'User does not have permissions'
        )

    try:
        return {
            variable: variable_store[user_properties['username']][variable]
        }
    except Exception:
        raise HTTPException(
            404,
            'User does not have a variable with that identifier saved'
        )


@app.get('/variable-store/list')
def list_variables(user_properties: dict = Depends(verify_credentials)):
    """
    List your variables
    """
    if user_properties['role'] not in ['admin', 'data_scientist']:
        raise HTTPException(
            403,
            'User does not have permissions'
        )

    # Try to return list of variable names
    try:
        return list(variable_store[user_properties['username']].keys())

    # No variables for user, return empty list
    except Exception:
        return []


@app.post('/variable-store/set/{variable}')
def set_variable(variable, variable_properties: VariableSetRequest, user_properties: dict = Depends(verify_credentials)):
    """
    Set a variable

    Parameters:
    variable : str
        The variable identifier
    variable_properties : VariableSetRequest
        JSON payload with the value for the variable and whether to overwrite the variable if it is already set
    """
    if user_properties['role'] not in ['admin', 'data_scientist']:
        raise HTTPException(
            403,
            'User does not have permissions'
        )

    # Check if the variable exists and overwrite is False
    if not variable_properties.overwrite:
        try:
            existing_variable = variable_store[user_properties['username']][variable]
        except Exception:
            existing_variable = None

        if existing_variable:
            raise HTTPException(
                400,
                'Variable already exists and overwrite was False'
            )

    # Now, try to write to the variable store, but be careful about edge cases
    if not variable_store.get(user_properties['username']):
        variable_store[user_properties['username']] = {
            variable: variable_properties.value
        }
    else:
        variable_store[user_properties['username']
                       ][variable] = variable_properties.value

    # Write the variable store to disk
    with open(VARIABLE_STORE_FILE, 'w') as f:
        json.dump(variable_store, f)

    return {
        'success': True
    }


@app.delete('/variable-store/delete/{variable}')
def delete_variable(variable, user_properties: dict = Depends(verify_credentials)):
    """
    Delete a variable
    """
    if not user_properties['role'] in ['admin', 'data_scientist']:
        raise HTTPException(
            403,
            'User does not have permissions'
        )

    # Try to delete the specified variable for the user and rewrite the variable store
    try:
        del variable_store[user_properties['username']][variable]
        with open(VARIABLE_STORE_FILE, 'w') as f:
            json.dump(variable_store, f)

    # If any error occurs, return HTTPException with 404 code
    except Exception:
        raise HTTPException(
            404,
            'No variable to delete'
        )
