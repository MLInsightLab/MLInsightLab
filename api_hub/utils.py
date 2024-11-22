from transformers import pipeline, BitsAndBytesConfig
from db_utils import SERVED_MODEL_CACHE_FILE
from subprocess import check_call
from pydantic import BaseModel
import numpy as np
import subprocess
import mlflow
import base64
import json
import os

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

DATA_DIRECTORY = os.environ['DATA_DIRECTORY']

VARIABLE_STORE_DIRECTORY = os.environ['VARIABLE_STORE_DIRECTORY']
VARIABLE_STORE_FILE = os.path.join(
    VARIABLE_STORE_DIRECTORY, 'variable_store.json')

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


def upload_data_to_fs(
        filename: str,
        file_bytes: str,
        overwrite: bool = False
):
    """
    Upload data to the data store

    Parameters:
    -----------
    filename : str
        The name of the file, either with or without /data prepended
    file_bytes : str
        The bytes of the file, encoded base64 and then to utf-8, if a binary file
    overwrite : bool (default False)
        Whether to overwrite the file if it already exists

    Returns
    -------
    filename : str
        The final filename of the file, on disk
    """

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
        os.makedirs(directory, mode=771)

    # Determine the content of the file
    file_content = base64.b64decode(
        file_bytes.encode('utf-8')
    )

    def opener(path, flags):
        return os.open(path, flags, 0o776)

    with open(filename, 'wb', opener=opener) as f:
        f.write(file_content)

    check_call(
        ['chgrp', 'mlil', filename]
    )

    return filename


def download_data_from_fs(
        filename: str
):
    """
    Download a file from the file system

    Parameters
    ----------
    filename : str
        The name of the file

    Returns
    -------
    content : str
        The content of the file, as a string
    """
    if not filename.startswith(DATA_DIRECTORY):
        filename = os.path.join(
            DATA_DIRECTORY,
            filename.lstrip('/').strip()
        )

    if not os.path.exists(filename):
        raise FileNotFoundError('File does not exist')

    with open(filename, 'rb') as f:
        content = f.read()
    content = base64.b64encode(content).decode('utf-8')

    return content


def list_fs_directory(dirname: str = None) -> list[str]:
    """
    List the contents of a directory in the file store

    Parameters
    ----------
    dirname : str or None (default None)
        The directory name to list

    Returns
    -------
    files : list[str]
        The files in that directory
    """

    if dirname is None:
        dirname = DATA_DIRECTORY

    if not dirname.startswith(DATA_DIRECTORY):
        dirname = os.path.join(
            DATA_DIRECTORY,
            dirname.lstrip('/').strip()
        )

    if not os.path.isdir(dirname):
        raise TypeError('No directory found')

    return os.listdir(dirname)


class PredictRequest(BaseModel):
    data: list
    predict_function: str = 'predict'
    dtype: str = None
    params: dict = None
    convert_to_numpy: bool = True


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
    file_bytes: str
    overwrite: bool = False


class DataDownloadRequest(BaseModel):
    filename: str


class VariableSetRequest(BaseModel):
    variable_name: str
    value: str | int | float | bool | dict | list
    overwrite: bool = False


class VariableDownloadRequest(BaseModel):
    variable_name: str | int | float | bool | dict | list


class VariableDeleteRequest(BaseModel):
    variable_name: str


class VerifyPasswordInfo(BaseModel):
    username: str
    password: str


class DataListRequest(BaseModel):
    directory: str | None = None
