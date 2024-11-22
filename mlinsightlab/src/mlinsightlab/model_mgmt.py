# Helper functions to manage and interat with MLFlow models
from .MLILException import MLILException
from typing import Union, List, Optional
from .endpoints import LOAD_MODEL_ENDPOINT, LIST_MODELS_ENDPOINT, UNLOAD_MODEL_ENDPOINT, PREDICT_ENDPOINT
import pandas as pd
import requests


def _load_model(
    url: str,
    creds: dict,
    model_name: str,
    model_flavor: str,
    model_version_or_alias: str,
    load_request: dict
):
    """
    NOT MEANT TO BE CALLED BY THE END USER

    Loads a saved model into memory.
    Called within the MLILClient class.

    Parameters
    ----------
    url: str
        String containing the URL of your deployment of the platform.
    creds: dict
        Dictionary that must contain keys "username" and "key", and associated values.
    model_name: str
        The name of the model to load
    model_flavor: str
        The flavor of the model, e.g. "transformers", "pyfunc", etc.
    model_version_or_alias: str
        The version of the model that you wish to load (from MLFlow).
    load_request : dict = None
        A dictionary containing additional parameters for loading the model,
        including 'requirements', 'quantization_kwargs', and 'kwargs'.
    """

    url = f"""{
        url}/{LOAD_MODEL_ENDPOINT}/{model_name}/{model_flavor}/{model_version_or_alias}"""

    with requests.Session() as sess:
        resp = sess.post(
            url,
            auth=(creds['username'], creds['key']),
            json=load_request
        )
    if not resp.ok:
        raise MLILException(str(resp.json()))
    return resp


def _list_models(
    url: str,
    creds: dict
):
    """
    NOT MEANT TO BE CALLED BY THE END USER

    Lists all *loaded* models. To view unloaded models, check the MLFlow UI.
    Called within the MLILClient class.

    Parameters
    ----------
    url: str
        String containing the URL of your deployment of the platform.
    creds:
        Dictionary that must contain keys "username" and "key", and associated values.
    """

    url = f"{url}/{LIST_MODELS_ENDPOINT}"

    with requests.Session() as sess:
        resp = sess.get(
            url,
            auth=(creds['username'], creds['key']),
        )
    if not resp.ok:
        raise MLILException(str(resp.json()))
    return resp


def _unload_model(
    url: str,
    creds: dict,
    model_name: str,
    model_flavor: str,
    model_version_or_alias: str
):
    """
    NOT MEANT TO BE CALLED BY THE END USER

    Removes a loaded model from memory.
    Called within the MLILClient class.

    Parameters
    ----------
    url: str
        String containing the URL of your deployment of the platform.
    creds:
        Dictionary that must contain keys "username" and "key", and associated values.
    model_name: str
        The name of the model to unload.
    model_flavor: str
        The flavor of the model, e.g. "transformers", "pyfunc", etc.
    model_version_or_alias: str
        The version of the model that you wish to unload (from MLFlow).
    """

    json_data = {
        'model_name': model_name,
        'model_flavor': model_flavor,
        'model_version_or_alias': model_version_or_alias
    }

    url = f"""{
        url}/{UNLOAD_MODEL_ENDPOINT}/{model_name}/{model_flavor}/{model_version_or_alias}"""

    with requests.Session() as sess:
        resp = sess.delete(
            url,
            auth=(creds['username'], creds['key'])
        )
    if not resp.ok:
        raise MLILException(str(resp.json()))
    return resp


def _predict(
    url: str,
    creds: dict,
    model_name: str,
    model_flavor: str,
    model_version_or_alias: str,
    data: Union[list, int, float, str],
    convert_to_numpy: bool = True,
    predict_function: str = "predict",
    dtype: str = None,
    params: Optional[dict] = None
):
    """
    NOT MEANT TO BE CALLED BY THE END USER

    Calls the 'predict' function of the specified MLFlow model.

    Called within the MLILClient class.

    Parameters
    ----------
    url: str
        String containing the URL of your deployment of the platform.
    creds:
        Dictionary that must contain keys "username" and "key", and associated values.
    model_name: str
        The name of the model to be invoked.
    model_flavor: str
        The flavor of the model, e.g. "transformers", "pyfunc", etc.
    model_version_or_alias: str
        The version of the model that you wish to invoke (from MLFlow).
    data: varied types
        The input data for prediction. Can be a single value or list of values
    convert_to_numpy: bool = True
        Whether or not to convert inputs to a NumPy array.
    predict_function: str, optional
        The name of the prediction function to call. Default is "predict".
    dtype: str, optional
        The data type of the input
    params: dict, optional
        Additional parameters for the prediction.
    """
    if isinstance(data, str):
        data = [data]

    json_data = {
        "data": data,
        "predict_function": predict_function,
        "params": params if params else {},
        "convert_to_numpy": convert_to_numpy
    }

    if dtype:
        json_data.update({'dtype': dtype})

    url = f"""{
        url}/{PREDICT_ENDPOINT}/{model_name}/{model_flavor}/{model_version_or_alias}"""

    with requests.Session() as sess:
        resp = sess.post(
            url,
            auth=(creds['username'], creds['key']),
            json=json_data
        )
    if not resp.ok:
        raise MLILException(str(resp.json()))
    return resp
