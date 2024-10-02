# Helper functions to manage and interat with MLFlow models
from .MLILException import MLILException
from .endpoints import FILE_UPLOAD, FILE_DOWNLOAD, GET_VARIABLE, LIST_VARIABLES, SET_VARIABLE, DELETE_VARIABLE
from typing import Any
import requests

def _upload_file(
    url: str,
    creds: dict,
    file_path: str,
    file_name: str,
    overwrite: bool = False
):
    """
    NOT MEANT TO BE CALLED BY THE END USER

    Uploads a file to the MLIL platform's data store.
    Called within the MLILClient class.

    Parameters
    ----------
    url: str
        String containing the URL of your deployment of the platform.
    creds: dict
        Dictionary that must contain keys "username" and "key", and associated values.
    file_path: str
        Path to the file to be uploaded to MLIL.
    file_name: str
        The name to give your file in the MLIL datastore.
    overwrite: bool
        Whether or not to overwrite the file, if a file of the same name
        already exists.
    """

    url = f"{url}/{FILE_UPLOAD}"

    with open(file_path, 'rb') as f:
        file_bytes = f.read()

    json_data = {
        'filename': file_name,
        'file_bytes': str(file_bytes),
        'overwrite': overwrite
    }

    with requests.Session() as sess:
        resp = sess.post(
            url,
            auth=(creds['username'], creds['key']),
            json=json_data
        )

    if not resp.ok:
        raise MLILException(str(resp.json()))
    return resp

def _download_file(
    url: str,
    creds: dict,
    file_name: str
):
    """
    NOT MEANT TO BE CALLED BY THE END USER

    Downloads a file from the MLIL platform's data store as a byte string.
    Called within the MLILClient class.

    Parameters
    ----------
    url: str
        String containing the URL of your deployment of the platform.
    creds: dict
        Dictionary that must contain keys "username" and "key", and associated values.
    file_name: str
        The name of the file to download.
    """

    url = f"{url}/{FILE_DOWNLOAD}"

    json_data = {
        'filename': file_name
    }

    with requests.Session() as sess:
        resp = sess.post(
            url,
            auth=(creds['username'], creds['key']),
            json=json_data
        )

    if not resp.ok:
        raise MLILException(str(resp.json()))
    return resp

def _get_variable(
    url: str,
    creds: dict,
    variable_name: str
):
    """
    NOT MEANT TO BE CALLED BY THE END USER

    Retrieve a variable from the MLIL variable store.
    Called within the MLILClient class.

    Parameters
    ----------
    url: str
        String containing the URL of your deployment of the platform.
    creds: dict
        Dictionary that must contain keys "username" and "key", and associated values.
    variable_name: str
        The name of the variable to access.
    """

    url = f"{url}/{GET_VARIABLE}"

    json_data = {
        'variable_name': variable_name,
        'username' : creds['username']
    }

    with requests.Session() as sess:
        resp = sess.post(
            url,
            auth=(creds['username'], creds['key']),
            json=json_data
        )

    if not resp.ok:
        raise MLILException(str(resp.json()))
    return resp

def _list_variables(
    url: str,
    creds: dict
):
    """
    NOT MEANT TO BE CALLED BY THE END USER

    Lists all variables associated with a user.
    Called within the MLILClient class.

    Parameters
    ----------
    url: str
        String containing the URL of your deployment of the platform.
    creds: dict
        Dictionary that must contain keys "username" and "key", and associated values.
    """

    url = f"{url}/{LIST_VARIABLES}"

    json_data = {
        'username' : creds['username']
    }

    with requests.Session() as sess:
        resp = sess.post(
            url,
            auth=(creds['username'], creds['key']),
            json=json_data
        )

    if not resp.ok:
        raise MLILException(str(resp.json()))
    return resp

def _set_variable(
    url: str,
    creds: dict,
    variable_name: str,
    value: Any,
    overwrite: bool = False
):
    """
    NOT MEANT TO BE CALLED BY THE END USER

    Creates a variable within the MLIL variable store.
    Called within the MLILClient class.

    Parameters
    ----------
    url: str
        String containing the URL of your deployment of the platform.
    creds: dict
        Dictionary that must contain keys "username" and "key", and associated values.
    variable_name: str
        The name of the variable to set.
    overwrite: bool = False
        Whether to overwrite any variables that currently exist in MLIL and have the same name.
    value: Any
        Your variable. Can be of type string | integer | number | boolean | object | array<any>.
    """

    url = f"{url}/{SET_VARIABLE}"

    json_data = {
        'variable_name': variable_name,
        'value' : value,
        'overwrite': overwrite,
        'username' : creds['username']
    }

    with requests.Session() as sess:
        resp = sess.post(
            url,
            auth=(creds['username'], creds['key']),
            json=json_data
        )

    if not resp.ok:
        raise MLILException(str(resp.json()))
    return resp

def _delete_variable(
    url: str,
    creds: dict,
    variable_name: str
):
    """
    NOT MEANT TO BE CALLED BY THE END USER

    Removes a variable from the MLIL variable store.
    Called within the MLILClient class.

    Parameters
    ----------
    url: str
        String containing the URL of your deployment of the platform.
    creds: dict
        Dictionary that must contain keys "username" and "key", and associated values.
    variable_name: str
        The name of the variable to delete.
    """

    url = f"{url}/{DELETE_VARIABLE}"

    json_data = {
        'variable_name': variable_name,
        'username' : creds['username']
    }

    with requests.Session() as sess:
        resp = sess.post(
            url,
            auth=(creds['username'], creds['key']),
            json=json_data
        )

    if not resp.ok:
        raise MLILException(str(resp.json()))
    return resp