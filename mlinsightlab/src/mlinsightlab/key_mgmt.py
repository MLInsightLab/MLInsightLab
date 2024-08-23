# Manage authentication keys
from .MLILException import MLILException
from .endpoints import NEW_API_KEY_ENDPOINT
import requests


def _create_api_key(
    url: str,
    creds: dict,
    username: str
):
    """
    NOT MEANT TO BE CALLED BY THE END USER

    Create a new API key for a user.
    Called within the MLIL_client class.

    Parameters
    ----------
    url: str
        String containing the URL of your deployment of the platform.
    creds:
        Dictionary that must contain keys "username" and "key", and associated values.
    username: str
        The user's display name and login credential
    role: str
        The role to be given to the user
    password: str
        Password for user login
    """

    json_data = {
        'username': username
    }

    url = f"{url}/{NEW_API_KEY_ENDPOINT}"

    with requests.Session() as sess:
        resp = sess.put(
            url,
            auth=(creds['username'], creds['key']),
            json=json_data
        )
    if not resp.ok:
        raise MLILException(str(resp.json()))
    return resp
