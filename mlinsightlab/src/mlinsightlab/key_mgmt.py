# Manage authentication keys
from .MLILException import MLILException
from .endpoints import NEW_API_KEY_ENDPOINT
import requests


def _create_api_key(
    url: str,
    username: str,
    password: str
):
    """
    NOT MEANT TO BE CALLED BY THE END USER

    Create a new API key for a user.
    Called within the MLILClient class.

    Parameters
    ----------
    url: str
        String containing the URL of your deployment of the platform.
    username: str
        The user's display name and login credential
    password: str
        Password for user verification
    """

    json_data = {
        'username': username
    }

    url = f"{url}/{NEW_API_KEY_ENDPOINT}/{username}"

    with requests.Session() as sess:
        resp = sess.put(
            url,
            auth=(username, password),
            json=json_data
        )
        if not resp.ok:
            raise MLILException(str(resp.json()))
    return resp
