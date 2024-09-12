# Perform platform admin tasks
from .MLILException import MLILException
from .endpoints import RESET_ENDPOINT
import requests


def _reset_platform(
    url: str,
    creds: dict
):
    """
    NOT MEANT TO BE CALLED BY THE END USER

    Resets the MLIL platform

    Parameters
    ----------
    url: str
        String containing the URL of your deployment of the platform.
    creds:
        Dictionary that must contain keys "username" and "key", and associated values.
    """

    url = f"{url}/{RESET_ENDPOINT}"

    with requests.Session() as sess:
        resp = sess.get(
            url,
            auth=(creds['username'], creds['key']),
        )
    if not resp.ok:
        raise MLILException(str(resp.json()))
    return resp
