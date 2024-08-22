# Manage users and credentials
from .MLIL_APIException import MLIL_APIException
from .endpoints import ENDPOINTS
import requests

def _create_user(
    url: str,
    creds: dict,
    username: str,
    role: str,
    api_key: str or None,
    password: str or None
    ):
    """
    NOT MEANT TO BE CALLED BY THE END USER

    Create a user within the platform. 
    Called within the MLIL_client class.

    Parameters
    ----------
    url: str
        String containing the URL of your deployment of the platform.
    cred: 
        Dictionary that must contain keys "username" and "key", and associated values.
    username: str
        The user's display name and login credential
    role: str
        The role to be given to the user
    api_key: str or NULL
        An API key for the new user
    password: str or NULL
        Password for user login
    """

    json_data = {
        'username' : username,
        'role' : role
    }

    if api_key:
        json_data['api_key'] = api_key
    if password:
        json_data['password'] = password

    url = f"{url}/{ENDPOINTS['create_user']}"

    with requests.Session() as sess:
        resp = sess.post(
            url,
            auth=(creds['username'], creds['key']),
            json=json_data
        )
    if not resp.ok:
        raise MLIL_APIException(resp.json())
    return resp
def _delete_user(
    url: str,
    creds: dict,
    username: str
    ):
    """
    NOT MEANT TO BE CALLED BY THE END USER

    Delete a user within the platform. 
    Called within the MLIL_client class.

    Parameters
    ----------
    url: str
        String containing the URL of your deployment of the platform.
    creds: 
        Dictionary that must contain keys "username" and "key", and associated values.
    username: str
        The user's display name and login credential
    """

    json_data = {
        'username' : username
    }

    url = f"{url}/{ENDPOINTS['delete_user']}"

    with requests.Session() as sess:
        resp = sess.delete(
            url,
            auth=(creds['username'], creds['key']),
            json=json_data
        )
    if not resp.ok:
        raise MLIL_APIException(resp.json())
    return resp
def _verify_password(
    url: str,
    creds: dict,
    username: str,
    password: str
    ):
    """
    NOT MEANT TO BE CALLED BY THE END USER

    Verify a user's password. 
    Called within the MLIL_client class.

    Parameters
    ----------
    url: str
        String containing the URL of your deployment of the platform.
    creds: 
        Dictionary that must contain keys "username" and "key", and associated values.
    username: str
        The user's display name and login credential
    password: str
        Password for user login
    """

    json_data = {
        'username' : username,
        'password' : password
    }

    url = f"{url}/{ENDPOINTS['verify_password']}"

    with requests.Session() as sess:
        resp = sess.post(
            url,
            auth=(creds['username'], creds['key']),
            json=json_data
        )
    if not resp.ok:
        raise MLIL_APIException(resp.json())
    return resp
def _issue_new_password(
    url: str,
    creds: dict,
    username: str,
    new_password: str
    ):
    """
    NOT MEANT TO BE CALLED BY THE END USER

    Create a new a password for an existing user. 
    Called within the MLIL_client class.

    Parameters
    ----------
    url: str
        String containing the URL of your deployment of the platform.
    creds: 
        Dictionary that must contain keys "username" and "key", and associated values.
    username: str
        The user's display name and login credential
    new_password: str
        New password for user authentication
    """

    json_data = {
        'username' : username,
        'new_password' : new_password
    }

    url = f"{url}/{ENDPOINTS['issue_new_password']}"

    with requests.Session() as sess:
        resp = sess.put(
            url,
            auth=(creds['username'], creds['key']),
            json=json_data
        )
    if not resp.ok:
        raise MLIL_APIException(resp.json())
    return resp
def _get_user_role(
    url: str,
    creds: dict,
    username: str,
    ):
    """
    NOT MEANT TO BE CALLED BY THE END USER

    Check a user's role. 
    Called within the MLIL_client class.

    Parameters
    ----------
    url: str
        String containing the URL of your deployment of the platform.
    creds: 
        Dictionary that must contain keys "username" and "key", and associated values.
    username: str
        The user's display name and login credential.
    """

    json_data = {
        'username' : username
    }

    url = f"{url}/{ENDPOINTS['get_user_role']}"

    with requests.Session() as sess:
        resp = sess.get(
            url,
            auth=(creds['username'], creds['key']),
            json=json_data
        )
    if not resp.ok:
        raise MLIL_APIException(resp.json())
    return resp
def _update_user_role(
    url: str,
    creds: dict,
    username: str,
    new_role: str
    ):
    """
    NOT MEANT TO BE CALLED BY THE END USER

    Update a user's role. 
    Called within the MLIL_client class.

    Parameters
    ----------
    url: str
        String containing the URL of your deployment of the platform.
    creds: 
        Dictionary that must contain keys "username" and "key", and associated values.
    username: str
        The user's display name and login credential
    new_role: str
        New role to attribute to the specified user
    """

    json_data = {
        'username' : username,
        'new_role' : new_role
    }

    url = f"{url}/{ENDPOINTS['update_user_role']}"

    with requests.Session() as sess:
        resp = sess.put(
            url,
            auth=(creds['username'], creds['key']),
            json=json_data
        )
    if not resp.ok:
        raise MLIL_APIException(resp.json())
    return resp
def _list_users(
    url: str,
    creds: dict,
    ):
    """
    NOT MEANT TO BE CALLED BY THE END USER

    Update a user's role. 
    Called within the MLIL_client class.

    Parameters
    ----------
    url: str
        String containing the URL of your deployment of the platform.
    creds: 
        Dictionary that must contain keys "username" and "key", and associated values.    
    """

    url = f"{url}/{ENDPOINTS['list_users']}"

    with requests.Session() as sess:
        resp = sess.get(
            url,
            auth=(creds['username'], creds['key']),
        )
    if not resp.ok:
        raise MLIL_APIException(resp.json())
    return resp