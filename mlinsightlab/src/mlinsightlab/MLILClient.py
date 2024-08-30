from typing import Union, List, Optional
from pathlib import Path
import pandas as pd
import warnings
import requests
import getpass
import json
import os

#from .endpoints import *

from .MLILException import MLILException
from .user_mgmt import _create_user, _delete_user, _verify_password, _issue_new_password, _get_user_role, _update_user_role, _list_users
from .key_mgmt import _create_api_key
from .model_mgmt import _load_model, _unload_model, _list_models, _predict


class MLILClient:
    """
    Client for interacting with the ML Insights Lab (MLIL) Platform
    """

    def __init__(
        self,
        auth: dict = None,
        url: str = None
    ):
        """
        Initializes the class and sets configuration variables.
        MVP design is to pass in dict of with the following k-v pairs:
        {'username' : username, 'key' : your api key}
        Parameters
        ----------
        """

        self.config_path = Path((f"{Path.home()}/.mlil/config.json"))

        if auth is None:
            auth = self.login()

        self.username = auth.get('username')
        self.api_key = auth.get('key')
        self.url = auth.get('url') or url
        self.password = auth.get('password')

        if not self.username or not self.api_key or not self.password:
            raise ValueError("You must provide your username, password, and API key.")

        if not self.url:
            raise ValueError(
                "You must provide the base URL of your instance of the platform.")

        self.creds = {'username': self.username, 'key': self.api_key}

    """
    ###########################################################################
    ########################## Login Operations ################################
    ###########################################################################
    """

    def login(self):
        if self.config_path.exists():
            use_stored = input("Found stored credentials. Use them? (y/n): ").lower() == 'y'
            if use_stored:
                return self.load_stored_credentials()

        url = input("Enter platform URL: ")
        username = input("Enter username: ")
        password = getpass.getpass("Enter password: ")
        api_key = getpass.getpass("Enter API key (or leave blank to generate new): ")

        if not api_key:
            generate_new = input("Generate new API key? (y/n): ").lower() == 'y'
            if generate_new:
                api_key = self.issue_api_key(username=username, password=password, url=url)

        resp = self.verify_password(url=url, creds={"username": username, "key": api_key}, username=username, password=password)

        if resp.ok:
            print(f"User verified...welcome {username}!")
        else:
            print('User not verified.')
            raise MLILException(str(resp.json()))

        auth = {'username': username, 'key': api_key, 'url': url, 'password': password}
        self.save_credentials(auth)
        return auth
    
    def load_stored_credentials(self):
        """
        Loads stored credentials from the config file.
        """
        with open(self.config_path, 'r') as f:
            return json.load(f)
    
    def save_credentials(self, auth):
        """
        Saves credentials to the config file.
        """
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(auth, f)
    
    def purge_credentials(self):
        """
        Enables user to delete the file containing cached credentials.
        """
        purge_creds = input("Are you sure you want to delete your saved credentials? This cannot be undone. (y/n): ").lower() == 'y'
        if purge_creds:
            if os.path.exists(self.config_path):
                os.remove(self.config_path)
            else:
                print("No credentials file found.")
    """
    ###########################################################################
    ########################## User Operations ################################
    ###########################################################################
    """

    def create_user(
        self,
        role: str | None,
        api_key: str | None,
        password: str | None,
        username: str,
        url: str = None,
        creds: dict = None,
        verbose: bool = False
    ):
        """
        Create a user within the platform.

        >>> import mlil
        >>> client = mlil.MLILClient()
        >>> client.create_user()

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
        api_key: str or NULL
            An API key for the new user
        password: str or NULL
            Password for user login
        """
        if url is None:
            url = self.url
        if creds is None:
            creds = self.creds

        resp = _create_user(url, creds, username, role, api_key, password)

        if verbose:
            if resp.status_code == 200:
                print(f'user {username} is now on the platform! Go say hi!')
            else:
                print(
                    f'Something went wrong, request returned a satus code {resp.status_code}')

        return resp.json()

    def delete_user(
        self,
        username: str,
        url: str = None,
        creds: dict = None,
        verbose: bool = False
    ):
        """
        Delete a user of the platform.

        >>> import mlil
        >>> client = mlil.MLILClient()
        >>> client.delete_user()

        Parameters
        ----------
        url: str
            String containing the URL of your deployment of the platform.
        creds:
            Dictionary that must contain keys "username" and "key", and associated values.
        username: str
            The display name of the user to be deleted.
        """
        if url is None:
            url = self.url
        if creds is None:
            creds = self.creds

        resp = _delete_user(url, creds, username)

        if verbose:
            if resp.status_code == 200:
                print(
                    f'user {username} is now off the platform! Good riddance!')
            else:
                print(
                    f'Something went wrong, request returned a satus code {resp.status_code}')

        return resp.json()

    def verify_password(
        self,
        password: str,
        url: str = None,
        creds: dict = None,
        username: str = None,
        verbose: bool = False
    ):
        """
        Verify a user's password.

        >>> import mlil
        >>> client = mlil.MLILClient()
        >>> client.verify_password()

        Parameters
        ----------
        url: str
            String containing the URL of your deployment of the platform.
        creds:
            Dictionary that must contain keys "username" and "key", and associated values.
        username: str
            The user's display name and login credential
        password: str
            Password for user login.
        """
        if url is None:
            url = self.url
        if creds is None:
            creds = self.creds
        if username is None:
            username = self.username

        resp = _verify_password(url, creds, username, password)

        if verbose:
            if resp.status_code == 200:
                print(
                    f'Your password "{password}" is verified. Congratulations!')
            else:
                print(
                    f'Something went wrong, request returned a satus code {resp.status_code}')

        return resp

    def issue_new_password(
        self,
        new_password: str,
        overwrite_password: bool = True,
        url: str = None,
        creds: dict = None,
        username: str = None,
        verbose: bool = False
    ):
        """
        Create a new a password for an existing user.

        >>> import mlil
        >>> client = mlil.MLILClient()
        >>> client.issue_new_password()

        Parameters
        ----------

        new_password: str
            New password for user authentication.
            It must have:
            - At least 8 characters
            - At least 1 uppercase character
            - At least 1 lowercase character
        overwrite_password: bool = True
            Whether or not to overwrite the password in the config file. Defaults to True.
        url: str
            String containing the URL of your deployment of the platform.
        creds:
            Dictionary that must contain keys "username" and "key", and associated values.
        username: str
            The user's display name and login credential
        """
        if url is None:
            url = self.url
        if creds is None:
            creds = self.creds
        if username is None:
            username = self.username

        resp = _issue_new_password(
            url, creds, username, new_password=new_password)
        
        if resp.ok:
            self.password = new_password
        else:
            return MLILException(str(resp.json()))
        
        if overwrite_password:
            auth = {'username': self.username, 'key': self.api_key, 'url': url, 'password': new_password}
            print(f'Your password has been overwritten.')
            self.save_credentials(auth)

        if verbose:
            if resp.status_code == 200:
                print(
                    f'Your new password "{new_password}" is created. Try not to lose this one!')
            else:
                print(
                    f'Something went wrong, request returned a satus code {resp.status_code}')

        return resp.json()

    def get_user_role(
        self,
        username: str,
        url: str = None,
        creds: dict = None,
        verbose: bool = False
    ):
        """
        Check a user's role.

        >>> import mlil
        >>> client = mlil.MLILClient()
        >>> client.get_user_role()

        Parameters
        ----------
        url: str
            String containing the URL of your deployment of the platform.
        creds:
            Dictionary that must contain keys "username" and "key", and associated values.
        username: str
            The user's display name and login credential.
        """
        if url is None:
            url = self.url
        if creds is None:
            creds = self.creds

        resp = _get_user_role(url, creds, username=username)

        if verbose:
            if resp.status_code == 200:
                print(
                    f'User {username} works here, and they sound pretty important.')
            else:
                print(
                    f'Something went wrong, request returned a satus code {resp.status_code}')

        return resp.json()

    def update_user_role(
        self,
        username: str,
        new_role: str,
        url: str = None,
        creds: dict = None,
        verbose: bool = False
    ):
        """
        Update a user's role.

        >>> import mlil
        >>> client = mlil.MLILClient()
        >>> client.update_user_role()

        Parameters
        ----------
        url: str
            String containing the URL of your deployment of the platform.
        creds: dict
            Dictionary that must contain keys "username" and "key", and associated values.
        username: str
            The user's display name and login credential
        new_role: str
            New role to attribute to the specified user
        """
        if url is None:
            url = self.url
        if creds is None:
            creds = self.creds

        resp = _update_user_role(
            url, creds, username=username, new_role=new_role)

        if verbose:
            if resp.status_code == 200:
                print(f'User {username} now has the role {new_role}.')
            else:
                print(
                    f'Something went wrong, request returned a satus code {resp.status_code}')

        return resp.json()

    def list_users(
        self,
        url: str = None,
        creds: dict = None,
        verbose: bool = False
    ):
        """
        Update a user's role.

        >>> import mlil
        >>> client = mlil.MLILClient()
        >>> client.create_user()

        Parameters
        ----------
        url: str
            String containing the URL of your deployment of the platform.
        creds: dict
            Dictionary that must contain keys "username" and "key", and associated values.
        """
        if url is None:
            url = self.url
        if creds is None:
            creds = self.creds

        resp = _list_users(url, creds)

        if verbose:
            if resp.status_code == 200:
                print(f'Gaze upon your co-workers in wonder!')
            else:
                print(
                    f'Something went wrong, request returned a satus code {resp.status_code}')

        return resp.json()

    """
    ###########################################################################
    ########################## Key Operations ################################
    ###########################################################################
    """

    def issue_api_key(
        self,
        username: str,
        password: str,
        url: str = None,
        creds: dict = None,
        overwrite_api_key: bool = True,
        verbose: bool = False
    ):
        """
        Create a new API key for a user.

        Parameters
        ----------
        url: str
            String containing the URL of your deployment of the platform.
        username: str
            The display name of the user for whom you're creating a key.
        password: str
            Password for user verification.
        overwrite_api_key: bool = True
            Overwrites the API key stored in the credentials cached in config.js
        """
        if url is None:
            url = self.url
        if username is None:
            username = self.username
        if password is None:
            password = self.password

        resp = _create_api_key(url, username=username, password = password)

        self.api_key = resp

        if overwrite_api_key:
            auth = {'username': username, 'key': self.api_key, 'url': url, 'password': password}
            self.save_credentials(auth)

        if verbose:
            if resp.status_code == 200:
                print(f'New key granted. Please only use this power for good.')
            else:
                print(
                    f'Something went wrong, request returned a satus code {resp.status_code}')

        return resp.json()

    """
    ###########################################################################
    ########################## Model Operations ###############################
    ###########################################################################
    """

    def load_model(
        self,
        model_name: str,
        model_flavor: str,
        model_version_or_alias: str,
        url: str = None,
        creds: dict = None,
        verbose: bool = False
    ):
        """
        Loads a saved model into memory within the platform.

        >>> import mlil
        >>> client = mlil.MLILClient()
        >>> client.load_model(model_name, model_flavor, model_version_or_alias)

        Parameters
        ----------
        url: str
            String containing the URL of your deployment of the platform.
        creds:
            Dictionary that must contain keys "username" and "key", and associated values.
        model_name: str
            The name of the model to load
        model_flavor: str
            The flavor of the model, e.g. "transformers", "pyfunc", etc.
        model_version_or_alias: str
            The version of the model that you wish to load (from MLFlow).
        """

        if url is None:
            url = self.url
        if creds is None:
            creds = self.creds

        resp = _load_model(url,
                           creds,
                           model_name=model_name,
                           model_flavor=model_flavor,
                           model_version_or_alias=model_version_or_alias
                           )

        if verbose:
            if resp.status_code == 200:
                print(f'{model_name} is loading. This may take a few minutes, so go grab a doughnut. Mmmmmmm…doughnuts…')
            else:
                print(
                    f'Something went wrong, request returned a satus code {resp.status_code}')

        return resp.json()

    def list_models(
        self,
        url: str = None,
        creds: dict = None,
        verbose: bool = False
    ):
        """
        Lists all *loaded* models. To view unloaded models, check the MLFlow UI.

        >>> import mlil
        >>> client = mlil.MLILClient()
        >>> client.create_user()

        Parameters
        ----------
        url: str
            String containing the URL of your deployment of the platform.
        creds:
            Dictionary that must contain keys "username" and "key", and associated values.
        """

        if url is None:
            url = self.url
        if creds is None:
            creds = self.creds

        resp = _list_models(url=url, creds=creds)

        if verbose:
            if resp.status_code == 200:
                print(f'These are your models, Simba, as far as the eye can see.')
            else:
                print(
                    f'Something went wrong, request returned a satus code {resp.status_code}')

        return resp.json()

    def unload_model(
        self,
        model_name: str,
        model_flavor: str,
        model_version_or_alias: str,
        url: str = None,
        creds: dict = None,
        verbose: bool = False
    ):
        """
        Removes a loaded model from memory.

        >>> import mlil
        >>> client = mlil.MLILClient()
        >>> client.unload_model()

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

        if url is None:
            url = self.url
        if creds is None:
            creds = self.creds

        resp = _unload_model(
            url,
            creds,
            model_name=model_name,
            model_flavor=model_flavor,
            model_version_or_alias=model_version_or_alias
        )

        if verbose:
            if resp.status_code == 200:
                print(f'{model_name} has been unloaded from memory.')
            else:
                print(
                    f'Something went wrong, request returned a satus code {resp.status_code}')

        return resp.json()

    def predict(
        self,
        model_name: str,
        model_flavor: str,
        model_version_or_alias: str,
        data: Union[str, List[str]],
        predict_function: str = "predict",
        dtype: str = "string",
        params: Optional[dict] = None,
        url: str = None,
        creds: dict = None,
        verbose: bool = False
    ):
        """
        Calls the 'predict' function of the specified MLFlow model.

        >>> import mlil
        >>> client = mlil.MLILClient()
        >>> client.predict()

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
        data: Union[str, List[str]]
            The input data for prediction. Can be a single string or a list of strings.
        predict_function: str, optional
            The name of the prediction function to call. Default is "predict".
        dtype: str, optional
            The data type of the input. Default is "string".
        params: dict, optional
            Additional parameters for the prediction.
        """
        if url is None:
            url = self.url
        if creds is None:
            creds = self.creds

        resp = _predict(
            url=url,
            creds=creds,
            model_name=model_name,
            model_flavor=model_flavor,
            model_version_or_alias=model_version_or_alias,
            data=data,
            predict_function=predict_function,
            dtype=dtype,
            params=params
        )

        if verbose:
            if resp.status_code == 200:
                print(f'Sometimes I think I think')
            else:
                print(
                    f'Something went wrong, request returned a satus code {resp.status_code}')

        return resp.json()
