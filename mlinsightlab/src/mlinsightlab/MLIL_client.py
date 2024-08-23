from typing import Union,List, Optional
import pandas as pd
import warnings
import requests
import getpass
import json
import os

from .endpoints import ENDPOINTS 

from .MLIL_APIException import MLIL_APIException
from .user_mgmt import _create_user, _delete_user, _verify_password, _issue_new_password, _get_user_role, _update_user_role, _list_users
from .key_mgmt import  _create_api_key 
from .model_mgmt import _load_model, _unload_model, _list_models, _predict

class MLIL_client:
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
        if auth is None:
            auth = self.login()
        
        self.username = auth.get('username')
        self.api_key = auth.get('key')
        self.url = auth.get('url') or url  # Use URL from auth if available, otherwise use the provided url
        
        if not self.username or not self.api_key:
            raise ValueError("Both username and API key are required.")
        
        if not self.url:
            raise ValueError("You must provide the base URL of your instance of the platform.")
        
        self.creds = {'username': self.username, 'key': self.api_key}

    def login(self):
        # Implement login logic here
        # This method should return a dictionary with 'username', 'key', and 'url'
        # For example:
        username = input("Enter username: ")
        api_key = getpass.getpass("Enter API key: ")
        url = input("Enter platform URL: ")
        return {'username': username, 'key': api_key, 'url': url}
       
    ###########################################################################
    ########################## User Operations ################################
    ###########################################################################
    
    def create_user(
        self, 
        role: str or None,
        api_key: str or None,
        password: str or None,
        username: str,
        url: str = None, 
        creds: dict = None,
        verbose: bool = False
        ):
        """
        Create a user within the platform. 

        >>> import mlil
        >>> client = mlil.MLIL_client()
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
                print(f'Something went wrong, request returned a satus code {resp.status_code}')
            
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
        >>> client = mlil.MLIL_client()
        >>> client.delete_user()

        Parameters
        ----------
        url: str
            String containing the URL of your deployment of the platform.
        creds: 
            Dictionary that must contain keys "username" and "key", and associated values.
        username: str
            The display name of the user to be deleted
        """
        if url is None:
            url = self.url
        if creds is None:
            creds = self.creds
        
        resp = _delete_user(url, creds, username)
        
        if verbose:
            if resp.status_code == 200:
                print(f'user {username} is now off the platform! Good riddance!')
            else:
                print(f'Something went wrong, request returned a satus code {resp.status_code}')
            
        return resp.json()
    
    def verify_password(
        self,
        url: str = None,
        creds: dict = None,
        username: str = None,
        password: str = None,
        verbose: bool = False
        ):
        """
        Verify a user's password. 

        >>> import mlil
        >>> client = mlil.MLIL_client()
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
            Password for user login
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
                print(f'Your password "{password}" is verified. Congratulations!')
            else:
                print(f'Something went wrong, request returned a satus code {resp.status_code}')
            
        return resp.json()
    
    def issue_new_password(
        self,
        new_password: str,
        url: str = None,
        creds: dict = None,
        username: str = None,
        verbose: bool = False
        ):
        """
        Create a new a password for an existing user. 

        >>> import mlil
        >>> client = mlil.MLIL_client()
        >>> client.issue_new_password()

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
        if url is None:
            url = self.url
        if creds is None:
            creds = self.creds
        if username is None:
            username = self.username
            
        resp = _issue_new_password(url, creds, username, new_password=new_password)

        if verbose:
            if resp.status_code == 200:
                print(f'Your new password "{password}" is created. Try not to lose this one!')
            else:
                print(f'Something went wrong, request returned a satus code {resp.status_code}')
            
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
        >>> client = mlil.MLIL_client()
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
            
        resp = _get_user_role(url, creds, username = username)
        
        if verbose:
            if resp.status_code == 200:
                print(f'User {username} works here, and they sound pretty important.')
            else:
                print(f'Something went wrong, request returned a satus code {resp.status_code}')
            
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
        >>> client = mlil.MLIL_client()
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
            
        resp = _update_user_role(url, creds, username=username, new_role=new_role)

        if verbose:
            if resp.status_code == 200:
                print(f'User {username} now has the role {new_role}.')
            else:
                print(f'Something went wrong, request returned a satus code {resp.status_code}')
            
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
        >>> client = mlil.MLIL_client()
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
                print(f'Something went wrong, request returned a satus code {resp.status_code}')
            
        return resp.json()

    ###########################################################################
    ########################## Key Operations ################################
    ###########################################################################
    
    def issue_api_key(
        self,
        username: str,
        url: str = None,
        creds: dict = None,
        verbose: bool = False
        ):
        """
        Create a new API key for a user. 

        Parameters
        ----------
        url: str
            String containing the URL of your deployment of the platform.
        creds: 
            Dictionary that must contain keys "username" and "key", and associated values.
        username: str
            The display name of the user for whom you're creating a key.
        role: str
            The role to be given to the user
        password: str
            Password for user login
        """
        if url is None:
            url = self.url
        if creds is None:
            creds = self.creds

        resp = _create_api_key(url, creds, username = username)

        if verbose:
            if resp.status_code == 200:
                print(f'{username} can now get back to work.')
            else:
                print(f'Something went wrong, request returned a satus code {resp.status_code}')
            
        return resp.json()

    ###########################################################################
    ########################## Model Operations ###############################
    ###########################################################################
    
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
        >>> client = mlil.MLIL_client()
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
            model_name = model_name,
            model_flavor=model_flavor,
            model_version_or_alias=model_version_or_alias
            )

        if verbose:
            if resp.status_code == 200:
                print(f'{model_name} is locked and loaded.')
            else:
                print(f'Something went wrong, request returned a satus code {resp.status_code}')
            
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
        >>> client = mlil.MLIL_client()
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
                print(f'Something went wrong, request returned a satus code {resp.status_code}')
            
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
        >>> client = mlil.MLIL_client()
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
                print(f'These are your models, Simba, as far as the eye can see.')
            else:
                print(f'Something went wrong, request returned a satus code {resp.status_code}')
            
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
        >>> client = mlil.MLIL_client()
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
                print(f'Something went wrong, request returned a satus code {resp.status_code}')
            
        return resp.json() 