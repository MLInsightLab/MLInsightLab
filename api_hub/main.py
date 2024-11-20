from fastapi.security import HTTPBasic, HTTPBasicCredentials, OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi import FastAPI, HTTPException, Depends, Body, BackgroundTasks
from datetime import datetime, timedelta, timezone
from fastapi.responses import RedirectResponse
from jose import JWTError, jwt
import numpy as np
import subprocess
import secrets
import signal
import string
import json
import os

from db_utils import setup_database, validate_user_key, validate_user_password, fcreate_user, fdelete_user, fissue_new_api_key, fissue_new_password, fget_user_role, fupdate_user_role, flist_users, SERVED_MODEL_CACHE_FILE
from utils import ALLOWED_MODEL_FLAVORS, PYFUNC_FLAVOR, SKLEARN_FLAVOR, TRANSFORMERS_FLAVOR, HUGGINGFACE_FLAVOR, VARIABLE_STORE_FILE, fload_model, load_models_from_cache, predict_model, upload_data_to_fs, download_data_from_fs, PredictRequest, LoadRequest, UserInfo, DataUploadRequest, DataDownloadRequest, VariableSetRequest, VariableDownloadRequest, VariableDeleteRequest, VerifyPasswordInfo, list_fs_directory, DataListRequest

# Set up variables for JWT authentication
SECRET_KEY = ''.join([secrets.choice(string.ascii_letters) for _ in range(32)])
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# The MLFlow tracking uri
MLFLOW_TRACKING_URI = os.environ['MLFLOW_TRACKING_URI']

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

# Set up the database
setup_database()

# Load the variable store
try:
    with open(VARIABLE_STORE_FILE, 'r') as f:
        variable_store = json.load(f)
except Exception:
    variable_store = {}


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


def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + \
            timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


# Initialize the app and Basic Auth
app = FastAPI()
security = HTTPBasic(auto_error=False)

# Function to verify user credentials


def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    """
    Verify a user's API key credentials
    """
    if not credentials:
        return None
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


def verify_jwt_token(token: str = Depends(oauth2_scheme)):
    if not token:
        return None

    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        if username is None or role is None:
            raise credentials_exception
        return {"username": username, "role": role}
    except JWTError:
        raise credentials_exception


def verify_credentials_or_token(
    api_key: HTTPBasicCredentials = Depends(security),
    token: str = Depends(oauth2_scheme),
):
    """
    Verify either API Key (Basic Auth) or JWT token.
    """
    if api_key and (api_key.username and api_key.password):
        # If API key credentials are provided
        return verify_credentials(api_key)
    elif token:
        # If JWT token is provided
        return verify_jwt_token(token)
    else:
        raise HTTPException(
            status_code=401,
            detail="No credentials provided"
        )


@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = validate_user_password(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=401, detail="Incorrect username or password"
        )

    # Create JWT token with user details
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": form_data.username, "role": user},
        expires_delta=access_token_expires,
    )
    return {"access_token": access_token, "token_type": "bearer"}

# Verify a user's password


@app.post('/password/verify')
# , user_properties: dict = Depends(verify_credentials_or_token)):
def verify_password(body: VerifyPasswordInfo):
    """
    Verify a password

    Parameters
    ----------
    username : str
        The user's username
    password : str
        The user's password
    """

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
def load_model(model_name: str, model_flavor: str, model_version_or_alias: str | int, body: LoadRequest, background_tasks: BackgroundTasks, user_properties: dict = Depends(verify_credentials_or_token)):
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

    if user_properties['role'] not in ['admin', 'data_scientist']:
        raise HTTPException(
            403,
            'User does not have permission'
        )

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
def list_models(user_properties: dict = Depends(verify_credentials_or_token)):
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
def unload_model(model_name: str, model_flavor: str, model_version_or_alias: str | int, user_properties: dict = Depends(verify_credentials_or_token)):
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

    if user_properties['role'] not in ['admin', 'data_scientist']:
        raise HTTPException(
            403,
            'User does not have permission'
        )

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
def predict(model_name: str, model_flavor: str, model_version_or_alias: str | int, body: PredictRequest, user_properties: dict = Depends(verify_credentials_or_token)):
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
        if model_flavor not in [TRANSFORMERS_FLAVOR, HUGGINGFACE_FLAVOR] and body.convert_to_numpy:
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
def create_user(user_info: UserInfo, user_properties: dict = Depends(verify_credentials_or_token)):
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
def delete_user(username, user_properties: dict = Depends(verify_credentials_or_token)):
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
    if user_properties['role'] != 'admin' and username != user_properties['username']:
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
def issue_new_password(username, new_password: str = Body(embed=True), user_properties: dict = Depends(verify_credentials_or_token)):
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
def get_user_role(username: str):
    """
    Get a user's role

    Parameters
    ----------
    username : str
        The username of the user
    """
    try:
        return fget_user_role(username)
    except Exception:
        raise HTTPException(500, 'An unknown error occurred')

# Update user role


@app.put('/users/role/{username}')
def update_user_role(username: str, new_role=Body(embed=True), user_properties: dict = Depends(verify_credentials_or_token)):
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
def list_users(user_properties: dict = Depends(verify_credentials_or_token)):
    """
    List all users
    """

    try:
        return flist_users()
    except Exception:
        raise HTTPException(500, 'An unknown error occurred')


@app.get('/reset')
def reset(user_properties: dict = Depends(verify_credentials_or_token)):
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
def get_usage(user_properties: dict = Depends(verify_credentials_or_token)):
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
def upload_file(body: DataUploadRequest, user_properties: dict = Depends(verify_credentials_or_token)):
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

    try:
        filename = upload_data_to_fs(
            body.filename,
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
def download_file(body: DataDownloadRequest, user_properties: dict = Depends(verify_credentials_or_token)):
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
            body.filename
        )
        return content

    except Exception as e:
        raise HTTPException(
            400,
            f'The following error occurred: {str(e)}'
        )


@app.post('/data/list')
def list_files(body: DataListRequest, user_properties: dict = Depends(verify_credentials_or_token)):
    """
    List data files within a directory in the data store

    Parameters
    ----------
    body : DataListRequest
        The information about the directory to list

    Returns
    -------
    files : str
        The files and directories within the directory
    """

    try:
        return list_fs_directory(body.directory)
    except Exception as e:
        raise HTTPException(
            500,
            f'The following error occurred: {str(e)}'
        )


@app.get('/variable-store/get/{variable_name}')
def get_variable(variable_name: str, user_properties: dict = Depends(verify_credentials_or_token)):
    """
    Retrieve a variable from the variable store

    Parameters
    ----------
    variable_name : str
        The name of the variable
    """

    username = user_properties['username']

    try:
        return variable_store[username][variable_name]
    except Exception:
        raise HTTPException(
            404,
            'User does not have a variable with that identifier saved'
        )


@app.get('/variable-store/list')
def list_variables(user_properties: dict = Depends(verify_credentials_or_token)):
    """
    List Variables
    """

    username = user_properties['username']

    # Try to return list of variable names
    try:
        return list(variable_store[username].keys())

    # No variables for user, return empty list
    except Exception:
        return []


@app.post('/variable-store/set')
def set_variable(body: VariableSetRequest, user_properties: dict = Depends(verify_credentials_or_token)):
    """
    Set a variable

    Parameters:
    variable : str
        The variable identifier
    variable_properties : VariableSetRequest
        JSON payload with the value for the variable and whether to overwrite the variable if it is already set
    """

    username = user_properties['username']

    # Check if the variable exists and overwrite is False
    if not body.overwrite:
        try:
            existing_variable = variable_store[user_properties[username]
                                               ][body.variable_name]
        except Exception:
            existing_variable = None

        if existing_variable:
            raise HTTPException(
                400,
                'Variable already exists and overwrite was False'
            )

    # Now, try to write to the variable store, but be careful about edge cases
    if not variable_store.get(username):
        variable_store[username] = {
            body.variable_name: body.value
        }
    else:
        variable_store[user_properties['username']
                       ][body.variable_name] = body.value

    # Write the variable store to disk
    with open(VARIABLE_STORE_FILE, 'w') as f:
        json.dump(variable_store, f)

    return {
        'success': True
    }


@app.delete('/variable-store/delete/{variable_name}')
def delete_variable(variable_name: str, user_properties: dict = Depends(verify_credentials_or_token)):
    """
    Delete a variable

    Parameters
    ----------
    variable_name : str
        The name of the variable
    """

    username = user_properties['username']

    # Try to delete the specified variable for the user and rewrite the variable store
    try:
        del variable_store[username][variable_name]
        with open(VARIABLE_STORE_FILE, 'w') as f:
            json.dump(variable_store, f)
        return {
            'success': True
        }

    # If any error occurs, return HTTPException with 404 code
    except Exception:
        raise HTTPException(
            404,
            'No variable to delete'
        )
