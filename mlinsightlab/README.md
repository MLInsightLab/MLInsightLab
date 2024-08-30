# MLIL Python Client
This package allows you to more easily interact with the MLIL platform from a Python interface.

Please note that this is a supplement to the JupyterLab instance in the platform and, as such, 
is not intended to replicate the end-to-end data science workflow that MLIL enables. Rather, this 
client is designed to make it easier to make requests to MLIL's model invocation, model management, 
and administrative API endpoints.

## Basic usage

### Installation
```bash
pip install mlinsightlab
```
### Getting started
When first creating a MLILClient object, you'll be prompted to input your MLIL platform credentials, 
including the API key that you have been issued to interact with the platform. This information will 
be stored in a configuration file located at ```HOME/.mlil/config.json```.

```python
    >>> from mlinsightlab import MLILClient
    # creates a new client object
    >>> client = mlil.MLILClient() 
```
Now that you've logged in once, you'll be able to use these saved credentials to create MLILClient 
objects more easily in the future.

### Basic usage
Now that you're authenticated, you can more easily interact with your deployment of MLIL!
```python
# list all users on the platform
client.list_users()

# create a new user in the platform
client.create_user(role = 'user', api_key='mmm',username='Homer.Simpson', password='Doughnuts!')

# double-check a user's role
client.get_user_role(username='Homer.Simpson')

# verify a user's password

# issue a user a new password
client.issue_new_password(new_password='new_password') # by default updates the config.json file

# issue a user a new API key
client.issue_api_key(username='Homer.Simpson',password='new_password') # by default updates the config.json file

# delete a user
client.delete_user(username='Homer.Simpson', verbose=True
```

Now that your platform users are all set, it's time to manage and use your models.

```python
# list models
client.list_models()

# predict - in this case model flavor is transformers, but you can use e.g. pyfunc, etc.
client.predict(model_name='GPT-AGI', model_flavor='transformers', model_version_or_alias='1',data='Hello AI overlord!'
```
If you've been working on the public library's computers, or just want to erase the ```config.json``` 
file containing your credentials, you can also do so via the Python client.

```python
client.purge_credentials()
'''