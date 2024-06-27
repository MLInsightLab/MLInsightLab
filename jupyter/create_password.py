from jupyter_server.auth import passwd
import os

password = passwd(os.environ['JUPYTER_PASSWORD'])
with open(os.environ['PASSWORD_FILE'], 'w') as f:
    f.write(password)
