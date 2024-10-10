from fastapi.responses import HTMLResponse, RedirectResponse, Response
from starlette.middleware.sessions import SessionMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, Request, Form
from urllib.parse import urljoin
import requests
import secrets
import string
import time
import os

MLFLOW_TRACKING_URI = os.environ['MLFLOW_TRACKING_URI']
API_URL = os.environ['API_URL']

SYSTEM_USERNAME = os.environ['SYSTEM_USERNAME']
SYSTEM_KEY = os.environ['SYSTEM_KEY']

SECRET_KEY = ''.join([secrets.choice(string.ascii_letters) for _ in range(32)])

# Timeout in seconds (5 minutes)
INACTIVITY_TIMEOUT = 5 * 60

app = FastAPI(docs_url=None, redoc_url=None)
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

templates = Jinja2Templates(directory="templates")

app.mount('/static', StaticFiles(directory = 'static'), name = 'static')

def authenticate(username: str, password: str):
    with requests.Session() as sess:
        resp = sess.post(
            f'{API_URL}/password/verify',
            json={
                'username': username,
                'password': password
            },
            auth=(SYSTEM_USERNAME, SYSTEM_KEY)
        )
    if resp.ok:
        return True


def check_inactivity(request: Request):
    last_active = request.session.get('last_active', None)
    if last_active:
        if time.time() - last_active > INACTIVITY_TIMEOUT:
            request.session.clear()
            return False
    request.session['last_active'] = time.time()
    return True


@app.get("/login", response_class=HTMLResponse)
async def login_form(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    if authenticate(username, password):
        request.session['user'] = username
        request.session['last_active'] = time.time()
        return RedirectResponse(url="/", status_code=302)
    return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid credentials"})


@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    if 'user' not in request.session or not check_inactivity(request):
        return RedirectResponse(url="/login")
    return templates.TemplateResponse("home.html", {"request": request})


@app.api_route("/mlflow/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_mlflow(path: str, request: Request):
    if 'user' not in request.session or not check_inactivity(request):
        return RedirectResponse(url="/login")

    mlflow_url = urljoin(MLFLOW_TRACKING_URI, path)
    query_string = request.url.query

    response = requests.request(
        method=request.method,
        url=mlflow_url,
        headers={key: value for (key, value)
                 in request.headers.items() if key != 'Host'},
        params=query_string,
        data=await request.body(),
        cookies=request.cookies,
        allow_redirects=False,
        auth=(SYSTEM_USERNAME, SYSTEM_KEY)
    )

    client_response = Response(
        content=response.content,
        status_code=response.status_code,
        headers={key: value for key, value in response.headers.items()
                 if key.lower() != 'transfer-encoding'}
    )

    return client_response


@app.get("/user/settings", response_class=HTMLResponse)
async def user_settings(request: Request):
    if 'user' not in request.session or not check_inactivity(request):
        return RedirectResponse(url="/login")
    return templates.TemplateResponse("user_settings.html", {"request": request})


@app.get("/models", response_class=HTMLResponse)
async def list_models(request: Request):
    if 'user' not in request.session or not check_inactivity(request):
        return RedirectResponse(url="/login")

    return templates.TemplateResponse("list_models.html", {"request": request})


@app.get("/variables", response_class=HTMLResponse)
async def manage_variables(request: Request):
    if 'user' not in request.session or not check_inactivity(request):
        return RedirectResponse(url="/login")

    return templates.TemplateResponse("manage_variables.html", {"request": request})
