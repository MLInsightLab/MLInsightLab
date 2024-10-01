from fastapi.responses import HTMLResponse, RedirectResponse, Response
from starlette.middleware.sessions import SessionMiddleware
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from urllib.parse import urljoin, urlparse
import requests
import os
import time

MLFLOW_TRACKING_URI = os.environ['MLFLOW_TRACKING_URI']
SECRET_KEY = os.environ['SECRET_KEY']
API_URL = os.environ['API_URL']

SYSTEM_USERNAME = os.environ['SYSTEM_USERNAME']
SYSTEM_KEY = os.environ['SYSTEM_KEY']

# Timeout in seconds (5 minutes)
INACTIVITY_TIMEOUT = 5 * 60

app = FastAPI(docs_url=None, redoc_url=None)
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

templates = Jinja2Templates(directory="templates")


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
        try:
            role = resp.json()
            if role in ['admin', 'data_scientist']:
                return True
        except Exception:
            pass


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
        allow_redirects=False
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

    response = requests.get(f'{API_URL}/models/list',
                            auth=(SYSTEM_USERNAME, SYSTEM_KEY))
    models = response.json() if response.ok else []

    return templates.TemplateResponse("list_models.html", {"request": request, "models": models})


@app.get("/variables", response_class=HTMLResponse)
async def manage_variables(request: Request):
    if 'user' not in request.session or not check_inactivity(request):
        return RedirectResponse(url="/login")

    username = request.session['user']
    response = requests.post(f'{API_URL}/variable-store/list',
                             json={'username': username}, auth=(SYSTEM_USERNAME, SYSTEM_KEY))
    variables = response.json() if response.ok else []

    variables_and_values = []
    if len(variables) > 0:
        for variable in variables:
            variable_value = requests.post(f'{API_URL}/variable-store/get', json={
                                           'username': username, 'variable_name': variable}, auth=(SYSTEM_USERNAME, SYSTEM_KEY)).json()
            variables_and_values.append(
                {
                    'variable': variable,
                    'variable_value': variable_value
                }
            )

    return templates.TemplateResponse("manage_variables.html", {"request": request, "variables": variables_and_values})
