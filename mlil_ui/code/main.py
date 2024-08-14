from fastapi.responses import HTMLResponse, RedirectResponse, Response
from starlette.middleware.sessions import SessionMiddleware
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from urllib.parse import urljoin, urlparse
import requests
import os

MLFLOW_TRACKING_URI = os.environ['MLFLOW_TRACKING_URI']
SECRET_KEY = os.environ['SECRET_KEY']
API_URL = os.environ['API_URL']

SYSTEM_USERNAME = os.environ['SYSTEM_USERNAME']
SYSTEM_KEY = os.environ['SYSTEM_KEY']

app = FastAPI(docs_url=None, redoc_url=None)
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

templates = Jinja2Templates(directory="templates")

def authenticate(username: str, password: str):
    with requests.Session() as sess:
        resp = sess.post(
            f'{API_URL}/password/verify',
            json = {
                'username' : username,
                'password' : password
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


@app.get("/login", response_class=HTMLResponse)
async def login_form(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    if authenticate(username, password):
        request.session['user'] = username
        return RedirectResponse(url="/", status_code=302)
    return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid credentials"})


@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    if 'user' not in request.session:
        return RedirectResponse(url="/login")
    return templates.TemplateResponse("home.html", {"request": request})


@app.api_route("/mlflow/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_mlflow(path: str, request: Request):
    if 'user' not in request.session:
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
