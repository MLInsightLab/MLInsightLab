from flask import Flask, render_template
import waitress

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    waitress.serve(app, host='0.0.0.0', port=80)
