from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello from minimal test!"

if __name__ == '__main__':
    print("Starting minimal test server...")
    app.run(debug=False, port=5050, use_reloader=False)