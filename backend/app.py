from flask import Flask
from flask_cors import CORS
from routes.api import api_bp

app = Flask(__name__)
CORS(app)

# Register blueprints
app.register_blueprint(api_bp, url_prefix="/api")


@app.route("/")
def home():
    return {"message": "Flask backend is running!"}


if __name__ == "__main__":
    app.run(debug=True, port=5000)
