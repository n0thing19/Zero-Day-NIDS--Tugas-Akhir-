"""Entry point Server Backend Flask untuk NIDS Studio."""
from flask import Flask
from flask_cors import CORS

from config import HOST, PORT
from routes import api_bp


def create_app() -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")
    CORS(app)
    app.register_blueprint(api_bp)
    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=False, host=HOST, port=PORT)