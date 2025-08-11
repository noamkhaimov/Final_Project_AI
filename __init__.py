from flask import Flask
from dotenv import load_dotenv
import os
from .config import settings
from flask_swagger_ui import get_swaggerui_blueprint
from .api import api_bp, swagger_bp

# Load environment variables from .env file
# load_dotenv()

def create_app() -> Flask:
    app = Flask(__name__)
    
    # Configure Flask settings
    app.config.update(
        ENV=settings.flask_env,
        DEBUG=settings.flask_debug,
        MAX_CONTENT_LENGTH=int(os.getenv('MAX_CONTENT_LENGTH', 16777216))  # Default to 16MB if not set
    )
    
    # Configure other settings
    app.config.from_mapping(settings.dict())

    # Register API blueprint
    app.register_blueprint(api_bp)

    # Register Swagger spec endpoint
    app.register_blueprint(swagger_bp)  # This will expose /spec

    # Register Swagger UI to point to our spec
    SWAGGER_URL = '/swagger'  # URL for exposing Swagger UI
    API_URL = '/spec'  # Our API url (can be a local file)

    # Call factory function to create our blueprint
    swaggerui_blueprint = get_swaggerui_blueprint(
        SWAGGER_URL,  # Swagger UI static files will be mapped to '{SWAGGER_URL}/dist/'
        API_URL,
        config={  # Swagger UI config overrides
            'app_name': "AI Agent",
            'docExpansion': "list"
        }
    )

    # Register Swagger UI blueprint
    app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

    return app
