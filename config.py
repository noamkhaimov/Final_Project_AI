from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # Flask settings
    flask_env: str = Field("development", env="FLASK_ENV")
    flask_debug: bool = Field(True, env="FLASK_DEBUG")
    
    # Database settings
    mongo_uri: str = Field("mongodb://mongo:27017", env="MONGO_URI")
    mongo_db: str = Field("pdf_upload_db", env="MONGO_DB")
    chroma_host: str = Field("chroma", env="CHROMA_HOST")
    chroma_port: int = Field(8000, env="CHROMA_PORT")
    ollama_model: str = Field("phi3:mini", env="OLLAMA_MODEL")

    class Config:
        case_sensitive = False


settings = Settings()
