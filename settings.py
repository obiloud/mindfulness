import os
from functools import lru_cache

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    # Backend
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")

    # Hugging Face
    hf_token: str = Field(..., env="HF_TOKEN")

    # Cartesia
    cartesia_api_key: str = Field(..., env="CARTESIA_API_KEY")
    cartesia_model_id: str = Field("sonic-3", env="CARTESIA_MODEL_ID")

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()


if __name__ == "__main__":
    # Simple debug print
    s = get_settings()
    print(f"API: http://{s.api_host}:{s.api_port}")
    print(f"Cartesia model: {s.cartesia_model_id}")
