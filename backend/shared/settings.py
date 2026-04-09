import os
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Backend
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")

    synth_agent_url: str = Field(
        "http://synth-agent:8001", env="SYNTH_AGENT_URL")

    # Hugging Face
    hf_token: str = Field("", env="HF_TOKEN")

    # Cartesia
    cartesia_api_key: str = Field("", env="CARTESIA_API_KEY")
    cartesia_model_id: str = Field("sonic-3", env="CARTESIA_MODEL_ID")

    # Tavily
    tavily_api_key: str = Field("", env="TAVILY_API_KEY")

    # PostgreSQL
    postgres_connection_string: str = Field(
        "postgresql://user:password@localhost:5432/dbname",
        env="POSTGRES_CONNECTION_STRING"
    )
    async_postgres_connection_string: str = Field(
        "",
        env="ASYNC_POSTGRES_CONNECTION_STRING"
    )

    @field_validator('async_postgres_connection_string', mode='before')
    @classmethod
    def ensure_async_protocol(cls, v):
        if v == "":
            v = os.getenv("POSTGRES_CONNECTION_STRING")
        if isinstance(v, str) and v.startswith("postgresql://"):
            return v.replace("postgresql://", "postgresql+asyncpg://")
        return v

    database_pool_size: int = Field(10, env="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(20, env="DATABASE_MAX_OVERFLOW")
    database_echo: bool = Field(False, env="DATABASE_ECHO")

    inference_provider: str = Field("huggingface", env="INFERENCE_PROVIDER")

    secret_key: str = Field("your-super-secret-key", env="SECRET_KEY")

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
    print(f"PostgreSQL connection: {s.postgres_connection_string}")
