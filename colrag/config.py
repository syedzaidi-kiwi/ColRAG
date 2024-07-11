import os
from typing import Dict, Any
from pydantic import BaseSettings, Field
from dotenv import load_dotenv

load_dotenv()

class ColRAGConfig(BaseSettings):
    MODEL_NAME: str = Field(default="colbert-ir/colbertv2.0", env="COLRAG_MODEL_NAME")
    INDEX_DIRECTORY: str = Field(default="./indices", env="COLRAG_INDEX_DIRECTORY")
    BATCH_SIZE: int = Field(default=1000, env="COLRAG_BATCH_SIZE")
    MAX_WORKERS: int = Field(default=os.cpu_count() or 1, env="COLRAG_MAX_WORKERS")
    CACHE_DIR: str = Field(default="./cache", env="COLRAG_CACHE_DIR")
    LOG_LEVEL: str = Field(default="INFO", env="COLRAG_LOG_LEVEL")

    class Config:
        env_prefix = "COLRAG_"

config = ColRAGConfig()

def get_config() -> Dict[str, Any]:
    return config.dict()