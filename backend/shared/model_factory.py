# backend/agent_a_chat/factory.py

from typing import Dict, Any
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_ollama import ChatOllama
from shared.settings import get_settings
import os

os.environ["OLLAMA_HOST"] = "http://host.docker.internal:11434"


class MindfulnessChatModel:
    """
    Base factory class for creating chat models with configurable parameters.
    Supports switching between Hugging Face and Ollama models.
    """

    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize the chat model factory with configuration.

        Args:
            model_config: Dictionary containing model configuration
            including 'type' (huggingface or ollama), 'model_name', 
            'max_new_tokens', 'temperature', and 'api_token'
        """
        self.model_config = model_config
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate the model configuration."""
        required_fields = ['type', 'model_name',
                           'max_new_tokens', 'temperature']
        for field in required_fields:
            if field not in self.model_config:
                raise ValueError(f"Missing required field: {field}")

        if self.model_config['type'] not in ['huggingface', 'ollama']:
            raise ValueError("Model type must be 'huggingface' or 'ollama'")

    def get_model(self) -> Any:
        """
        Create and return the appropriate chat model based on configuration.

        Returns:
            ChatHuggingFace or ChatOllama instance
        """
        model_type = self.model_config['type']
        model_name = self.model_config['model_name']
        max_new_tokens = self.model_config['max_new_tokens']
        temperature = self.model_config['temperature']

        if model_type == 'huggingface':
            # Use Hugging Face endpoint with token from settings
            hf_token = get_settings().hf_token
            repo_id = model_name

            # Create Hugging Face endpoint
            endpoint = HuggingFaceEndpoint(
                repo_id=repo_id,
                task="text-generation",
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                huggingfacehub_api_token=hf_token,
            )

            return ChatHuggingFace(llm=endpoint)

        elif model_type == 'ollama':
            # Use Ollama model
            return ChatOllama(
                model=model_name,
                temperature=temperature
            )

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

# Factory functions for specific use cases


def create_fast_hf_llm() -> MindfulnessChatModel:
    """Create a fast LLM for Agent A (lightweight, fast responses)."""
    return MindfulnessChatModel({
        'type': 'huggingface',
        'model_name': 'meta-llama/Meta-Llama-3-70B-Instruct',
        'max_new_tokens': 512,
        'temperature': 0.7
    })


def create_heavy_hf_llm() -> MindfulnessChatModel:
    """Create a heavy LLM for Agent B (deep reflection, analysis)."""
    return MindfulnessChatModel({
        'type': 'huggingface',
        'model_name': 'meta-llama/Meta-Llama-3-70B-Instruct',
        'max_new_tokens': 2048,
        'temperature': 0.4
    })


def create_fast_ollama_llm() -> MindfulnessChatModel:
    """Create an Ollama LLM for use with Qwen3-long model."""
    return MindfulnessChatModel({
        'type': 'ollama',
        'model_name': 'qwen3-long:latest',
        'max_new_tokens': 512,
        'temperature': 0.5
    })


def create_heavy_ollama_llm() -> MindfulnessChatModel:
    """Create an Ollama LLM for use with Qwen3-long model."""
    return MindfulnessChatModel({
        'type': 'ollama',
        'model_name': 'qwen3-long:latest',
        'max_new_tokens': 2048,
        'temperature': 0.3
    })

# Convenience functions to get models directly


def get_fast_hf_llm() -> ChatHuggingFace:
    """Get the fast LLM for Agent A."""
    model = create_fast_hf_llm()
    return model.get_model()


def get_heavy_hf_llm() -> ChatHuggingFace:
    """Get the heavy LLM for Agent B."""
    model = create_heavy_hf_llm()
    return model.get_model()


def get_fast_ollama_llm() -> ChatOllama:
    """Get the Ollama LLM for Qwen3-long."""
    model = create_fast_ollama_llm()
    return model.get_model()


def get_heavy_ollama_llm() -> ChatOllama:
    """Get the Ollama LLM for Qwen3-long."""
    model = create_heavy_ollama_llm()
    return model.get_model()
