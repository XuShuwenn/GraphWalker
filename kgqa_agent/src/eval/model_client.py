import os
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Unified model configuration for API-based model clients."""
    model: str  # Model name
    base_url: Optional[str] = None  # API base URL (if None, uses default OpenAI API)
    base_url_env: Optional[str] = "OPENAI_BASE_URL"  # Environment variable for base URL
    api_key: Optional[str] = None  # API key (optional)
    api_key_env: str = "OPENAI_API_KEY"  # Environment variable for API key
    temperature: float = 0.6
    max_tokens: int = 4096
    stop: Optional[List[str]] = None
    timeout: float = 60.0  # Request timeout in seconds (default: 60s)


class BaseModelClient:
    """Unified model client that uses OpenAI-compatible API."""
    
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        # Get API key from config or environment
        api_key = cfg.api_key or os.getenv(cfg.api_key_env, "")
        # Get base URL from config or environment
        base_url = cfg.base_url
        if not base_url and cfg.base_url_env:
            base_url = os.getenv(cfg.base_url_env, None)
        
        from openai import OpenAI
        
        # Configure timeout for API requests
        # OpenAI SDK accepts timeout as a float (seconds) or tuple (connect, read)
        client_kwargs = {"timeout": cfg.timeout}
        
        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url, **client_kwargs)
            logger.info(f"Using API server at {base_url} with model {cfg.model} (timeout={cfg.timeout}s)")
        else:
            self.client = OpenAI(api_key=api_key, **client_kwargs)
            logger.info(f"Using default OpenAI API with model {cfg.model} (timeout={cfg.timeout}s)")

    def generate(self, prompt: str, *, system: Optional[str] = None, messages: Optional[List[Dict[str, str]]] = None, **gen_kwargs) -> str:
        """Generate response.
        
        Args:
            prompt: Single prompt string (for backward compatibility)
            system: System message (optional)
            messages: Full conversation history as messages list (preferred for multi-turn)
            **gen_kwargs: Generation parameters
        
        If messages is provided, it takes precedence over prompt/system.
        """
        if messages is not None:
            return self.generate_from_messages(messages, **gen_kwargs)
        # Backward compatibility: build messages from prompt/system
        msgs: List[Dict[str, str]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return self.generate_from_messages(msgs, **gen_kwargs)
    
    def generate_from_messages(self, messages: List[Dict[str, str]], **gen_kwargs) -> str:
        """Generate response from messages list (true multi-turn format).
        
        Args:
            messages: List of message dicts with "role" and "content" keys
            **gen_kwargs: Generation parameters
        """
        if self.client is None:
            return "[API client not available]"
        
        temperature = gen_kwargs.get("temperature", self.cfg.temperature)
        max_tokens = gen_kwargs.get("max_tokens", self.cfg.max_tokens)
        stop = gen_kwargs.get("stop", self.cfg.stop)
        
        resp = self.client.chat.completions.create(
            model=self.cfg.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
        )
        
        if not resp.choices:
            return "[No response choices returned]"
        return resp.choices[0].message.content or ""


def build_model_client(cfg: dict) -> BaseModelClient:
    """Build a model client from configuration.
    
    The API source is controlled by the 'base_url' field in cfg:
    - If 'base_url' is provided: uses that API server (e.g., local vLLM or remote API)
    - If 'base_url' is None and 'base_url_env' is set: reads from environment variable
    - If both are None: uses default OpenAI API
    
    Args:
        cfg: Configuration dictionary with:
            - model: Model name/identifier (required)
            - base_url: API base URL (optional, e.g., "http://localhost:8000/v1" for local vLLM)
            - base_url_env: Environment variable name for base URL (optional, default: "OPENAI_BASE_URL")
            - api_key: API key (optional)
            - api_key_env: Environment variable name for API key (optional, default: "OPENAI_API_KEY")
            - temperature: Temperature for generation (optional, default: 0.6)
            - max_tokens: Max tokens to generate (optional, default: 4096)
            - stop: Stop sequences (optional)
            - timeout: Request timeout in seconds (optional, default: 60.0)
    
    Returns:
        BaseModelClient instance
    """
    cfg = cfg.copy()
    
    # If cfg has 'model_path', use it as 'model' (for backward compatibility)
    if 'model_path' in cfg and 'model' not in cfg:
        cfg['model'] = cfg.pop('model_path')
    
    # Ensure 'model' is present
    if 'model' not in cfg:
        raise ValueError("'model' field is required in model_cfg")
    
    # Set default base_url_env if not specified
    if 'base_url' not in cfg and 'base_url_env' not in cfg:
        cfg['base_url_env'] = "OPENAI_BASE_URL"
    
    return BaseModelClient(ModelConfig(**cfg))
