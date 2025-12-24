"""
API-based model logic for text generation.
Supports OpenAI and Groq APIs with automatic fallback.
"""
import os
from typing import Optional, Tuple
from openai import OpenAI
import requests


class APIModel:
    """
    Wrapper for API-based LLM models (OpenAI, Groq, etc.) with fallback support.
    
    Automatically falls back to secondary provider if primary fails.
    """
    
    def __init__(
        self,
        provider: str = "openai",
        model_name: str = "gpt-3.5-turbo",
        enable_fallback: bool = True,
        fallback_provider: Optional[str] = "groq",
        fallback_model_name: Optional[str] = "llama2-70b-4096"
    ):
        """
        Initialize the API model with fallback support.
        
        Args:
            provider: Primary API provider ("openai" or "groq")
            model_name: Name of the primary model to use
            enable_fallback: Whether to enable fallback to secondary provider
            fallback_provider: Fallback provider if primary fails
            fallback_model_name: Model name for fallback provider
        """
        self.primary_provider = provider.lower()
        self.primary_model_name = model_name
        self.enable_fallback = enable_fallback
        self.fallback_provider = fallback_provider.lower() if fallback_provider else None
        self.fallback_model_name = fallback_model_name
        
        # Track which provider/model is currently active
        self.active_provider = None
        self.active_model_name = None
        self.client = None
        
        # Initialize primary client
        self._initialize_client(self.primary_provider, self.primary_model_name, is_primary=True)
    
    def _initialize_client(self, provider: str, model_name: str, is_primary: bool = True):
        """
        Initialize the API client based on provider.
        
        Args:
            provider: API provider ("openai" or "groq")
            model_name: Name of the model to use
            is_primary: Whether this is the primary provider
        """
        provider = provider.lower()
        
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY", "")
            if not api_key:
                if is_primary:
                    raise ValueError(
                        "OPENAI_API_KEY not found in environment variables. "
                        "Please set it in your .env file."
                    )
                return False  # Fallback failed, but don't raise
            
            self.client = OpenAI(api_key=api_key)
            self.active_provider = "openai"
            self.active_model_name = model_name
            status = "primary" if is_primary else "fallback"
            print(f"✓ OpenAI client initialized ({status}) with model: {model_name}")
            return True
            
        elif provider == "groq":
            api_key = os.getenv("GROQ_API_KEY", "")
            if not api_key:
                if is_primary:
                    raise ValueError(
                        "GROQ_API_KEY not found in environment variables. "
                        "Please set it in your .env file."
                    )
                return False  # Fallback failed, but don't raise
            
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1"
            )
            self.active_provider = "groq"
            self.active_model_name = model_name
            status = "primary" if is_primary else "fallback"
            print(f"✓ Groq client initialized ({status}) with model: {model_name}")
            return True
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _try_fallback(self):
        """
        Attempt to initialize fallback provider.
        
        Returns:
            True if fallback succeeded, False otherwise
        """
        if not self.enable_fallback or not self.fallback_provider:
            return False
        
        print(f"\n⚠ Primary provider ({self.primary_provider}) failed. Attempting fallback to {self.fallback_provider}...")
        
        try:
            success = self._initialize_client(
                self.fallback_provider,
                self.fallback_model_name,
                is_primary=False
            )
            if not success:
                # Check if API key is missing
                fallback_key = "GROQ_API_KEY" if self.fallback_provider == "groq" else "OPENAI_API_KEY"
                api_key = os.getenv(fallback_key, "")
                if not api_key:
                    print(f"✗ Fallback failed: {fallback_key} not found in environment variables.")
                    print(f"  Please add {fallback_key} to your .env file to enable fallback.")
                else:
                    print(f"✗ Fallback initialization failed (unknown reason)")
            return success
        except Exception as e:
            print(f"✗ Fallback initialization failed: {e}")
            return False
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 150,
        top_p: float = 0.9
    ) -> str:
        """
        Generate text from a prompt using API with automatic fallback.
        
        Tries primary provider first, then falls back to secondary if enabled.
        
        Args:
            prompt: Input text prompt
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum number of tokens to generate
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated text string
            
        Raises:
            RuntimeError: If both primary and fallback providers fail
        """
        if self.client is None:
            raise RuntimeError("API client not initialized.")
        
        # Try primary provider first
        try:
            response = self.client.chat.completions.create(
                model=self.active_model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p
            )
            
            generated_text = response.choices[0].message.content.strip()
            return generated_text
            
        except Exception as primary_error:
            # Primary provider failed - try fallback if enabled
            error_msg = str(primary_error)
            print(f"\n✗ Primary provider ({self.active_provider}) error: {error_msg}")
            
            # Check if this is a retryable error (quota, timeout, rate limit, etc.)
            retryable_errors = [
                "quota", "rate limit", "timeout", "429", "503", "500",
                "insufficient_quota", "billing", "payment"
            ]
            
            is_retryable = any(err.lower() in error_msg.lower() for err in retryable_errors)
            
            if is_retryable and self._try_fallback():
                # Retry with fallback provider
                try:
                    print(f"Retrying with {self.active_provider} ({self.active_model_name})...")
                    response = self.client.chat.completions.create(
                        model=self.active_model_name,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p
                    )
                    
                    generated_text = response.choices[0].message.content.strip()
                    print(f"✓ Fallback succeeded!")
                    return generated_text
                    
                except Exception as fallback_error:
                    raise RuntimeError(
                        f"Both primary ({self.primary_provider}) and fallback "
                        f"({self.fallback_provider}) providers failed.\n"
                        f"Primary error: {error_msg}\n"
                        f"Fallback error: {str(fallback_error)}"
                    )
            else:
                # Not retryable or fallback not available/enabled
                if not is_retryable:
                    raise RuntimeError(
                        f"Non-retryable error from {self.active_provider}: {error_msg}"
                    )
                else:
                    # Fallback was attempted but failed
                    fallback_key = "GROQ_API_KEY" if self.fallback_provider == "groq" else "OPENAI_API_KEY"
                    api_key = os.getenv(fallback_key, "")
                    if not api_key:
                        raise RuntimeError(
                            f"Primary provider ({self.primary_provider}) failed and fallback unavailable.\n"
                            f"Reason: {fallback_key} not found in environment variables.\n"
                            f"Please add {fallback_key} to your .env file to enable fallback.\n"
                            f"Primary error: {error_msg}"
                        )
                    else:
                        raise RuntimeError(
                            f"Primary provider ({self.primary_provider}) failed and fallback unavailable.\n"
                            f"Fallback provider ({self.fallback_provider}) initialization failed.\n"
                            f"Primary error: {error_msg}"
                        )

