"""
HuggingFace model logic for text generation.
Handles loading and running open-source models.
"""
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Optional
import warnings

warnings.filterwarnings("ignore")


class HuggingFaceModel:
    """Wrapper for HuggingFace text generation models."""
    
    def __init__(self, model_name: str = "gpt2"):
        """
        Initialize the HuggingFace model.
        
        Args:
            model_name: Name of the HuggingFace model to use
        """
        self.model_name = model_name
        self.generator = None
        self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer."""
        try:
            print(f"Loading HuggingFace model: {self.model_name}...")
            # Use pipeline for easier text generation
            self.generator = pipeline(
                "text-generation",
                model=self.model_name,
                tokenizer=self.model_name,
                device=-1 if torch.cuda.is_available() else 0,  # Use GPU if available
                model_kwargs={"torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32}
            )
            print(f"✓ Model {self.model_name} loaded successfully!")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            print("Falling back to CPU-only mode...")
            try:
                self.generator = pipeline(
                    "text-generation",
                    model=self.model_name,
                    tokenizer=self.model_name,
                    device=0  # CPU
                )
            except Exception as e2:
                raise RuntimeError(f"Failed to load model: {e2}")
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_new_tokens: int = 150,
        top_p: float = 0.9,
        top_k: int = 50
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text prompt
            temperature: Sampling temperature (0.0 to 1.0+)
            max_new_tokens: Maximum number of tokens to generate
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            
        Returns:
            Generated text string
        """
        if self.generator is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        try:
            # Generate text with specified parameters
            results = self.generator(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.generator.tokenizer.eos_token_id
            )
            
            # Extract generated text
            generated_text = results[0]["generated_text"]
            
            # Remove the original prompt from the output
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            raise RuntimeError(f"Error during text generation: {e}")

