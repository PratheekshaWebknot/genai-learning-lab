"""
Entry point for the text generation tool.
Takes user input, calls both models, loops over temperatures, and saves JSON.
"""
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    TEMPERATURES,
    HF_MODEL_NAME,
    API_PROVIDER,
    API_MODEL_NAME,
    ENABLE_FALLBACK,
    FALLBACK_PROVIDER,
    FALLBACK_MODEL_NAME,
    OUTPUT_FILE,
    MAX_NEW_TOKENS,
    TOP_P,
    TOP_K
)
from hf_model import HuggingFaceModel
from api_model import APIModel


def collect_outputs(user_input: str) -> Dict[str, Any]:
    """
    Collect outputs from both models at different temperatures.
    
    Args:
        user_input: The user's text input prompt
        
    Returns:
        Dictionary containing all outputs in the required format
    """
    print("\n" + "="*60)
    print("Text Generation Tool - Day 1 Assignment")
    print("="*60)
    print(f"\nInput prompt: {user_input}\n")
    
    # Initialize models
    print("Initializing models...")
    try:
        hf_model = HuggingFaceModel(model_name=HF_MODEL_NAME)
    except Exception as e:
        print(f"Error initializing HuggingFace model: {e}")
        sys.exit(1)
    
    try:
        api_model = APIModel(
            provider=API_PROVIDER,
            model_name=API_MODEL_NAME,
            enable_fallback=ENABLE_FALLBACK,
            fallback_provider=FALLBACK_PROVIDER if ENABLE_FALLBACK else None,
            fallback_model_name=FALLBACK_MODEL_NAME if ENABLE_FALLBACK else None
        )
    except Exception as e:
        print(f"Error initializing API model: {e}")
        print("\nNote: Make sure you have set your API key in the .env file")
        sys.exit(1)
    
    # Structure to store outputs
    outputs = {
        "huggingface": {},
        "api_model": {}
    }
    
    # Generate with HuggingFace model
    print("\n" + "-"*60)
    print("Generating with HuggingFace model...")
    print("-"*60)
    
    for temp in TEMPERATURES:
        print(f"\nTemperature: {temp}")
        try:
            generated_text = hf_model.generate(
                prompt=user_input,
                temperature=temp,
                max_new_tokens=MAX_NEW_TOKENS,
                top_p=TOP_P,
                top_k=TOP_K
            )
            outputs["huggingface"][f"temp={temp}"] = generated_text
            print(f"✓ Generated {len(generated_text)} characters")
        except Exception as e:
            print(f"✗ Error at temperature {temp}: {e}")
            outputs["huggingface"][f"temp={temp}"] = f"Error: {str(e)}"
    
    # Generate with API model
    print("\n" + "-"*60)
    active_provider = api_model.active_provider or API_PROVIDER
    print(f"Generating with {active_provider.upper()} API model...")
    if ENABLE_FALLBACK and api_model.fallback_provider:
        print(f"(Fallback to {api_model.fallback_provider.upper()} if needed)")
    print("-"*60)
    
    for temp in TEMPERATURES:
        print(f"\nTemperature: {temp}")
        try:
            generated_text = api_model.generate(
                prompt=user_input,
                temperature=temp,
                max_tokens=MAX_NEW_TOKENS,
                top_p=TOP_P
            )
            outputs["api_model"][f"temp={temp}"] = generated_text
            print(f"✓ Generated {len(generated_text)} characters")
        except Exception as e:
            print(f"✗ Error at temperature {temp}: {e}")
            outputs["api_model"][f"temp={temp}"] = f"Error: {str(e)}"
    
    return outputs


def save_outputs(user_input: str, outputs: Dict[str, Any], output_file: Path):
    """
    Save outputs to JSON file.
    
    Args:
        user_input: The original user input
        outputs: Dictionary containing model outputs
        output_file: Path to output JSON file
    """
    result = {
        "input": user_input,
        "timestamp": datetime.now().isoformat(),
        "outputs": outputs
    }
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Outputs saved to: {output_file}")
    except Exception as e:
        print(f"\n✗ Error saving outputs: {e}")


def display_comparison(user_input: str, outputs: Dict[str, Any]):
    """
    Display a comparison of outputs from different models and temperatures.
    
    Args:
        user_input: The original user input
        outputs: Dictionary containing model outputs
    """
    print("\n" + "="*60)
    print("OUTPUT COMPARISON")
    print("="*60)
    print(f"\nInput: {user_input}\n")
    
    for model_name, model_outputs in outputs.items():
        print(f"\n{model_name.upper()} Model:")
        print("-" * 60)
        for temp_key, text in model_outputs.items():
            print(f"\n  {temp_key}:")
            print(f"  {text[:200]}{'...' if len(text) > 200 else ''}")


def main():
    """Main entry point."""
    # Get user input
    if len(sys.argv) > 1:
        # Input provided as command-line argument
        user_input = " ".join(sys.argv[1:])
    else:
        # Interactive input
        print("\nEnter your text prompt (or press Enter to use default):")
        user_input = input("> ").strip()
        
        if not user_input:
            user_input = "The future of artificial intelligence"
            print(f"Using default prompt: {user_input}")
    
    # Collect outputs
    outputs = collect_outputs(user_input)
    
    # Save to JSON
    save_outputs(user_input, outputs, OUTPUT_FILE)
    
    # Display comparison
    display_comparison(user_input, outputs)
    
    print("\n" + "="*60)
    print("Generation complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

