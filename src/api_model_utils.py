from functools import lru_cache

import torch

from model_utils import load_model_and_tokenizer, generate_response

# Global variable for caching the model and tokenizer
_model_cache = {}

@lru_cache(maxsize=1)
def _get_cached_model_and_tokenizer(model_name):
    """Load and cache the model and tokenizer"""
    # For CPU or MPS (Apple Silicon), force CPU to avoid BFloat16 issues
    device_map = "cpu" if torch.backends.mps.is_available() else "auto"

    # Don't use quantization for CPU
    quantization = not (torch.backends.mps.is_available() or not torch.cuda.is_available())

    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        device_map=device_map,
        quantization=quantization
    )

    return model, tokenizer

def test_model_before_finetuning(model_name, query_text):
    """Test a model's response before fine-tuning to establish baseline performance"""
    try:
        model, tokenizer = _get_cached_model_and_tokenizer(model_name)

        response = generate_response(model, tokenizer, query_text)

        return response

    except Exception as e:
        return f"Error generating response with model: {str(e)}"
