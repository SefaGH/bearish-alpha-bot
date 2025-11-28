import torch
import torch.quantization
import os
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml.neural_networks import LSTMRegimePredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quantize_model(model_path, output_path, input_size=82):
    """
    Quantize a PyTorch LSTM model to INT8.
    """
    logger.info(f"Attempting to quantize model at {model_path}")
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return False

    try:
        # Load the model
        # Try loading as TorchScript first
        try:
            model = torch.jit.load(model_path)
            logger.info("Loaded model as TorchScript")
        except Exception as e:
            # Fallback to state dict
            logger.info(f"Could not load as TorchScript ({e}), falling back to state dict")
            model = LSTMRegimePredictor(input_size=input_size)
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict)
            model.eval()

        # Apply dynamic quantization
        logger.info("Applying dynamic quantization (INT8)...")
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear, torch.nn.LSTM},  # Layers to quantize
            dtype=torch.qint8
        )

        # Save the quantized model
        logger.info(f"Saving quantized model to {output_path}")
        
        # For TorchScript models, we should try to save as TorchScript if possible, 
        # but quantize_dynamic returns a standard nn.Module. 
        # If the original was TorchScript, we might want to script/trace it again, 
        # but that requires example inputs. 
        # For now, we save as a standard PyTorch model (pickle) which can be loaded with torch.load
        # or we can try to script it if it's simple.
        
        try:
            # Attempt to script it back to TorchScript
            scripted_quantized = torch.jit.script(quantized_model)
            torch.jit.save(scripted_quantized, output_path)
            logger.info("Saved as TorchScript")
        except Exception as e:
            logger.warning(f"Could not save as TorchScript ({e}), saving as standard PyTorch model")
            torch.save(quantized_model, output_path)
        
        # Compare sizes
        orig_size = os.path.getsize(model_path) / (1024 * 1024)
        quant_size = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(f"Original size: {orig_size:.2f} MB")
        logger.info(f"Quantized size: {quant_size:.2f} MB")
        logger.info(f"Reduction: {(1 - quant_size/orig_size)*100:.1f}%")
        
        return True

    except Exception as e:
        logger.error(f"Quantization failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    # GEMMA Configuration
    BUNDLE_DIR = Path("artifacts/gemma/final")
    
    # Models to quantize
    models_to_process = [
        ("gemma_regime.pt", "gemma_regime_int8.pt"),
        ("gemma_price.pt", "gemma_price_int8.pt")
    ]
    
    # Ensure directory exists
    if not BUNDLE_DIR.exists():
        logger.error(f"Bundle directory not found: {BUNDLE_DIR}")
        sys.exit(1)
        
    for input_name, output_name in models_to_process:
        model_path = BUNDLE_DIR / input_name
        output_path = BUNDLE_DIR / output_name
        
        if model_path.exists():
            logger.info(f"Processing {input_name}...")
            quantize_model(model_path, output_path, input_size=82)
        else:
            logger.warning(f"Skipping {input_name}: File not found")
