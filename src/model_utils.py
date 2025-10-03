import json
import os

import torch
from datasets import Dataset
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)


def can_use_quantization():
    """Check if accelerate library is available and if CUDA is available for quantization"""
    try:
        import accelerate
        has_accelerate = True
    except ImportError:
        has_accelerate = False

    # Check if all requirements are met for quantization
    return has_accelerate and torch.cuda.is_available()

def load_model_and_tokenizer(model_name, device_map="cpu", quantization=True):
    """Load the foundation model and tokenizer"""
    # Get token from environment variable (set by the Streamlit app)
    hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN", None)

    if quantization and can_use_quantization:
        try:
            model = quantized_model_load(device_map, hf_token, model_name)
        except Exception as e:
            print(f"Quantization failed: {str(e)}. Falling back to standard loading.")
            model = standard_model_load(device_map, hf_token, model_name)
    else:
        model = standard_model_load(device_map, hf_token, model_name)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=hf_token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def quantized_model_load(device_map, hf_token, model_name):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        token=hf_token
    )
    model.is_quantized = True
    return model


def standard_model_load(device_map, hf_token, model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        trust_remote_code=True,
        token=hf_token
    )
    model.is_quantized = False
    return model


def prepare_model_for_training(model):
    """Prepare model for training by moving parameters from meta device to actual device"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Unwrap DDP model if needed
    if hasattr(model, "module"):
        model = model.module

    # Move model parameters from meta device if needed
    for param in model.parameters():
        if hasattr(param, "device") and param.device.type == 'meta':
            param.data = param.data.to(device)

    return model


# Setup LoRA for efficient fine-tuning
def setup_lora_model(model):
    """Configure the model for Parameter-Efficient Fine-Tuning with LoRA"""
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj",
                        "gate_proj", "up_proj", "down_proj"]
    )

    if hasattr(model, 'is_quantized') and model.is_quantized:
        model = prepare_model_for_kbit_training(model)

    lora_model = get_peft_model(model, peft_config)
    return lora_model

# Prepare dataset for training
def prepare_dataset(data):
    """Prepare dataset for fine-tuning"""
    formatted_data = []

    for item in data:
        prompt = f"Given the product title, provide a detailed description of the product.\n\nProduct: {item['input']}\n\nDescription:"
        response = item['output']
        formatted_text = f"{prompt} {response}"
        formatted_data.append({"text": formatted_text})

    return Dataset.from_list(formatted_data)

# Fine-tune model
def fine_tune_model(model, tokenizer, dataset, output_dir, epochs=1, batch_size=2):
    """Fine-tune the model using the prepared dataset"""
    # Check if MPS (Metal Performance Shaders) is available for Apple Silicon
    use_mps = hasattr(torch, 'mps') and torch.backends.mps.is_available()
    device = "mps" if use_mps else "cpu"
    print(f"Training on device: {device}")

    # Prepare model by moving parameters from meta device if needed
    model = prepare_model_for_training(model)

    # Move model to the appropriate device
    model.to(device)

    # Configure training arguments optimized for Apple M4 Pro
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=8,  # Increased for memory efficiency
        optim="adamw_torch",
        learning_rate=2e-4,
        weight_decay=0.01,
        fp16=False,  # Disable fp16 for Apple Silicon
        logging_steps=20,
        save_strategy="epoch",
        evaluation_strategy="no",  # Disable evaluation to speed up training
        save_total_limit=1,
        load_best_model_at_end=False,
        report_to=None
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)  # Reduced length for memory efficiency

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return model, tokenizer

# Generate response using the model
def generate_response(model, tokenizer, query):
    """Generate a product description from a title"""
    prompt = f"Given the user question, provide a detailed description of the product.\n\nQuery: {query}\n\nDescription:"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=20)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.replace(prompt,"")

# Load Amazon dataset
def load_amazon_dataset(file_path, max_samples=None):
    """Load data from trn.json file which is in JSON Lines format"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                try:
                    item = json.loads(line.strip())
                    if 'title' in item and 'content' in item:
                        data.append({
                            "input": item['title'],
                            "output": item['content']
                        })
                except json.JSONDecodeError:
                    continue  # Skip lines that can't be parsed

            # Apply max_samples limit if specified
            if max_samples and len(data) >= max_samples:
                break

    return data

# Save preprocessed data
def save_preprocessed_data(data, output_path):
    """Save preprocessed data to a JSON file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
