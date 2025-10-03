import os

import streamlit as st
import torch

from api_model_utils import test_model_before_finetuning
from model_utils import (
    load_model_and_tokenizer,
    setup_lora_model,
    prepare_dataset,
    fine_tune_model,
    generate_response,
    load_amazon_dataset
)


def check_model_exists(model_dir):
    """Check if model files exist in the directory (not just the directory itself)"""
    # Check for essential files that would indicate a properly saved model
    required_files = ["config.json", "adapter_config.json", "adapter_model.bin"]
    return os.path.isdir(model_dir) and any(
        os.path.exists(os.path.join(model_dir, file)) for file in required_files
    )


def main():
    st.set_page_config(page_title="Fine-Tuning Foundation Models - Tech Challenge", layout="wide")
    st.title("Fine-Tuning Gemma Model for Product Descriptions")
    st.markdown("This application demonstrates fine-tuning of the Gemma-2B model to generate better product descriptions based on product titles.")

    # Sidebar for configuration
    st.sidebar.header("Configuration")

    # Hugging Face authentication
    hf_token = st.sidebar.text_input("Hugging Face Token", type="password", help="Required for Gemma model access. Get your token at https://huggingface.co/settings/tokens")
    if hf_token:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

    # Model selection - using only Gemma model
    model_path = "google/gemma-2b"  # Direct path to the Gemma model

    # Dataset configuration
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "trn.json")
    st.sidebar.subheader("Dataset Configuration")
    max_samples = st.sidebar.slider("Maximum samples for training", 10, 1000, 100)

    # Fine-tuning configuration
    st.sidebar.subheader("Fine-tuning Configuration")
    epochs = st.sidebar.slider("Number of epochs", 1, 5, 1)
    batch_size = st.sidebar.slider("Batch size", 1, 8, 2)
    output_dir = os.path.join(base_dir, "models", "gemma-2b-finetuned")

    # Main content area with tabs
    tabs = st.tabs(["Compare Before/After Fine-tuning", "Dataset Preview", "Fine-tuning Process"])

    # Tab 1: Compare Before/After Fine-tuning
    with tabs[0]:
        st.header("Compare Model Results Before & After Fine-tuning")

        query = st.text_input("Enter a product title:", key="compare_query")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Before Fine-tuning")
            if not hf_token:
                st.warning("⚠️ Hugging Face token required. Please enter your token in the sidebar.")
                button_disabled = True
            else:
                button_disabled = False

            if st.button("Generate with Original Model", disabled=button_disabled):
                with st.spinner("Loading the original Gemma-2B model..."):
                    try:
                        response = test_model_before_finetuning(model_path, query)
                        st.markdown("### Generated Description:")
                        st.write(response)
                    except Exception as e:
                        st.error(f"Error using the original model: {str(e)}")

        with col2:
            st.subheader("After Fine-tuning")

            if check_model_exists(output_dir):
                fine_tuned_available = True
                st.success("Fine-tuned model is available!")
            else:
                fine_tuned_available = False
                st.info("Fine-tuned model is not available yet. Please run the fine-tuning process first.")

            if st.button("Generate with Fine-tuned Model", disabled=not fine_tuned_available):
                with st.spinner("Loading fine-tuned model..."):
                    try:
                        model, tokenizer = load_model_and_tokenizer(output_dir, quantization=False)

                        with st.spinner("Generating response..."):
                            response = generate_response(model, tokenizer, query)
                            st.markdown("### Generated Description:")
                            st.write(response)

                        # Clear GPU memory
                        del model
                        torch.cuda.empty_cache()
                    except Exception as e:
                        st.error(f"Error loading or using the fine-tuned model: {str(e)}")

    # Tab 2: Dataset Preview
    with tabs[1]:
        st.header("Amazon Dataset Preview")

        if os.path.exists(data_path):
            with st.spinner("Loading sample data..."):
                preview_data = load_amazon_dataset(data_path, max_samples=10)

            st.info(f"Showing {len(preview_data)} samples from the dataset")

            # Filter out items with empty content
            valid_samples = [item for item in preview_data if item['output']]
            empty_samples = len(preview_data) - len(valid_samples)

            if empty_samples > 0:
                st.warning(f"{empty_samples} items had empty descriptions and will be skipped during training.")

            for i, item in enumerate(valid_samples):
                with st.expander(f"Sample {i+1}: {item['input']}"):
                    st.markdown("**Title:**")
                    st.write(item['input'])
                    st.markdown("**Description:**")
                    st.write(item['output'])
        else:
            st.error(f"Dataset file not found: {data_path}")

    # Tab 3: Fine-tuning Process
    with tabs[2]:
        st.header("Run Fine-tuning Process")
        st.info("This tab allows you to fine-tune the Gemma-2B model on Amazon product data.")
        st.warning("Fine-tuning may take a long time depending on your hardware. Make sure you have enough GPU memory.")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Fine-tuning Configuration:")
            st.write(f"- Selected model: **Gemma-2B** ({model_path})")
            st.write(f"- Maximum training samples: **{max_samples}**")
            st.write(f"- Number of epochs: **{epochs}**")
            st.write(f"- Batch size: **{batch_size}**")
            st.write(f"- Output directory: **{output_dir}**")

        with col2:
            st.subheader("Execute Fine-tuning")
            if st.button("Start Fine-tuning Process", use_container_width=True, disabled=button_disabled):
                with st.spinner("Loading dataset..."):
                    data = load_amazon_dataset(data_path, max_samples=max_samples)
                    st.info(f"Loaded {len(data)} samples for training.")

                with st.spinner("Loading the Gemma-2B model..."):
                    try:
                        model, tokenizer = load_model_and_tokenizer(model_path, quantization=False)

                        with st.spinner("Setting up LoRA for efficient fine-tuning..."):
                            lora_model = setup_lora_model(model)

                        with st.spinner("Preparing dataset..."):
                            dataset = prepare_dataset(data)

                        with st.spinner(f"Fine-tuning the model ({epochs} epochs)..."):
                            progress_text = st.empty()
                            progress_bar = st.progress(0)

                            for i in range(epochs):
                                progress_text.text(f"Training epoch {i+1}/{epochs}...")
                                progress_bar.progress((i / epochs))
                                fine_tune_model(lora_model, tokenizer, dataset, output_dir, epochs=1, batch_size=batch_size)

                            progress_bar.progress(1.0)

                        st.success(f"Fine-tuning completed! Model saved to {output_dir}")

                        # Clear GPU memory
                        del model
                        del lora_model
                        torch.cuda.empty_cache()
                    except Exception as e:
                        st.error(f"Error during fine-tuning: {str(e)}")


if __name__ == "__main__":
    main()
