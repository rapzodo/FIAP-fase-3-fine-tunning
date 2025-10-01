import json
import os
import pandas as pd
import streamlit as st
from progress_utils import process_json_with_progress, iterate_with_progress
from model_utils import (
    load_model_and_tokenizer,
    setup_peft_model,
    prepare_dataset_for_training,
    fine_tune_model,
    generate_response,
    find_similar_products,
    setup_groq_client,
    GROQ_AVAILABLE
)

def load_json_data(file_path, max_samples=None, min_content_length=10):
    """Load and filter JSON data with progress tracking."""
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            json_data = json.load(f)

            def process_item(item):
                if item.get('content') and len(item['content']) >= min_content_length:
                    return {
                        'title': item['title'],
                        'content': item['content']
                    }
                return None

            data = process_json_with_progress(
                json_data=json_data,
                process_function=process_item,
                max_samples=max_samples,
                update_frequency=100
            )

            return data
        except Exception as e:
            st.error(f"Error processing JSON file: {e}")
            return []

def preprocess_amazon_dataset(input_path, output_path, max_samples=None, min_content_length=10):
    st.write(f"Processing dataset from {input_path}...")

    data = load_json_data(input_path, max_samples, min_content_length)

    df = pd.DataFrame(data)
    st.write(f"Total entries after filtering: {len(df)}")

    st.subheader("Sample entries:")
    st.dataframe(df.head(3))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_json(output_path, orient='records')
    st.success(f"Preprocessed data saved to {output_path}")

    return df

def create_fine_tuning_format(input_path, output_path, model_type="llama"):
    st.write(f"Creating fine-tuning format for {model_type} from {input_path}...")

    with open(input_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    if model_type.lower() in ["llama", "gpt"]:
        def format_item(item, _):
            return {
                "instruction": "Given the product title, provide a detailed description of the product.",
                "input": item["title"],
                "output": item["content"]
            }

        formatted_data = iterate_with_progress(
            items=data,
            process_function=format_item,
            description=f"Formatting data for {model_type}"
        )
    else:
        st.error(f"Unsupported model type: {model_type}")
        return []

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(formatted_data, file, indent=2)

    st.success(f"Fine-tuning data saved to {output_path}")
    st.write(f"Total examples: {len(formatted_data)}")

    st.subheader("Sample formatted data:")
    for i, item in enumerate(formatted_data[:2]):
        with st.expander(f"Example {i+1}"):
            for k, v in item.items():
                if len(str(v)) > 100:
                    st.text(f"{k}: {str(v)[:100]}...")
                else:
                    st.text(f"{k}: {v}")

    return formatted_data

def main():
    st.set_page_config(page_title="Amazon Product Description Generator", layout="wide")

    # Initialize session state variables
    if "fine_tuning_complete" not in st.session_state:
        st.session_state.fine_tuning_complete = False
    if "products_data" not in st.session_state:
        st.session_state.products_data = []
    if "model" not in st.session_state:
        st.session_state.model = None
    if "tokenizer" not in st.session_state:
        st.session_state.tokenizer = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False
    if "before_training_results" not in st.session_state:
        st.session_state.before_training_results = []
    if "after_training_results" not in st.session_state:
        st.session_state.after_training_results = []

    st.title("Fine-Tuning Foundation Models on AmazonTitles Dataset")

    tab_preprocess, tab_model, tab_chat = st.tabs(["1. Preprocess Data", "2. Model", "3. Compare Results"])

    with tab_preprocess:
        st.header("Data Preprocessing")

        col1, col2 = st.columns([1, 1])

        with col1:
            input_path = st.text_input(
                "Input Dataset Path",
                value="data/trn.json"
            )

            output_path = st.text_input(
                "Output Path for Preprocessed Data",
                value="data/preprocessed_amazon.json"
            )

        with col2:
            ft_output = st.text_input(
                "Output Path for Fine-Tuning Data",
                value="data/amazon_ft_data.json"
            )

            model_type = st.selectbox(
                "Model Type for Formatting",
                options=["llama", "gpt"],
                index=0
            )

        col3, col4 = st.columns([1, 1])

        with col3:
            max_samples = st.number_input(
                "Maximum Number of Samples",
                min_value=100,
                max_value=100000,
                value=10000,
                step=100
            )

        with col4:
            min_content_length = st.number_input(
                "Minimum Content Length",
                min_value=1,
                max_value=1000,
                value=20,
                step=1
            )

        if st.button("Start Preprocessing", key="preprocess_button"):
            df = preprocess_amazon_dataset(
                input_path,
                output_path,
                max_samples,
                min_content_length
            )

            create_fine_tuning_format(
                output_path,
                ft_output,
                model_type
            )

            with open(output_path, 'r') as f:
                st.session_state.products_data = json.load(f)

            st.success("✅ Preprocessing complete!")

    with tab_model:
        st.header("Model Loading and Fine-Tuning")

        model_selection = st.radio(
            "Select Foundation Model",
            [
                "TinyLlama (1.1B - Fastest, less accurate)",
                "Llama-2 (7B - Balanced)",
                "Mistral (7B - Best quality)",
                "Custom (specify below)"
            ],
            index=0
        )

        model_mapping = {
            "TinyLlama (1.1B - Fastest, less accurate)": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "Llama-2 (7B - Balanced)": "meta-llama/Llama-2-7b-hf",
            "Mistral (7B - Best quality)": "mistralai/Mistral-7B-v0.1",
            "Custom (specify below)": ""
        }

        col1, col2 = st.columns([1, 1])

        with col1:
            if model_selection == "Custom (specify below)":
                model_name = st.text_input("Foundation Model Path", value="")
            else:
                model_name = st.text_input(
                    "Foundation Model Path",
                    value=model_mapping[model_selection],
                    disabled=True
                )

            data_path = st.text_input(
                "Fine-Tuning Data Path",
                value="data/amazon_ft_data.json"
            )

        with col2:
            output_dir = st.text_input(
                "Output Directory",
                value="models/amazon_ft_model"
            )

            use_quantization = st.checkbox("Use 4-bit Quantization", value=True)

        col3, col4 = st.columns([1, 1])

        with col3:
            epochs = st.number_input("Number of Epochs", min_value=1, max_value=10, value=3, step=1)

        with col4:
            batch_size = st.number_input("Batch Size", min_value=1, max_value=32, value=4, step=1)

        ft_samples = st.number_input(
            "Number of Training Examples (use smaller value for testing)",
            min_value=10,
            max_value=50000,
            value=500,
            step=10
        )

        if not st.session_state.model_loaded:
            if st.button("Load Model", key="load_model_button"):
                final_model_name = model_name if model_name else model_mapping[model_selection]

                with st.spinner(f"Loading model: {final_model_name}"):
                    model, tokenizer = load_model_and_tokenizer(
                        final_model_name,
                        device_map="auto",
                        quantization=use_quantization
                    )

                st.session_state.model = model
                st.session_state.tokenizer = tokenizer
                st.session_state.model_loaded = True

                st.success(f"Model {final_model_name} loaded successfully!")

        if st.session_state.model_loaded and not st.session_state.fine_tuning_complete:
            st.subheader("Test Model Before Fine-Tuning")

            test_query = st.text_input("Enter a product title to test:", key="before_query")

            if st.button("Test Before Fine-Tuning", key="test_before"):
                with st.spinner("Generating response with the model before fine-tuning..."):
                    response = generate_response(
                        st.session_state.model,
                        st.session_state.tokenizer,
                        test_query
                    )

                    st.session_state.before_training_results.append({
                        "query": test_query,
                        "response": response
                    })

                    st.write("**Response before fine-tuning:**")
                    st.write(response)

        if st.session_state.model_loaded and not st.session_state.fine_tuning_complete:
            if st.button("Run Fine-Tuning", key="finetune_button"):
                with st.spinner("Fine-tuning in progress... This may take a while."):
                    with open(data_path, 'r') as f:
                        data = json.load(f)

                    dataset = prepare_dataset_for_training(data)

                    if ft_samples < len(dataset):
                        dataset = dataset.select(range(ft_samples))
                        st.write(f"Using {ft_samples} examples from the dataset")

                    peft_model = setup_peft_model(st.session_state.model)

                    fine_tuned_model, tokenizer = fine_tune_model(
                        peft_model,
                        st.session_state.tokenizer,
                        dataset,
                        output_dir,
                        epochs=epochs,
                        batch_size=batch_size
                    )

                    st.session_state.model = fine_tuned_model
                    st.session_state.tokenizer = tokenizer
                    st.session_state.fine_tuning_complete = True

                    if not st.session_state.products_data:
                        try:
                            with open(output_path, 'r') as f:
                                st.session_state.products_data = json.load(f)
                        except Exception as e:
                            st.warning(f"Could not load product data: {e}")

                    st.success("🎉 Fine-tuning complete!")

        if st.session_state.fine_tuning_complete:
            st.subheader("Test Model After Fine-Tuning")

            test_query = st.text_input("Enter a product title to test:", key="after_query")

            if st.button("Test After Fine-Tuning", key="test_after"):
                with st.spinner("Generating response with the fine-tuned model..."):
                    response = generate_response(
                        st.session_state.model,
                        st.session_state.tokenizer,
                        test_query
                    )

                    st.session_state.after_training_results.append({
                        "query": test_query,
                        "response": response
                    })

                    st.write("**Response after fine-tuning:**")
                    st.write(response)

    with tab_chat:
        st.header("Compare Results")

        if st.session_state.before_training_results or st.session_state.after_training_results:
            st.subheader("Before vs. After Fine-Tuning")

            all_queries = set()
            for result in st.session_state.before_training_results:
                all_queries.add(result["query"])
            for result in st.session_state.after_training_results:
                all_queries.add(result["query"])

            for query in all_queries:
                st.markdown(f"**Query:** {query}")

                col1, col2 = st.columns(2)

                before_response = "Not tested"
                for result in st.session_state.before_training_results:
                    if result["query"] == query:
                        before_response = result["response"]
                        break

                after_response = "Not tested"
                for result in st.session_state.after_training_results:
                    if result["query"] == query:
                        after_response = result["response"]
                        break

                with col1:
                    st.markdown("**Before Fine-Tuning:**")
                    st.write(before_response)

                with col2:
                    st.markdown("**After Fine-Tuning:**")
                    st.write(after_response)

                st.markdown("---")
        else:
            st.info("No results to compare yet. Test the model before and after fine-tuning to see the comparison.")

        if st.session_state.fine_tuning_complete:
            st.subheader("Chat with Fine-Tuned Model")

            chat_container = st.container()
            with chat_container:
                for message in st.session_state.chat_history:
                    if message["role"] == "user":
                        st.markdown(f"**You:** {message['content']}")
                    else:
                        st.markdown(f"**AI:** {message['content']}")

                        if "references" in message:
                            st.markdown("**References:**")
                            for i, ref in enumerate(message["references"]):
                                with st.expander(f"Reference {i+1}: {ref['title']}"):
                                    st.write(ref["content"])

            product_query = st.text_input("Enter a product title to generate a description:", key="chat_query")

            if st.button("Generate Description", key="chat_generate"):
                if product_query:
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": product_query
                    })

                    with st.spinner("Generating product description..."):
                        response = generate_response(
                            st.session_state.model,
                            st.session_state.tokenizer,
                            product_query
                        )

                    references = []
                    if st.session_state.products_data:
                        similar_products = find_similar_products(
                            product_query,
                            st.session_state.products_data,
                            top_k=3
                        )

                        references = [
                            {
                                "title": product["title"],
                                "content": product["content"]
                            }
                            for product in similar_products
                        ]

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response,
                        "references": references
                    })

                    st.experimental_rerun()

if __name__ == "__main__":
    main()
