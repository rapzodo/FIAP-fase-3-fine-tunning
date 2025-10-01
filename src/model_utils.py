import chromadb
import torch
from datasets import Dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)


def load_model_and_tokenizer(model_name, device_map="auto", quantization=True):
    """Load the foundation model and tokenizer"""
    if quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            quantization_config=bnb_config,
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            trust_remote_code=True
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def setup_peft_model(model):
    """Configure the model for Parameter-Efficient Fine-Tuning (PEFT)"""
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]
    )

    if hasattr(model, 'is_quantized') and model.is_quantized:
        model = prepare_model_for_kbit_training(model)

    peft_model = get_peft_model(model, peft_config)
    return peft_model

def prepare_dataset_for_training(data):
    """Prepare dataset for fine-tuning"""
    formatted_data = []

    for item in data:
        prompt = f"Given the product title, provide a detailed description of the product.\n\nProduct: {item['input']}\n\nDescription:"
        response = item['output']
        formatted_text = f"{prompt} {response}"
        formatted_data.append({"text": formatted_text})

    return Dataset.from_list(formatted_data)

def fine_tune_model(model, tokenizer, dataset, output_dir, epochs=3, batch_size=4):
    """Fine-tune the model using the prepared dataset"""
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        optim="adamw_torch",
        learning_rate=2e-4,
        weight_decay=0.01,
        fp16=True,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none"
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

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

def generate_response(model, tokenizer, title, max_new_tokens=256):
    """Generate a product description from a title"""
    formatted_prompt = f"Given the product title, provide a detailed description of the product.\n\nProduct: {title}\n\nDescription:"

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
def setup_groq_client(api_key):
    """Setup Groq client for inference"""
    if not GROQ_AVAILABLE:
        raise ImportError("Groq library not installed. Install with: pip install groq")

    return Groq(api_key=api_key)

def generate_response_groq(client, title, model="llama3-8b-8192", max_tokens=256):
    """Generate a product description using Groq API"""
    prompt = f"Given the product title, provide a detailed description of the product.\n\nProduct: {title}\n\nDescription:"

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        model=model,
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9
    )

    return chat_completion.choices[0].message.content.strip()

def generate_response(model, tokenizer, title, max_new_tokens=256, use_groq=False, groq_client=None, groq_model="llama3-8b-8192"):
    """Generate a product description from a title - supports both local and Groq inference"""
    if use_groq and groq_client:
        return generate_response_groq(groq_client, title, groq_model, max_new_tokens)

    # Local inference
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_response = generated_text[len(formatted_prompt):]

    return generated_response.strip()

def setup_vector_db(products, db_path="./chroma_db"):
    """Setup ChromaDB with product embeddings"""
    client = chromadb.PersistentClient(path=db_path)

    try:
        collection = client.get_collection("amazon_products")
        return client, collection
    except:
        collection = client.create_collection("amazon_products")

        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        documents = []
        metadatas = []
        ids = []

        for i, product in enumerate(products):
            text = f"{product['title']} {product['content']}"
            documents.append(text)
            metadatas.append({
                "title": product['title'],
                "content": product['content']
            })
            ids.append(str(i))

        embeddings = embedding_model.encode(documents).tolist()

        collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

        return client, collection

def find_similar_products_rag(query, products, top_k=3, db_path="./chroma_db"):
    """Find similar products using RAG with ChromaDB and embeddings"""
    try:
        client, collection = setup_vector_db(products, db_path)

        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = embedding_model.encode([query]).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )

        similar_products = []
        for metadata in results['metadatas'][0]:
            similar_products.append({
                'title': metadata['title'],
                'content': metadata['content']
            })

        return similar_products

    except Exception as e:
        print(f"Error with RAG retrieval: {e}")

def find_similar_products(query, products, top_k=3):
    """Find similar products - uses RAG if available, fallback to TF-IDF"""
    return find_similar_products_rag(query, products, top_k)
