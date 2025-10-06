# Fine-Tuning Gemma-2B for Amazon Product Descriptions

## 📋 Project Overview

This project demonstrates fine-tuning the Gemma-2B foundation model using LoRA (Low-Rank Adaptation) on Amazon product data. The model learns to generate product descriptions based on product titles, using the AmazonTitles-1.3MM dataset.

**Tech Challenge - Fase 3 - FIAP**

---

## 🎯 Purpose

1. **Fine-tune a foundation model** (Gemma-2B) on domain-specific data (Amazon products)
2. **Use Parameter-Efficient Fine-Tuning (PEFT)** with LoRA to train on consumer hardware
3. **Compare model performance** before and after fine-tuning
4. **Demonstrate retrieval of training sources** using RAG (Retrieval-Augmented Generation)

---

## 🏗️ Architecture

### Components:
- **Base Model**: Google Gemma-2B (2 billion parameters)
- **Fine-Tuning Method**: LoRA (Low-Rank Adaptation)
- **Dataset**: AmazonTitles-1.3MM (131,262 products with titles and descriptions)
- **Vector Database**: ChromaDB for semantic search
- **Embeddings**: SentenceTransformer (all-MiniLM-L6-v2)
- **UI**: Streamlit web application

### Training Flow:
```
Amazon Dataset → LoRA Fine-Tuning → Fine-tuned Model
                       ↓
                 ChromaDB Indexing (for references)
```

### Inference Flow:
```
User Query → Fine-tuned Model → Generated Description
                ↓
         ChromaDB Semantic Search → Training References (for transparency)
```

---

## 🚀 Quick Start

### Prerequisites

1. **Python 3.10+**
2. **24GB+ RAM** (for model loading)
3. **Apple Silicon (M1/M2/M3/M4)** or **CUDA GPU** (optional, for faster training)
4. **Hugging Face Token** (required for Gemma model access)
   - Get token at: https://huggingface.co/settings/tokens
   - Accept Gemma license at: https://huggingface.co/google/gemma-2b

### Installation

```bash
# Clone the repository
cd FIAP-fase-3-fine-tunning

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
# Start the Streamlit app
streamlit run src/streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📊 Training Configuration

### Default (Overnight Training - Recommended for Demo)

**Optimized for 8-10 hour training on Apple M4 Pro (24GB RAM)**

```python
max_samples = 3000      # 3,000 Amazon products
epochs = 3              # Train for 3 epochs
batch_size = 2          # Batch size of 2
```

**Expected Results:**
- ✅ Training Time: ~8-10 hours
- ✅ Good coverage of product categories
- ✅ Suitable for demonstration purposes
- ✅ Model learns Amazon product Q&A patterns

### Training Time Estimates

| Samples | Epochs | Batch Size | Estimated Time | Use Case |
|---------|--------|------------|----------------|----------|
| 500     | 3      | 2          | ~2-3 hours     | Quick test |
| 1,000   | 3      | 2          | ~3-4 hours     | Small demo |
| **3,000**   | **3**      | **2**          | **~8-10 hours**    | **Overnight demo** ✅ |
| 5,000   | 3      | 2          | ~12-16 hours   | Extended demo |
| 10,000  | 3      | 4          | ~24-36 hours   | Production-lite |

---

## 🎓 LoRA Configuration Explained

### Current Settings (Optimized for Demo)

```python
LoraConfig(
    r=8,                    # Rank: number of trainable parameters
    lora_alpha=32,          # Scaling factor (alpha/r = 4x amplification)
    lora_dropout=0.1,       # 10% dropout to prevent overfitting
    target_modules=[        # Which model layers to fine-tune
        "q_proj", "o_proj", "k_proj", "v_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)
```

### Parameter Breakdown

| Parameter | Current | Purpose | Demo Impact |
|-----------|---------|---------|-------------|
| **r** (Rank) | 8 | Number of trainable parameters added | Good balance: fast training, decent capacity |
| **lora_alpha** | 32 | How much LoRA influences output (32/8 = 4x) | Strong learning of Amazon data |
| **lora_dropout** | 0.1 | Prevents overfitting on limited data | Helps generalization |

---

## 🏭 Full Production Fine-Tuning (All 131K Products)

### Recommended Configuration for Complete Dataset

```python
# Full dataset training
max_samples = 131262    # All products in trn.json
epochs = 3-5            # 3 for speed, 5 for better quality
batch_size = 4          # Increase if you have more GPU memory

# Enhanced LoRA configuration
LoraConfig(
    r=16,               # Double the rank for more capacity
    lora_alpha=32,      # Keep 2x amplification
    lora_dropout=0.05,  # Lower dropout with more data
    target_modules=[    # Same target modules
        "q_proj", "o_proj", "k_proj", "v_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)

# Training arguments
TrainingArguments(
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch size = 16
    learning_rate=1e-4,             # Lower LR for stability
    warmup_steps=100,               # Gradual warmup
    save_strategy="steps",
    save_steps=5000,                # Save checkpoints every 5K steps
    logging_steps=100,
)
```

### Full Training Estimates

| Configuration | Training Time | Hardware Required |
|--------------|---------------|-------------------|
| 131K × 3 epochs | ~3-4 days | 24GB RAM, Apple M4 Pro |
| 131K × 5 epochs | ~5-7 days | 24GB RAM, Apple M4 Pro |
| 131K × 3 epochs | ~12-18 hours | NVIDIA A100 40GB |

**Production Recommendations:**
- Use `r=16` or `r=32` for better memorization
- Train for 5-10 epochs for production quality
- Implement evaluation set to monitor overfitting
- Save checkpoints every 5,000 steps
- Use learning rate scheduling (warmup + decay)

---

## 📖 How to Use the Application

### 1. Enter Hugging Face Token
- Paste your token in the sidebar
- Required to download the Gemma-2B model

### 2. Configure Training (Optional)
Adjust in the sidebar:
- **Max Samples**: 3,000 (default for overnight)
- **Epochs**: 3 (default)
- **Batch Size**: 2 (safe for 24GB RAM)

### 3. Preview Dataset (Optional)
- Check "Dataset Preview" tab
- See sample Amazon products

### 4. Start Fine-Tuning
- Go to "Fine-tuning Process" tab
- Click "Start Fine-tuning Process"
- Leave running overnight (~8-10 hours)

### 5. Compare Results
- Go to "Compare Before/After Fine-tuning" tab
- Enter a question like: "What is Mog's Kittens?"
- Click "Generate with Original Model" (before)
- Click "Generate with Fine-tuned Model" (after)
- See the improvement and training references!

---

## 🔬 Technical Details

### LoRA (Low-Rank Adaptation)

LoRA adds small trainable matrices to the model's attention layers, allowing fine-tuning with:
- **99.9% fewer trainable parameters** (vs full fine-tuning)
- **3x less memory** required
- **Faster training** (hours instead of days)
- **Easy to share** (only need to share LoRA adapters, not full model)

### Why These Parameters Work

**For Demo (3,000 samples, r=8):**
- Trains quickly (overnight)
- Learns Amazon product style and formatting
- Good enough to demonstrate fine-tuning effectiveness
- Won't perfectly memorize all products (expected)

**For Production (131K samples, r=16-32):**
- Comprehensive coverage of all products
- Better memorization and recall
- More robust to variations in queries
- Production-ready quality

---

## 📈 Expected Results

### Before Fine-Tuning
- Generic responses
- May hallucinate information
- No knowledge of specific Amazon products
- Creative but incorrect

### After Fine-Tuning (3,000 samples)
- ✅ Amazon-style product descriptions
- ✅ Better formatting and structure
- ✅ Stops generating follow-up questions
- ✅ More focused responses
- ⚠️ May still hallucinate for unseen products (normal with limited training)

### After Full Fine-Tuning (131K samples)
- ✅ Comprehensive product knowledge
- ✅ Better recall of specific products
- ✅ More accurate descriptions
- ✅ Production-ready quality

---

## 🛠️ Project Structure

```
FIAP-fase-3-fine-tunning/
├── data/
│   └── trn.json                 # Amazon dataset (131,262 products)
├── models/
│   └── gemma-2b-finetuned/     # Fine-tuned LoRA adapters
├── chroma_db/                   # Vector database for references
├── src/
│   ├── streamlit_app.py        # Main UI application
│   ├── model_utils.py          # Fine-tuning and inference logic
│   ├── api_model_utils.py      # Model loading utilities
│   └── rag_utils.py            # RAG system for references
├── requirements.txt
└── README.md
```

---

## 🎬 Creating Your Demo Video

### What to Show (10 minutes max)

1. **Introduction (1 min)**
   - Explain the challenge: fine-tune Gemma on Amazon data
   - Show the dataset preview

2. **Before Fine-Tuning (2 min)**
   - Ask a question about a product
   - Show the original model's generic/incorrect response

3. **Fine-Tuning Process (1 min)**
   - Show the configuration (3,000 samples, 3 epochs)
   - Explain LoRA parameters briefly
   - (Skip the actual training - just show it started)

4. **After Fine-Tuning (3 min)**
   - Ask the same question
   - Show the improved, Amazon-style response
   - Show the training references (proof of learning)

5. **Technical Explanation (2 min)**
   - Explain LoRA and why it's efficient
   - Show the training parameters used
   - Mention scalability to full 131K dataset

6. **Conclusion (1 min)**
   - Summarize improvements
   - Mention limitations (limited training data)
   - Explain production path (full dataset training)

---

## ⚠️ Important Notes

### Limitations

1. **Small training set (3,000)** won't memorize all products perfectly
2. **Hallucination** may still occur for unseen products
3. **LoRA r=8** is parameter-efficient but has limited capacity
4. **Single model** (Gemma-2B) - not the largest available

### These Are Normal!

Fine-tuning 3,000 examples on a 2B model with r=8 LoRA is a **proof of concept**, not production deployment. The model learns:
- ✅ The pattern of Amazon product Q&A
- ✅ The style and format of descriptions
- ✅ General product knowledge

But won't perfectly memorize every product. That's expected and acceptable for a demo!

---

## 📚 References

- **Dataset**: [AmazonTitles-1.3MM](https://drive.google.com/file/d/12zH4mL2RX8iSvH0VCNnd3QxO4DzuHWnK/view)
- **Model**: [Google Gemma-2B](https://huggingface.co/google/gemma-2b)
- **LoRA Paper**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **PEFT Library**: [Hugging Face PEFT](https://github.com/huggingface/peft)

---

## 🤝 Tech Challenge Compliance

✅ **Fine-tuning execution**: LoRA-based PEFT on Gemma-2B  
✅ **Dataset preparation**: Amazon product titles + descriptions  
✅ **Before/After comparison**: Streamlit UI shows both  
✅ **Training documentation**: This README + code comments  
✅ **References/Sources**: ChromaDB-based RAG system  
✅ **Video demonstration**: 10-minute walkthrough  

---

## 💡 Tips for Success

1. **Start training tonight** with default settings (3,000 samples, 3 epochs)
2. **Let it run overnight** (~8-10 hours)
3. **Test in the morning** - try various product queries
4. **Record your demo** showing before/after comparison
5. **Be honest about limitations** in your video

**The goal is demonstrating fine-tuning works, not achieving perfection!** ✨

---

## 📞 Support

For questions about the Tech Challenge, use the FIAP Discord channel.

---

**Good luck with your demo! 🚀**

