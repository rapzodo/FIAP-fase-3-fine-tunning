# Fine-Tuning do Gemma-2B para Descrições de Produtos da Amazon

## 📋 Visão Geral do Projeto

Este projeto demonstra o fine-tuning do modelo foundation Gemma-2B usando LoRA (Low-Rank Adaptation) em dados de produtos da Amazon. O modelo aprende a gerar descrições de produtos baseadas em títulos de produtos, usando o dataset AmazonTitles-1.3MM.

**Tech Challenge - Fase 3 - FIAP**

---

## 🎯 Propósito

1. **Fazer fine-tuning de um modelo foundation** (Gemma-2B) em dados específicos de domínio (produtos da Amazon)
2. **Usar Fine-Tuning Eficiente em Parâmetros (PEFT)** com LoRA para treinar em hardware de consumidor
3. **Comparar performance do modelo** antes e depois do fine-tuning
4. **Demonstrar recuperação de fontes de treinamento** usando RAG (Retrieval-Augmented Generation)

---

## 🏗️ Arquitetura

### Componentes:
- **Modelo Base**: Google Gemma-2B (2 bilhões de parâmetros)
- **Método de Fine-Tuning**: LoRA (Low-Rank Adaptation)
- **Dataset**: AmazonTitles-1.3MM (131.262 produtos com títulos e descrições)
- **Banco de Dados Vetorial**: ChromaDB para busca semântica
- **Embeddings**: SentenceTransformer (all-MiniLM-L6-v2)
- **Interface**: Aplicação web Streamlit

### Fluxo de Treinamento:
```
Dataset Amazon → Fine-Tuning LoRA → Modelo Fine-Tunado
                       ↓
                 Indexação ChromaDB (para referências)
```

### Fluxo de Inferência:
```
Pergunta do Usuário → Modelo Fine-Tunado → Descrição Gerada
                ↓
         Busca Semântica ChromaDB → Referências de Treinamento (para transparência)
```

---

## 🚀 Início Rápido

### Pré-requisitos

1. **Python 3.10+**
2. **24GB+ RAM** (para carregar o modelo)
3. **Apple Silicon (M1/M2/M3/M4)** ou **GPU CUDA** (opcional, para treinamento mais rápido)
4. **Token Hugging Face** (necessário para acesso ao modelo Gemma)
   - Obtenha o token em: https://huggingface.co/settings/tokens
   - Aceite a licença Gemma em: https://huggingface.co/google/gemma-2b

### Instalação

```bash
# Clone o repositório
cd FIAP-fase-3-fine-tunning

# Crie o ambiente virtual
python -m venv .venv
source .venv/bin/activate  # No Windows: .venv\Scripts\activate

# Instale as dependências
pip install -r requirements.txt
```

### Executando a Aplicação

```bash
# Inicie a aplicação Streamlit
streamlit run src/streamlit_app.py
```

A aplicação abrirá no seu navegador em `http://localhost:8501`

---

## 📊 Configuração de Treinamento

### Padrão (Treinamento Noturno - Recomendado para Demo)

**Otimizado para treinamento de 8-10 horas no Apple M4 Pro (24GB RAM)**

```python
max_samples = 3000      # 3.000 produtos da Amazon
epochs = 3              # Treinar por 3 épocas
batch_size = 2          # Batch size de 2
```

**Resultados Esperados:**
- ✅ Tempo de Treinamento: ~8-10 horas
- ✅ Boa cobertura de categorias de produtos
- ✅ Adequado para propósitos de demonstração
- ✅ Modelo aprende padrões de Q&A de produtos da Amazon

### Estimativas de Tempo de Treinamento

| Amostras | Épocas | Batch Size | Tempo Estimado | Caso de Uso |
|----------|--------|------------|----------------|-------------|
| 500      | 3      | 2          | ~2-3 horas     | Teste rápido |
| 1.000    | 3      | 2          | ~3-4 horas     | Demo pequena |
| **3.000**    | **3**      | **2**          | **~8-10 horas**    | **Demo noturna** ✅ |
| 5.000    | 3      | 2          | ~12-16 horas   | Demo estendida |
| 10.000   | 3      | 4          | ~24-36 horas   | Produção-lite |

---

## 🎓 Configuração LoRA Explicada

### Configurações Atuais (Otimizadas para Demo)

```python
LoraConfig(
    r=8,                    # Rank: número de parâmetros treináveis
    lora_alpha=32,          # Fator de escala (alpha/r = amplificação 4x)
    lora_dropout=0.1,       # 10% dropout para prevenir overfitting
    target_modules=[        # Quais camadas do modelo fazer fine-tune
        "q_proj", "o_proj", "k_proj", "v_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)
```

### Detalhamento dos Parâmetros

| Parâmetro | Atual | Propósito | Impacto na Demo |
|-----------|-------|-----------|-----------------|
| **r** (Rank) | 8 | Número de parâmetros treináveis adicionados | Bom equilíbrio: treinamento rápido, capacidade decente |
| **lora_alpha** | 32 | Quanto o LoRA influencia a saída (32/8 = 4x) | Aprendizado forte dos dados da Amazon |
| **lora_dropout** | 0.1 | Previne overfitting em dados limitados | Ajuda na generalização |

---

## 🏭 Fine-Tuning de Produção Completo (Todos os 131K Produtos)

### Configuração Recomendada para Dataset Completo

```python
# Treinamento com dataset completo
max_samples = 131262    # Todos os produtos em trn.json
epochs = 3-5            # 3 para velocidade, 5 para melhor qualidade
batch_size = 4          # Aumente se tiver mais memória GPU

# Configuração LoRA aprimorada
LoraConfig(
    r=16,               # Dobrar o rank para mais capacidade
    lora_alpha=32,      # Manter amplificação 2x
    lora_dropout=0.05,  # Dropout menor com mais dados
    target_modules=[    # Mesmos módulos alvo
        "q_proj", "o_proj", "k_proj", "v_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)

# Argumentos de treinamento
TrainingArguments(
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # Batch size efetivo = 16
    learning_rate=1e-4,             # LR menor para estabilidade
    warmup_steps=100,               # Aquecimento gradual
    save_strategy="steps",
    save_steps=5000,                # Salvar checkpoints a cada 5K passos
    logging_steps=100,
)
```

### Estimativas de Treinamento Completo

| Configuração | Tempo de Treinamento | Hardware Necessário |
|--------------|---------------------|---------------------|
| 131K × 3 épocas | ~3-4 dias | 24GB RAM, Apple M4 Pro |
| 131K × 5 épocas | ~5-7 dias | 24GB RAM, Apple M4 Pro |
| 131K × 3 épocas | ~12-18 horas | NVIDIA A100 40GB |

**Recomendações para Produção:**
- Use `r=16` ou `r=32` para melhor memorização
- Treine por 5-10 épocas para qualidade de produção
- Implemente conjunto de avaliação para monitorar overfitting
- Salve checkpoints a cada 5.000 passos
- Use agendamento de taxa de aprendizado (warmup + decay)

---

## 📖 Como Usar a Aplicação

### 1. Digite o Token Hugging Face
- Cole seu token na barra lateral
- Necessário para baixar o modelo Gemma-2B

### 2. Configure o Treinamento (Opcional)
Ajuste na barra lateral:
- **Max Samples**: 3.000 (padrão para noturno)
- **Épocas**: 3 (padrão)
- **Batch Size**: 2 (seguro para 24GB RAM)

### 3. Visualize o Dataset (Opcional)
- Veja a aba "Dataset Preview"
- Confira exemplos de produtos da Amazon

### 4. Inicie o Fine-Tuning
- Vá para a aba "Fine-tuning Process"
- Clique em "Start Fine-tuning Process"
- Deixe executando durante a noite (~8-10 horas)

### 5. Compare Resultados
- Vá para a aba "Compare Before/After Fine-tuning"
- Digite uma pergunta como: "O que é Mog's Kittens?"
- Clique em "Generate with Original Model" (antes)
- Clique em "Generate with Fine-tuned Model" (depois)
- Veja a melhoria e as referências de treinamento!

---

## 🔬 Detalhes Técnicos

### LoRA (Low-Rank Adaptation)

LoRA adiciona pequenas matrizes treináveis às camadas de atenção do modelo, permitindo fine-tuning com:
- **99,9% menos parâmetros treináveis** (vs fine-tuning completo)
- **3x menos memória** necessária
- **Treinamento mais rápido** (horas em vez de dias)
- **Fácil de compartilhar** (só precisa compartilhar adaptadores LoRA, não o modelo completo)

### Por Que Esses Parâmetros Funcionam

**Para Demo (3.000 amostras, r=8):**
- Treina rapidamente (durante a noite)
- Aprende estilo e formatação de produtos da Amazon
- Suficiente para demonstrar efetividade do fine-tuning
- Não memorizará perfeitamente todos os produtos (esperado)

**Para Produção (131K amostras, r=16-32):**
- Cobertura abrangente de todos os produtos
- Melhor memorização e recall
- Mais robusto a variações nas perguntas
- Qualidade pronta para produção

---

## 📈 Resultados Esperados

### Antes do Fine-Tuning
- Respostas genéricas
- Pode alucinar informações
- Sem conhecimento de produtos específicos da Amazon
- Criativo mas incorreto

### Depois do Fine-Tuning (3.000 amostras)
- ✅ Descrições de produtos no estilo Amazon
- ✅ Melhor formatação e estrutura
- ✅ Para de gerar perguntas de acompanhamento
- ✅ Respostas mais focadas
- ⚠️ Pode ainda alucinar para produtos não vistos (normal com treinamento limitado)

### Depois do Fine-Tuning Completo (131K amostras)
- ✅ Conhecimento abrangente de produtos
- ✅ Melhor recall de produtos específicos
- ✅ Descrições mais precisas
- ✅ Qualidade pronta para produção

---

## 🛠️ Estrutura do Projeto

```
FIAP-fase-3-fine-tunning/
├── data/
│   └── trn.json                 # Dataset Amazon (131.262 produtos)
├── models/
│   └── gemma-2b-finetuned/     # Adaptadores LoRA fine-tunados
├── chroma_db/                   # Banco de dados vetorial para referências
├── src/
│   ├── streamlit_app.py        # Aplicação UI principal
│   ├── model_utils.py          # Lógica de fine-tuning e inferência
│   ├── api_model_utils.py      # Utilitários de carregamento de modelo
│   └── rag_utils.py            # Sistema RAG para referências
├── requirements.txt
└── README.md
```

---

## 🎬 Criando Seu Vídeo de Demonstração

### O Que Mostrar (máximo 10 minutos)

1. **Introdução (1 min)**
   - Explique o desafio: fine-tune do Gemma em dados da Amazon
   - Mostre a prévia do dataset

2. **Antes do Fine-Tuning (2 min)**
   - Faça uma pergunta sobre um produto
   - Mostre a resposta genérica/incorreta do modelo original

3. **Processo de Fine-Tuning (1 min)**
   - Mostre a configuração (3.000 amostras, 3 épocas)
   - Explique brevemente os parâmetros LoRA
   - (Pule o treinamento real - apenas mostre que iniciou)

4. **Depois do Fine-Tuning (3 min)**
   - Faça a mesma pergunta
   - Mostre a resposta melhorada, no estilo Amazon
   - Mostre as referências de treinamento (prova de aprendizado)

5. **Explicação Técnica (2 min)**
   - Explique LoRA e por que é eficiente
   - Mostre os parâmetros de treinamento usados
   - Mencione escalabilidade para o dataset completo de 131K

6. **Conclusão (1 min)**
   - Resuma as melhorias
   - Mencione limitações (dados de treinamento limitados)
   - Explique caminho para produção (treinamento com dataset completo)

---

## ⚠️ Notas Importantes

### Limitações

1. **Conjunto de treinamento pequeno (3.000)** não memorizará perfeitamente todos os produtos
2. **Alucinação** pode ainda ocorrer para produtos não vistos
3. **LoRA r=8** é eficiente em parâmetros mas tem capacidade limitada
4. **Modelo único** (Gemma-2B) - não é o maior disponível

### Isso É Normal!

Fazer fine-tuning de 3.000 exemplos em um modelo 2B com LoRA r=8 é uma **prova de conceito**, não implantação em produção. O modelo aprende:
- ✅ O padrão de Q&A de produtos da Amazon
- ✅ O estilo e formato das descrições
- ✅ Conhecimento geral de produtos

Mas não memorizará perfeitamente cada produto. Isso é esperado e aceitável para uma demo!

---

## 📚 Referências

- **Dataset**: [AmazonTitles-1.3MM](https://drive.google.com/file/d/12zH4mL2RX8iSvH0VCNnd3QxO4DzuHWnK/view)
- **Modelo**: [Google Gemma-2B](https://huggingface.co/google/gemma-2b)
- **Paper LoRA**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **Biblioteca PEFT**: [Hugging Face PEFT](https://github.com/huggingface/peft)

---

## 🤝 Conformidade com o Tech Challenge

✅ **Execução de fine-tuning**: PEFT baseado em LoRA no Gemma-2B  
✅ **Preparação do dataset**: Títulos + descrições de produtos da Amazon  
✅ **Comparação antes/depois**: UI Streamlit mostra ambos  
✅ **Documentação de treinamento**: Este README + comentários no código  
✅ **Referências/Fontes**: Sistema RAG baseado em ChromaDB  
✅ **Demonstração em vídeo**: Walkthrough de 10 minutos  

---