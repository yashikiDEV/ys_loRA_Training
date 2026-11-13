# 🚀 ys_LoRA_Training

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yashikiDEV/ys_loRA_Training/blob/main/ys_LoRA_Training.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Hugging Face PEFT](https://img.shields.io/badge/PEFT-LoRA%20%2F%20QLoRA-orange.svg)](https://github.com/huggingface/peft)

**ys_LoRA_Training** é uma ferramenta aberta, modular e extremamente simples de usar para realizar o treinamento eficiente (**Fine-Tuning com LoRA / QLoRA**) de qualquer Modelo de Linguagem de Grande Porte (LLM) diretamente no **Google Colab** (incluindo a **GPU T4 gratuita**).

*An open, modular, and beginner-friendly Google Colab notebook to fine-tune any LLM (Qwen, Llama, Mistral, Gemma, Phi, GPT-2, etc.) with LoRA / QLoRA using Hugging Face & PEFT.*

---

## 🌐 Índice / Table of Contents
- [Português (Brasil)](#-português-brasil)
  - [Visão Geral](#visão-geral)
  - [Como Usar em 5 Passos](#como-usar-em-5-passos)
  - [Formatos de Dataset Suportados](#formatos-de-dataset-suportados)
  - [Dicionário de Parâmetros](#dicionário-de-parâmetros)
  - [Dicas para GPU com Pouca Memória (Colab Free T4)](#dicas-para-gpu-com-pouca-memória-colab-free-t4)
  - [Perguntas Frequentes & Solução de Problemas (FAQ)](#perguntas-frequentes--solução-de-problemas-faq)
- [English](#-english)
  - [Overview](#overview)
  - [5-Step Quickstart](#5-step-quickstart)
  - [Supported Dataset Formats](#supported-dataset-formats)
  - [Parameter Reference](#parameter-reference)
  - [Low VRAM Tips (Free Colab T4)](#low-vram-tips-free-colab-t4)
  - [FAQ & Troubleshooting](#faq--troubleshooting)
- [Estrutura do Projeto / Project Structure](#-estrutura-do-projeto--project-structure)
- [Licença / License](#-licença--license)

---

# 🇧🇷 Português (Brasil)

## Visão Geral

O objetivo do **ys_LoRA_Training** é democratizar o fine-tuning de modelos de IA. Qualquer pessoa deve conseguir abrir o notebook no Google Colab, alterar **apenas uma única célula de configuração** visual (`#@param`) e treinar o modelo que desejar.

### 🌟 Diferenciais
- **Configuração Centralizada**: Todas as opções (modelo, caminhos, dataset, hiperparâmetros) ficam reunidas na Célula 2. Nada de valores fixos espalhados pelo código.
- **Zero Exemplos Embutidos no Código**: O código não possui nenhum dado de treino escrito manualmente; ele carrega e valida seu dataset a partir de arquivos externos (`.json`, `.jsonl` ou `.txt`).
- **Suporte Multi-Arquitetura**: Funciona perfeitamente com **Qwen, Llama, Mistral, Gemma, Phi, Falcon, GPT-2**, etc., detectando as camadas lineares automaticamente (`TARGET_MODULES = "auto"`).
- **Compatível com o Colab Gratuito**: Otimizado para a GPU NVIDIA T4 (15 GB VRAM) usando quantização QLoRA em 4-bit, Gradient Checkpointing e otimizador `paged_adamw_8bit`.
- **Prevenção de Conflitos**: Remove automaticamente o `torchao` no início da instalação para evitar incompatibilidades de compilação com `bitsandbytes`.
- **Segurança no Google Drive & Versionamento**: Salva checkpoints intermediários e cria versões automáticas (`v0`, `v1`, `v2`...) acompanhadas do arquivo de metadados `info_treino.json`.
- **Teste Integrado**: Permite conversar com o modelo recém-treinado diretamente no notebook ou carregar qualquer versão salva anteriormente.

---

## Como Usar em 5 Passos

1. **Abra o Notebook e Ative a GPU**:
   - Clique no badge [Open In Colab](https://colab.research.google.com/github/yashikiDEV/ys_loRA_Training/blob/main/ys_LoRA_Training.ipynb).
   - No menu superior: `Ambiente de execução` -> `Alterar tipo de ambiente de execução` -> escolha **T4 GPU** -> Salvar.
2. **Execute a Instalação (Célula 1)**:
   - Desinstala conflitos do `torchao` e instala `transformers`, `peft`, `datasets`, `bitsandbytes`, `accelerate` e `trl`.
3. **Configure seu Treino (Célula 2)**:
   - Preencha o formulário interativo com o modelo desejado (ex.: `"Qwen/Qwen2.5-0.5B"`), o caminho do seu dataset no Drive e os hiperparâmetros.
4. **Execute as Células 3 a 9**:
   - Conecte o Google Drive, valide o ambiente, carregue o modelo/tokenizer, processe o dataset, aplique o LoRA, treine e salve os pesos da versão com o `info_treino.json`.
5. **Converse com seu Modelo (Célula 10)**:
   - Digite uma pergunta no formulário de teste e veja a resposta gerada com os novos pesos ajustados!

---

## Formatos de Dataset Suportados

O notebook aceita 3 formatos sem necessidade de alterar o código:

### 1. JSON (`.json`)
Pode ser uma lista de instruções ou de mensagens:
```json
[
  {
    "instruction": "Explique o que é LoRA.",
    "input": "",
    "output": "LoRA é uma técnica de adaptação de baixo rank para LLMs..."
  },
  {
    "instruction": "Como somar dois números em Python?",
    "input": "",
    "output": "def soma(a, b):\n    return a + b"
  }
]
```

### 2. JSONL (`.jsonl`)
Uma linha por documento JSON, ideal para conversações (`messages`):
```jsonl
{"messages": [{"role": "user", "content": "O que é QLoRA?"}, {"role": "assistant", "content": "QLoRA combina quantização 4-bit com LoRA..."}]}
{"messages": [{"role": "user", "content": "Qual a capital da França?"}, {"role": "assistant", "content": "A capital da França é Paris."}]}
```

### 3. TXT (`.txt`)
Texto puro separado pelo delimitador padrão `\n---\n`:
```text
<|usuario|>
Explique o que é learning rate.
<|assistente|>
A taxa de aprendizado controla o tamanho do passo do otimizador na atualização dos pesos.
---
<|usuario|>
O que é VRAM?
<|assistente|>
É a memória de vídeo dedicada da GPU, essencial para armazenar os pesos e ativações de LLMs.
```

---

## Dicionário de Parâmetros

Todos os parâmetros são configurados exclusivamente na **Célula 2**:

| Parâmetro | Tipo | Padrão | Descrição |
| :--- | :--- | :--- | :--- |
| `MODEL_NAME_OR_PATH` | String | `"Qwen/Qwen2.5-0.5B"` | Nome no Hugging Face Hub ou caminho local/Drive do modelo base. |
| `HF_TOKEN_SECRET_NAME` | String | `"HF_TOKEN"` | Nome da chave no Colab Secrets contendo o token HF (para modelos restritos como Llama/Gemma). |
| `HF_TOKEN_MANUAL` | String | `""` | Opção alternativa para colar o token HF diretamente caso não use os Secrets. |
| `PROJECT_NAME` | String | `"meu_projeto_lora"` | Nome identificador do seu projeto. |
| `BASE_DIR` | String | `"/content/drive/MyDrive/meu_projeto_lora"` | Pasta raiz no Google Drive onde versões e checkpoints serão guardados. |
| `DATASET_PATH` | String | `"/content/drive/.../dataset.json"` | Caminho completo para o arquivo do seu dataset. |
| `DATASET_FORMAT` | Dropdown | `"auto"` | Formato do dataset: `"auto"` (pela extensão), `"jsonl"`, `"json"` ou `"txt"`. |
| `TEXT_FIELD_NAME` | String | `"text"` | Nome da coluna para datasets de texto livre. |
| `MESSAGES_FIELD_NAME` | String | `"messages"` | Nome do campo contendo a lista de mensagens para conversação. |
| `USER_TOKEN` | String | `"<|usuario|>\n"` | Prefixo que delimita o início da entrada do usuário no template. |
| `ASSISTANT_TOKEN` | String | `"\n<|assistente|>\n"` | Prefixo que delimita a resposta do assistente no template. |
| `END_TOKEN` | String | `""` | Token de fim de turno. Se vazio `""`, usa automaticamente `tokenizer.eos_token`. |
| `SYSTEM_PROMPT` | String | `""` | Instrução de sistema inicial opcional para condicionar o modelo. |
| `NUM_EPOCHS` | Inteiro | `3` | Número de épocas completas de treinamento sobre o dataset. |
| `LEARNING_RATE` | Float | `2e-4` | Taxa de aprendizado inicial para o otimizador AdamW. |
| `BATCH_SIZE` | Inteiro | `2` | Tamanho do batch por passo em GPU (diminua para 1 se houver falta de VRAM). |
| `GRAD_ACCUM_STEPS` | Inteiro | `4` | Passos de acumulação. Lote efetivo = `BATCH_SIZE * GRAD_ACCUM_STEPS` (ex: 2 * 4 = 8). |
| `MAX_LENGTH` | Inteiro | `512` | Comprimento máximo de tokens por exemplo (trunca exemplos mais longos). |
| `WARMUP_RATIO` | Float | `0.05` | Percentual inicial de passos para aquecer o learning rate gradualmente. |
| `WEIGHT_DECAY` | Float | `0.01` | Coeficiente de regularização de decaimento de peso para evitar overfitting. |
| `RANDOM_SEED` | Inteiro | `42` | Semente pseudoaleatória para reprodutibilidade dos experimentos. |
| `LOGGING_STEPS` | Inteiro | `5` | Intervalo de passos para imprimir loss e métricas no console. |
| `SAVE_STEPS` | Inteiro | `50` | Intervalo de passos para salvar checkpoints intermediários no Drive (0 = só final). |
| `LORA_R` | Inteiro | `16` | Dimensão interna das matrizes LoRA (Rank `r`). Valores comuns: 8, 16, 32, 64. |
| `LORA_ALPHA` | Inteiro | `32` | Fator de multiplicação de escala do LoRA (normalmente `2 * r`). |
| `LORA_DROPOUT` | Float | `0.05` | Probabilidade de dropout nas camadas LoRA. |
| `TARGET_MODULES` | String | `"auto"` | `"auto"` detecta todas as camadas lineares da arquitetura, ou lista manual (ex: `"q_proj,v_proj"`). |
| `QUANTIZATION` | Dropdown | `"4-bit"` | `"4-bit"` (QLoRA NF4, ideal para T4), `"8-bit"` ou `"none"` (16-bit). |
| `GRADIENT_CHECKPOINTING`| Booleano | `True` | Reduz brutalmente o pico de VRAM recalculando ativações no backward pass. |
| `VERSIONING_MODE` | Dropdown | `"auto"` | `"auto"` cria pastas sequenciais (`v0`, `v1`, `v2`...) ou `"manual"`. |
| `MANUAL_VERSION_NAME` | String | `"v0"` | Nome customizado da versão caso o modo seja manual. |
| `RESUME_TRAINING` | Booleano | `False` | Retoma o treino a partir do último checkpoint da versão indicada. |
| `RESUME_VERSION_NAME` | String | `"v0"` | Versão a ser retomada quando `RESUME_TRAINING=True`. |
| `MERGE_LORA` | Booleano | `False` | Realiza a fusão dos adaptadores no modelo base e salva o modelo completo unificado. |

---

## Dicas para GPU com Pouca Memória (Colab Free T4)

A GPU Tesla T4 gratuita do Colab oferece aproximadamente 15 GB de VRAM. Para modelos de 7B a 14B parâmetros:
1. **Ative sempre `QUANTIZATION = "4-bit"`**: O QLoRA reduz o consumo de memória em mais de 70% com perda desprezível de capacidade.
2. **Mantenha `GRADIENT_CHECKPOINTING = True`**: Economiza múltiplos gigabytes de ativações temporárias.
3. **Ajuste o Batch Size**: Use `BATCH_SIZE = 1` ou `2` e compense com `GRAD_ACCUM_STEPS = 4` ou `8` para manter o batch efetivo equilibrado.
4. **Controle o `MAX_LENGTH`**: A atenção tem complexidade quadrática com o comprimento do contexto. Usar `MAX_LENGTH = 512` ou `1024` consome muito menos memória do que 4096 tokens.
5. **Limpe a memória antes de novos testes**: A Célula 12 executa `torch.cuda.empty_cache()` e coleta o lixo do Python para restabelecer a VRAM livre.

---

## Perguntas Frequentes & Solução de Problemas (FAQ)

### 1. Por que o código desinstala o `torchao` na Célula 1?
O ambiente do Google Colab traz o pacote `torchao` pré-instalado em certas imagens de PyTorch. Esse pacote frequentemente entra em conflito de ABI e gera erros ao carregar `bitsandbytes` (ex.: falhas de importação de kernel CUDA). A desinstalação no início garante que o `bitsandbytes` funcione com estabilidade total.

### 2. Como usar modelos fechados como Llama 3 ou Gemma?
Esses modelos exigem aprovação na página do Hugging Face. Após aprovado:
1. Crie um token no Hugging Face (`Settings` -> `Access Tokens`).
2. No Google Colab, clique no ícone de chave (Segredos / Secrets) na barra lateral esquerda.
3. Adicione um segredo com o nome `HF_TOKEN` e cole o valor.
4. Conceda permissão de acesso ao notebook. O notebook fará o login automático na Célula 4.

### 3. O que fazer ao receber erro `CUDA out of memory` (OOM)?
- Certifique-se de que `QUANTIZATION` está em `"4-bit"`.
- Reduza `BATCH_SIZE` para `1`.
- Reduza `MAX_LENGTH` para `256` ou `512`.
- Execute a Célula 12 (Limpeza de Memória) ou reinicie a sessão (`Ambiente de execução` -> `Reiniciar sessão`).

### 4. Como usar o modelo treinado fora do Colab (no Ollama, vLLM ou LM Studio)?
- Se você ativou `MERGE_LORA = True`, o notebook gera uma pasta `modelo_completo_merged`. Você pode converter essa pasta para o formato `.gguf` usando o `llama.cpp` e utilizá-la diretamente no Ollama ou LM Studio.
- Se salvou apenas os adaptadores LoRA, você pode carregá-los em qualquer aplicação Python usando a biblioteca `peft`:
  ```python
  from peft import PeftModel
  from transformers import AutoModelForCausalLM

  base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
  model = PeftModel.from_pretrained(base_model, "caminho/para/versoes/v0")
  ```

---

# 🇺🇸 English

## Overview

**ys_LoRA_Training** aims to make Large Language Model fine-tuning accessible to everyone. Any developer or enthusiast can open the Google Colab notebook, adjust **just a single visual configuration cell** (`#@param`), and fine-tune their desired model without touching complex boilerplate code.

### 🌟 Key Highlights
- **Single-Cell Configuration**: Edit all variables (model, paths, dataset, hyperparameters, LoRA rank) in Cell 2.
- **Zero Hardcoded Training Examples**: Datasets are cleanly loaded from external `.json`, `.jsonl`, or `.txt` files.
- **Multi-Architecture Support**: Works out-of-the-box with **Qwen, Llama, Mistral, Gemma, Phi, Falcon, GPT-2**, etc., automatically discovering linear layers via `TARGET_MODULES = "auto"`.
- **Colab Free Tier Optimized**: Runs smoothly on free NVIDIA T4 GPUs using 4-bit QLoRA, Gradient Checkpointing, and `paged_adamw_8bit`.
- **Torchao Conflict Prevention**: Automatically uninstalls pre-installed `torchao` to avoid bitsandbytes CUDA kernel incompatibilities.
- **Google Drive Versioning**: Persists checkpoints and models into structured version folders (`v0`, `v1`, `v2`...) with execution reports (`info_treino.json`).
- **Inference & Testing Cell**: Interactive chat cell with customizable temperature, top-p, and repetition penalties.

---

## 5-Step Quickstart

1. **Open in Colab & Enable GPU**:
   - Click the [Open In Colab](https://colab.research.google.com/github/yashikiDEV/ys_loRA_Training/blob/main/ys_LoRA_Training.ipynb) badge.
   - Go to `Runtime` -> `Change runtime type` -> select **T4 GPU** -> Save.
2. **Run Installation (Cell 1)**:
   - Uninstalls `torchao` and installs `transformers`, `peft`, `datasets`, `bitsandbytes`, `accelerate`, and `trl`.
3. **Customize Settings (Cell 2)**:
   - Configure your model ID (e.g. `"Qwen/Qwen2.5-0.5B"`), Drive paths, dataset file, and hyperparameters in the form.
4. **Execute Cells 3 through 9**:
   - Mounts Google Drive, validates environment, loads model/tokenizer, tokenizes dataset, injects LoRA, runs training, and saves version metadata.
5. **Chat with Your Model (Cell 10)**:
   - Test prompts interactively directly in the notebook!

---

## Supported Dataset Formats

### 1. JSON (`.json`)
```json
[
  {
    "instruction": "Explain LoRA in simple terms.",
    "input": "",
    "output": "LoRA freezes the base model weights and trains low-rank adapter matrices..."
  }
]
```

### 2. JSONL (`.jsonl`)
```jsonl
{"messages": [{"role": "user", "content": "What is QLoRA?"}, {"role": "assistant", "content": "QLoRA combines 4-bit quantization with LoRA adapters..."}]}
```

### 3. TXT (`.txt`)
```text
<|usuario|>
What is learning rate?
<|assistente|>
It controls the step size taken by the optimizer during weight updates.
---
<|usuario|>
What is VRAM?
<|assistente|>
Video RAM is dedicated memory on the GPU used to store model weights and activations.
```

---

## Parameter Reference

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `MODEL_NAME_OR_PATH` | String | `"Qwen/Qwen2.5-0.5B"` | Hugging Face repository ID or local Drive folder path. |
| `HF_TOKEN_SECRET_NAME` | String | `"HF_TOKEN"` | Colab Secret key name holding your Hugging Face access token. |
| `HF_TOKEN_MANUAL` | String | `""` | Direct token string fallback if not using Colab Secrets. |
| `PROJECT_NAME` | String | `"meu_projeto_lora"` | Project name used for folder organization. |
| `BASE_DIR` | String | `"/content/drive/MyDrive/meu_projeto_lora"` | Base directory in Google Drive. |
| `DATASET_PATH` | String | `"/content/drive/.../dataset.json"` | Path to your dataset file. |
| `DATASET_FORMAT` | Dropdown | `"auto"` | Format: `"auto"` (by file extension), `"jsonl"`, `"json"`, or `"txt"`. |
| `TEXT_FIELD_NAME` | String | `"text"` | Key name for plain text entries. |
| `MESSAGES_FIELD_NAME` | String | `"messages"` | Key name for chat/conversation list. |
| `USER_TOKEN` | String | `"<|usuario|>\n"` | User turn delimiter prefix. |
| `ASSISTANT_TOKEN` | String | `"\n<|assistente|>\n"` | Assistant turn delimiter prefix. |
| `END_TOKEN` | String | `""` | End-of-turn delimiter. If empty `""`, uses `tokenizer.eos_token`. |
| `SYSTEM_PROMPT` | String | `""` | Optional system prompt prefix. |
| `NUM_EPOCHS` | Integer | `3` | Number of training epochs. |
| `LEARNING_RATE` | Float | `2e-4` | Initial AdamW learning rate. |
| `BATCH_SIZE` | Integer | `2` | Per-device training batch size. |
| `GRAD_ACCUM_STEPS` | Integer | `4` | Gradient accumulation steps (Effective batch = `BATCH_SIZE * GRAD_ACCUM_STEPS`). |
| `MAX_LENGTH` | Integer | `512` | Maximum token sequence length. |
| `WARMUP_RATIO` | Float | `0.05` | Ratio of training steps for linear warmup. |
| `WEIGHT_DECAY` | Float | `0.01` | Weight decay regularization factor. |
| `RANDOM_SEED` | Integer | `42` | Seed for reproducibility. |
| `LOGGING_STEPS` | Integer | `5` | Steps between training log outputs. |
| `SAVE_STEPS` | Integer | `50` | Steps between intermediate checkpoints (0 for end only). |
| `LORA_R` | Integer | `16` | LoRA rank dimension `r`. |
| `LORA_ALPHA` | Integer | `32` | LoRA alpha scaling factor (usually `2 * r`). |
| `LORA_DROPOUT` | Float | `0.05` | LoRA dropout probability. |
| `TARGET_MODULES` | String | `"auto"` | `"auto"` detects all linear modules, or comma-separated list. |
| `QUANTIZATION` | Dropdown | `"4-bit"` | `"4-bit"` (QLoRA NF4), `"8-bit"`, or `"none"` (16-bit). |
| `GRADIENT_CHECKPOINTING`| Boolean | `True` | Recalculates activations during backward pass to save VRAM. |
| `VERSIONING_MODE` | Dropdown | `"auto"` | Automatic (`v0`, `v1`, `v2`...) or `"manual"`. |
| `MANUAL_VERSION_NAME` | String | `"v0"` | Custom version label if manual mode is selected. |
| `RESUME_TRAINING` | Boolean | `False` | Resume training from existing checkpoint. |
| `RESUME_VERSION_NAME` | String | `"v0"` | Target version to resume if `RESUME_TRAINING=True`. |
| `MERGE_LORA` | Boolean | `False` | Merges LoRA weights into base model and exports standalone model. |

---

## Low VRAM Tips (Free Colab T4)

1. Keep `QUANTIZATION = "4-bit"` enabled for 7B+ models.
2. Keep `GRADIENT_CHECKPOINTING = True`.
3. Set `BATCH_SIZE = 1` and increase `GRAD_ACCUM_STEPS = 8` or `16`.
4. Keep `MAX_LENGTH` between `256` and `512`.
5. Run Cell 12 to free VRAM caches if restarting experiments.

---

## FAQ & Troubleshooting

### Why does Cell 1 uninstall `torchao`?
Google Colab's default environment contains a pre-installed `torchao` package that causes ABI and CUDA binding conflicts with `bitsandbytes`. Uninstalling it upfront guarantees smooth 4-bit and 8-bit quantization.

### How to access gated models (e.g. Llama 3, Gemma)?
Request access on the Hugging Face model card, create a Read token in Hugging Face Settings, and save it in Colab's **Secrets** tab under the key `HF_TOKEN`.

---

## 📁 Estrutura do Projeto / Project Structure

```text
├── ys_LoRA_Training.ipynb   # Caderno principal do Google Colab / Main Colab notebook
├── README.md                # Documentação bilíngue completa / Complete bilingual docs
├── exemplo_dataset.json     # Exemplo de dataset em formato JSON / Example JSON dataset
├── exemplo_dataset.jsonl    # Exemplo de dataset em formato JSONL / Example JSONL dataset
├── exemplo_dataset.txt      # Exemplo de dataset em formato TXT / Example TXT dataset
├── .gitignore               # Arquivos ignorados pelo Git / Git ignore rules
└── LICENSE                  # Licença MIT / MIT License
```

---

## 📜 Licença / License

Distribuído sob a licença **MIT**. Consulte o arquivo [LICENSE](LICENSE) para mais detalhes.

*Distributed under the **MIT** License. See [LICENSE](LICENSE) for more details.*
