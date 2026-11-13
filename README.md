
# ys_LoRA_Training

Fine-tuning de LLMs com LoRA/QLoRA, simples e modular, direto no Google Colab.

O ys_LoRA_Training é um notebook aberto que permite treinar adaptadores LoRA em
modelos de linguagem como Qwen, Llama, Mistral, Gemma e Phi, sem precisar mexer
no código. Você edita apenas uma célula de configuração, aponta para o seu
dataset e executa as células em sequência.

## Principais recursos

- **Configuração centralizada**: todos os parâmetros (modelo, caminhos,
  hiperparâmetros, LoRA, quantização) ficam em um único painel de formulário.
- **Dataset externo**: lê arquivos `.json`, `.jsonl` e `.txt`, nos formatos
  de conversação (`messages`), instrução/entrada/saída e texto livre.
  Nenhum dado de treino fica embutido no código.
- **Modelo à sua escolha**: qualquer modelo causal do Hugging Face ou
  salvo no Drive, com detecção automática das camadas-alvo do LoRA.
- **Amigável ao Colab gratuito**: quantização 4-bit/8-bit (QLoRA), otimizador
  paginado e gradient checkpointing para caber em GPUs T4.
- **Versionamento no Google Drive**: cada treino é salvo em `v0`, `v1`, `v2`...
  com checkpoints e um relatório `info_treino.json` (data, duração, loss,
  hiperparâmetros e hardware).
- **Retomada de treino** a partir de checkpoints.
- **Merge opcional** do adaptador no modelo base.
- **Teste integrado**: chat de geração com o mesmo template usado no treino,
  e carregamento de versões salvas para inferência sem retreinar.

## Como usar

1. Ative uma GPU no Colab (Ambiente de execução → T4 GPU).
2. Execute a Célula 1 (instalação).
3. Edite a Célula 2 (configuração).
4. Execute as células 3 a 9 em sequência.
5. Teste o modelo na Célula 10.

## Requisitos

- Conta Google (Colab e Drive)
- Dataset em `.json`, `.jsonl` ou `.txt`
- Token do Hugging Face (apenas para modelos fechados)

## Licença

MIT
