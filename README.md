# Análise de sentimento em avaliações de e-commerce com BERTimbau

Trabalho final da disciplina de Machine Learning.
**Dupla:** Guilherme Theis e Paulo Henrique.

## Tema

Classificação automática do **sentimento** de avaliações de produtos escritas em português, com redes
neurais.

| | |
|---|---|
| **Tarefa** | classificação de texto em 3 classes |
| **Entrada** | título + texto da avaliação (`review_title` + `review_text`) |
| **Saída** | sentimento derivado da nota `overall_rating`: 1–2 → `negativo`, 3 → `neutro`, 4–5 → `positivo` |
| **Modelo** (próxima etapa) | BERTimbau (`neuralmind/bert-base-portuguese-cased`) com *fine-tuning* |
| **Baseline** | TF-IDF + regressão logística |
| **Métrica principal** | F1-macro (classes desbalanceadas) |

## Fonte dos dados

**B2W-Reviews01**: cerca de 132 mil avaliações de clientes da Americanas.com, coletadas entre janeiro e
maio de 2018.

- Repositório oficial: <https://github.com/americanas-tech/b2w-reviews01> (antigo `b2wdigital/b2w-reviews01`)
- Licença: **Creative Commons BY-NC-SA 4.0**, com crédito à **B2W Digital**, uso **não comercial** e
  obras derivadas compartilhadas pela mesma licença. Os dados **não** são versionados neste
  repositório; o script de download os obtém da fonte original.
- Artigo de referência: *B2W-Reviews01: an open product reviews corpus* (STIL 2019), disponível no
  próprio repositório (`b2wreviews01_stil2019.pdf`).

`src/download.py` baixa o arquivo de um **commit fixo** do repositório oficial e confere o SHA-256, então
todos os integrantes trabalham exatamente com os mesmos dados.

### Por que a base do Consumidor.gov.br foi descartada

O tema original era classificar **reclamações** do Consumidor.gov.br (Senacon) pelo campo "Grupo
Problema". Antes de começar, validamos a fonte (novembro de 2026):

1. **Os dados abertos não têm texto livre.** Um arquivo mensal ("Base Completa", nov/2023, 133 mil
   linhas) tem 30 colunas, todas categóricas ou datas: Gestor, Canal de Origem, Região, UF, Cidade,
   Sexo, Faixa Etária, datas de abertura/resposta/finalização, Nome Fantasia, Segmento de Mercado,
   Área, Assunto, Grupo Problema, Problema, Como Comprou Contratou, Situação, Avaliação, Nota do
   Consumidor etc. **O relato do consumidor não é publicado.** Sem texto não há como treinar um
   modelo de linguagem como o BERTimbau.
2. **Os links de download estão quebrados.** O catálogo no dados.gov.br lista 89 arquivos, mas todos
   apontam para `dados.mj.gov.br`, domínio que deixou de existir (o DNS não resolve). O próprio
   verificador do portal marca os links como "Erro na conexão". Na página de dados abertos do
   consumidor.gov.br, a listagem de arquivos retornava erro 504. Só foi possível inspecionar as colunas
   por uma cópia no Internet Archive.

Escolhemos então o B2W-Reviews01, que tem texto livre, rótulo, volume suficiente e download estável. A
tarefa passou de "categoria do problema" para "sentimento".

## Como rodar

Requisitos: **Python 3.12** e acesso à internet (para baixar o corpus e o tokenizer do BERTimbau no
Hugging Face Hub). Não é preciso GPU nem PyTorch nesta etapa.

```bash
# 1. ambiente virtual
python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# Windows (Git Bash):    source .venv/Scripts/activate
# Linux/macOS:           source .venv/bin/activate

# 2. dependências (versões fixadas)
pip install -r requirements.txt

# 3. dados brutos -> data/raw/B2W-Reviews01.csv (cerca de 50 MB)
python -m src.download

# 4. executa o notebook do início ao fim, salvando as saídas
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda_preprocessamento.ipynb
```

O passo 3 é opcional: o notebook chama o download sozinho se o arquivo não existir. A execução completa
leva cerca de 1 a 2 minutos numa máquina comum e gera:

- `data/processed/train.parquet`, `val.parquet`, `test.parquet`: conjuntos prontos;
- `data/processed/label_mapping.json`: classe ↔ id (`negativo` = 0, `neutro` = 1, `positivo` = 2);
- `data/processed/class_weights.json`: pesos de classe calculados no treino;
- `data/processed/preprocess_config.json`: `max_length`, colunas de entrada, cortes temporais etc.;
- `reports/figures/*.png`: gráficos da análise exploratória.

As funções de `src/` têm *doctests*: `python -m doctest src/*.py`.

## Estrutura

```
.
├── README.md
├── requirements.txt          # versões fixadas (Python 3.12)
├── data/
│   ├── raw/                  # CSV baixado (não versionado)
│   └── processed/            # train/val/test em parquet + JSONs (não versionados)
├── notebooks/
│   └── 01_eda_preprocessamento.ipynb   # EDA + pré-processamento, executado com as saídas
├── src/
│   ├── download.py           # download reproduzível (commit fixo + SHA-256)
│   ├── anonymize.py          # remove colunas do avaliador; mascara e-mail, CPF, CNPJ, telefone, CEP, rastreio, URL
│   ├── clean.py              # normalização (encoding, NFC, espaços), texto para TF-IDF, funil de limpeza
│   ├── split.py              # divisão temporal 70/15/15 e pesos de classe (só treino)
│   └── augment.py            # data augmentation estilo EDA (avaliada e não aplicada)
└── reports/figures/          # gráficos gerados pelo notebook
```

## Resumo do pré-processamento

| Etapa | Decisão |
|---|---|
| Anonimização | colunas `reviewer_id`, ano de nascimento, gênero e estado removidas; regex no texto (o corpus quase não tem dados pessoais: 87 URLs, 4 códigos de rastreio, 1 CNPJ) |
| Limpeza | nulos, vazios, duplicados exatos e de conteúdo, entradas com < 10 palavras (título + texto): 132.373 → 121.138 linhas (−8,5%) |
| Rótulos contraditórios | nota 1 + "recomendaria" e nota 5 + "não recomendaria" **mantidos**: numa amostra lida, o texto concorda com a nota em cerca de 90% dos casos |
| Texto para o BERT | correção de encoding, NFC, espaços; **sem** minúsculas e **sem** remover acentos (modelo *cased*) |
| Texto para o baseline | minúsculas, sem pontuação, sem stopwords (negações como "não" e "mas" mantidas) |
| Entrada | `título. texto`: o título melhora o F1-macro do baseline em cerca de 5 pontos na validação |
| Divisão | **temporal** por `submission_date`: treino 84.797 (jan–16/abr), validação 18.170 (até 07/mai), teste 18.171 (até 31/mai) |
| Desbalanceamento | cerca de 61% positivo, 27% negativo e 13% neutro; tratado com **pesos de classe** do treino |
| Data augmentation | avaliada e **não aplicada**: mesmo ganho que pesos ou oversampling, e apaga negações em cerca de 15% dos casos |
| `max_length` | **128** tokens (P95 do treino = 104; cobre cerca de 97% sem truncar) |

Baseline de referência (TF-IDF + regressão logística com pesos de classe, validação): **F1-macro ≈ 0,76**,
com o neutro como classe mais difícil (F1 ≈ 0,50).
