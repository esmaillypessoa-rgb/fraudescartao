# Detecção de Fraudes em Cartões de Crédito

Projeto completo, simples e orientado ao mercado financeiro para classificar transações de cartão de crédito como legítimas ou suspeitas de fraude.

O formato segue uma entrega típica de bootcamp da DIO: problema de negócio, dataset, preparação dos dados, treinamento, avaliação, conclusão e instruções para reprodução. A principal diferença é que o projeto usa uma régua mais próxima de operação financeira: classe desbalanceada, validação temporal, `PR-AUC`, `recall`, matriz de confusão e seleção de limiar por custo.

## Problema de negócio

Em fraude financeira, acurácia isolada costuma enganar porque a quantidade de fraudes é muito menor que a de transações legítimas. Um modelo pode acertar quase tudo prevendo "não fraude" sempre, mas ainda assim deixar passar as perdas mais importantes.

Por isso, este projeto responde a três perguntas:

1. Qual modelo separa melhor fraudes de transações legítimas?
2. Qual limiar de alerta reduz melhor o custo esperado?
3. Quantas transações seriam enviadas para revisão e quantas fraudes seriam capturadas?

## Dataset

O projeto foi preparado para o dataset público **Credit Card Fraud Detection**, publicado na Kaggle pela ULB Machine Learning Group:

- Fonte: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Arquivo esperado: `data/raw/creditcard.csv`
- Target: `Class`, onde `1` indica fraude e `0` indica transação legítima.
- Features: `Time`, `Amount` e variáveis anonimizadas `V1` a `V28`.

Como a Kaggle exige aceite/licença e autenticação para download, o repositório não inclui o CSV real. Para rodar uma demonstração sem baixar dados, use o gerador sintético:

```bash
python -m fraud_detection.data --output data/raw/creditcard_sample.csv --rows 5000
```

## Referência técnica

O desenho do projeto usa como referência prática:

- **Credit card fraud detection using machine learning: a survey**: destaca os desafios de classe desbalanceada, mudança de distribuição ao longo do tempo, escolha de métricas e uso de curva precisão-recall.
- **Impact of Sampling Techniques and Data Leakage on XGBoost Performance in Credit Card Fraud Detection**: reforça o cuidado para aplicar qualquer tratamento de desbalanceamento apenas depois da separação treino/teste.
- **Minimizing the Societal Cost of Credit Card Fraud with Limited and Imbalanced Data**: mostra por que métricas de custo são importantes além de F1 ou acurácia.

Links completos ficam na seção **Referências**.

## Estrutura

```text
.
├── data/
│   └── raw/
├── models/
├── notebooks/
│   └── 01_roteiro_analise.py
├── reports/
├── src/
│   └── fraud_detection/
│       ├── config.py
│       ├── data.py
│       ├── predict.py
│       └── train.py
├── tests/
├── requirements.txt
└── README.md
```

## Como executar

Crie e ative um ambiente virtual:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Gere dados de exemplo:

```bash
python -m fraud_detection.data --output data/raw/creditcard_sample.csv --rows 5000
```

Treine os modelos:

```bash
python -m fraud_detection.train --data data/raw/creditcard_sample.csv
```

Faça predições em lote:

```bash
python -m fraud_detection.predict --input reports/predictions_sample.csv --output reports/scored_transactions.csv
```

Rode os testes:

```bash
pytest
```

## Modelos treinados

O script compara dois modelos:

- **Regressão Logística balanceada**: baseline simples e interpretável.
- **Random Forest balanceada**: referência robusta para dados tabulares e não linearidades.

O melhor modelo é escolhido por `average_precision`, equivalente à área sob a curva precisão-recall. Depois, o limiar final é escolhido com base em custo financeiro estimado:

```text
custo = falsos_positivos * custo_revisao + falsos_negativos * perda_media_fraude
```

Valores padrão:

- Custo de revisão manual: `R$ 8,00`
- Perda média por fraude não capturada: `R$ 450,00`

Esses valores podem ser alterados na linha de comando:

```bash
python -m fraud_detection.train --review-cost 6 --fraud-loss 600
```

## Saídas geradas

Depois do treino, o projeto grava:

- `models/fraud_model.joblib`: pipeline treinado, limiar e metadados.
- `reports/metrics.json`: métricas dos modelos e resultado final.
- `reports/business_summary.md`: resumo executivo para apresentação.
- `reports/confusion_matrix.csv`: matriz de confusão no teste.
- `reports/feature_importance.csv`: importância de variáveis ou coeficientes.
- `reports/predictions_sample.csv`: amostra pontuada para testar o script de predição.

## Como apresentar na DIO

Sugestão de roteiro:

1. Contextualizar o problema de fraude em cartões.
2. Explicar por que acurácia não é suficiente em bases desbalanceadas.
3. Mostrar a distribuição da variável `Class`.
4. Comparar Regressão Logística e Random Forest por `PR-AUC`, `recall`, `precision` e custo.
5. Defender o limiar escolhido por impacto financeiro.
6. Concluir com próximos passos: monitoramento de drift, explicabilidade e integração com uma fila de revisão.

## Referências

- Kaggle/ULB, Credit Card Fraud Detection: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Lucas, Y. e Jurgovsky, J. Credit card fraud detection using machine learning: a survey: https://arxiv.org/abs/2010.06479
- Kabane, S. Impact of Sampling Techniques and Data Leakage on XGBoost Performance in Credit Card Fraud Detection: https://arxiv.org/abs/2412.07437
- Showalter, S. e Wu, Z. Minimizing the Societal Cost of Credit Card Fraud with Limited and Imbalanced Data: https://arxiv.org/abs/1909.01486

