# %% [markdown]
# # Detecção de Fraudes em Cartões de Crédito
#
# Roteiro para exploração em Jupyter/VS Code. Execute as células após instalar
# as dependências do projeto com `pip install -r requirements.txt`.

# %%
from pathlib import Path

import pandas as pd

from fraud_detection.data import save_synthetic_dataset
from fraud_detection.train import BusinessCosts, train_and_evaluate

# %% [markdown]
# ## 1. Carregar ou gerar dados

# %%
data_path = Path("../data/raw/creditcard_sample.csv")
if not data_path.exists():
    save_synthetic_dataset(data_path, rows=5000, fraud_rate=0.012, random_state=42)

df = pd.read_csv(data_path)
df.head()

# %% [markdown]
# ## 2. Entender o desbalanceamento

# %%
df["Class"].value_counts(normalize=True).rename("proporcao")

# %% [markdown]
# ## 3. Treinar modelos e avaliar impacto financeiro

# %%
metrics = train_and_evaluate(
    data_path=data_path,
    model_path=Path("../models/fraud_model.joblib"),
    report_dir=Path("../reports"),
    random_state=42,
    costs=BusinessCosts(review_cost=8.0, fraud_loss=450.0),
)
metrics["test"]

# %% [markdown]
# ## 4. Ver resumo executivo

# %%
print(Path("../reports/business_summary.md").read_text(encoding="utf-8"))

