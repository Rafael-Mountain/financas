# %% [markdown]
# <a href="https://colab.research.google.com/github/ricospeloacaso/python_para_investimentos/blob/master/04_Simulando_CARTEIRA_DE_A%C3%87%C3%95ES_e_comparando_com_o_IBOVESPA_Python_para_Investimentos_com_Google_Colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# #Ricos pelo Acaso

# %% [markdown]
# Link para o vídeo: https://youtu.be/TiNLwmLN-iE

# %% [markdown]
# # 1. Importando bibliotecas

# %%
#!pip install yfinance --upgrade --no-cache-dir
import yfinance as yf

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown]
# # 2. Obtendo e tratando os dados

# %%
tickers = "ABEV3.SA ITSA4.SA WEGE3.SA USIM5.SA VALE3.SA"

carteira = yf.download(tickers, period="5y")["Close"]

ibov = yf.download("^BVSP", period="5y")["Close"]

# %%
carteira.dropna(inplace=True)
carteira

# %%
ibov.dropna(inplace=True)
ibov

# %% [markdown]
# # 3. Resultados

# %%
sns.set()
carteira.plot(figsize=(18,8));

# %%
carteira_normalizada = (carteira / carteira.iloc[0])*10000
carteira_normalizada.dropna(inplace=True)

# %%
carteira_normalizada.plot(figsize=(18,8));

# %%
carteira_normalizada["saldo"] = carteira_normalizada.sum(axis=1)

# %%
carteira_normalizada

# %%
ibov_normalizado = (ibov / ibov.iloc[0])*50000
ibov_normalizado

# %%
fig, ax = plt.subplots(figsize=(18,8))

carteira_normalizada["saldo"].plot(ax=ax, label="Minha Carteira")
ibov_normalizado.plot(ax=ax, label="IBOV")

ax.legend()
plt.show()

# %%
carteira_normalizada.describe()

# %%


# %%


# %%


# %%


# %%


# %%



