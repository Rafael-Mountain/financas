# %% [markdown]
# <a href="https://colab.research.google.com/github/ricospeloacaso/python_para_investimentos/blob/master/03_CORRELA%C3%87%C3%83O_entre_D%C3%93LAR_e_IBOVESPA_Python_para_Investimentos.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# ### Ricos pelo Acaso
# 
# 

# %% [markdown]
# Link para o vídeo: https://youtu.be/zjaGIcUb6Ek

# %% [markdown]
# # 1. Importando bibliotecas

# %%
#!pip install yfinance --upgrade --no-cache-dir
import yfinance as yf

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Seaborn é uma biblioteca usada principalmente para plotagem estatística em  Python. 
# Construída em cima do Matplotlib
#!pip install --upgrade seaborn
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# %% [markdown]
# # 2. Obtendo e tratando os dados

# %%
tickers = "^BVSP USDBRL=X"
carteira = yf.download(tickers, start="2007-01-01")["Close"]

# %%
carteira

# %%
# A função pandas.DataFrame.dropna() remove os valores nulos (valores ausentes, Nan, Nat) do DataFrame, 
#retirando as linhas ou colunas que contêm os valores nulos.
carteira = carteira.dropna()
carteira

# %%
carteira.columns = ["DOLAR", "IBOV"]
carteira

# %% [markdown]
# # 3. Resultados

# %%
sns.set()
carteira.plot(subplots=True, figsize=(22,5));

# %%
retornos = carteira.pct_change()[1:]
retornos

# %%
retornos.describe()

# %%
sns.heatmap(retornos.corr(), annot=True);

# %%
retornos["DOLAR"].rolling(252).corr(retornos["IBOV"]).plot(figsize=(22,8))

# %%
carteira.loc[:, "IBOV_DOLARIZADO"] = carteira["IBOV"] / carteira["DOLAR"]
carteira

# %%
# Substituir valores infinitos por NaN
retornos.replace([np.inf, -np.inf], np.nan, inplace=True)
sns.pairplot(retornos);

# %%
sns.set()
carteira.plot(subplots=True, figsize=(22,8));

# %%


# %%


# %%


# %%


# %%


# %%



