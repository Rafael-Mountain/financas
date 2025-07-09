# %% [markdown]
# <a href="https://colab.research.google.com/github/ricospeloacaso/python_para_investimentos/blob/master/06_Simulando_Carteiras_de_A%C3%A7%C3%B5es_Aleat%C3%B3rias_Python_para_Investimentos_com_Google_Colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# #Ricos pelo Acaso

# %% [markdown]
# Link para o vídeo: https://youtu.be/G2Tr2dcjR3U

# %% [markdown]
# # 1. Importando Bibliotecas
# 

# %%
# Configurando Yahoo Finance
#!pip install yfinance --upgrade --no-cache-dir
import yfinance as yf

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import random

# %% [markdown]
# # 2. Obtendo e tratando os dados 

# %%
tickers_ibov = "ABEV3.SA AZUL4.SA B3SA3.SA BBAS3.SA BBDC3.SA BBDC4.SA BBSE3.SA BPAC11.SA BRAP4.SA BRDT3.SA BRFS3.SA BRKM5.SA BTOW3.SA CCRO3.SA CIEL3.SA CMIG4.SA COGN3.SA CRFB3.SA CSAN3.SA CSNA3.SA CVCB3.SA CYRE3.SA ECOR3.SA EGIE3.SA ELET3.SA ELET6.SA EMBR3.SA EQTL3.SA FLRY3.SA GGBR4.SA GOAU4.SA GOLL4.SA HAPV3.SA HYPE3.SA IRBR3.SA ITSA4.SA ITUB4.SA JBSS3.SA KLBN11.SA LAME4.SA LREN3.SA MGLU3.SA MRFG3.SA MRVE3.SA MULT3.SA NTCO3.SA PCAR4.SA PETR3.SA PETR4.SA QUAL3.SA RADL3.SA RAIL3.SA RENT3.SA SANB11.SA SBSP3.SA SMLS3.SA SUZB3.SA TAEE11.SA TOTS3.SA UGPA3.SA USIM5.SA VALE3.SA WEGE3.SA YDUQ3.SA"

dados_yahoo = yf.download(tickers=tickers_ibov, period='1y')["Close"]

ibov = yf.download('BOVA11.SA', period='1y')["Close"]
ibov = ibov / ibov.iloc[0]

# %%
dados_yahoo.dropna(how='all', inplace=True)
dados_yahoo.dropna(axis=1, inplace=True, thresh=246)

# %%
dados_yahoo

# %%
retorno = dados_yahoo.pct_change()
retorno

# %%
retorno_acumulado = (1 + retorno).cumprod()
retorno_acumulado.iloc[0] = 1
retorno_acumulado

# %% [markdown]
# # 3. Resultados

# %%
carteira = random.sample(list(dados_yahoo.columns) , k=5)
carteira = 10000 * retorno_acumulado.loc[: , carteira]
carteira['saldo'] = carteira.sum(axis=1)
carteira["retorno"] = carteira['saldo'].pct_change()
carteira

# %%
# 500 carteiras de 5 ações cada escolhidas aleatoriamente baseadas nos dados 
for i in range(500):
  carteira = random.sample(list(dados_yahoo.columns) , k=5)
  carteira = 10000 * retorno_acumulado.loc[: , carteira]
  carteira['saldo'] = carteira.sum(axis=1)
  carteira['saldo'].plot(figsize=(18,8))

(ibov*50000).plot(linewidth=4, color='black')
  

# %%


# %%



