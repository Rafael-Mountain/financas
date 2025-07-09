# %% [markdown]
# <a href="https://colab.research.google.com/github/ricospeloacaso/python_para_investimentos/blob/master/09_Alpha_Vantage_Python_para_Investimentos.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# # Ricos pelo Acaso
# 
# *Link* para o vídeo: https://youtu.be/kB4jCoVyLRI

# %% [markdown]
# # Bibliotecas

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

import matplotlib
matplotlib.rcParams['figure.figsize'] = (16, 8)

# %%
pip install alpha_vantage

# %%
from alpha_vantage.timeseries import TimeSeries

# %% [markdown]
# # Chave de Acesso a API da Alpha Vantage

# %% [markdown]
# https://www.alphavantage.co/

# %%

ALPHAVANTAGE_API_KEY ='DCVKQOWZCMTBBES6'

# %% [markdown]
# # Acessando a API

# %%
ts = TimeSeries(key=ALPHAVANTAGE_API_KEY, output_format='pandas')

# %%
ts.get_symbol_search('bov')

# %%
dados, meta_dados = ts.get_daily(symbol='INDX.SAO', outputsize='full')

# %%
dados

# %%
meta_dados

# %%
dados['4. close'].plot()

# %%


# %%


# %%


# %%



