# %% [markdown]
# <a href="https://colab.research.google.com/github/ricospeloacaso/python_para_investimentos/blob/master/05_Backtesting_com_Pyfolio_Python_para_Investimentos.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# #Python para Investimentos
# 
# 

# %% [markdown]
# Link para o vídeo: https://youtu.be/d2qrsCfXung

# %%
#@title Vídeo
from IPython.display import YouTubeVideo
YouTubeVideo('d2qrsCfXung', width=854, height=480)

# %% [markdown]
# # 1. Importando bibliotecas

# %%
#!pip install yfinance --upgrade --no-cache-dir
import yfinance as yf
#import pandas_datareader.data as web
#yf.pdr_override()

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Para corrgir o bug: AttributeError: 'numpy.int64' object has no attribute 'to_pydatetime'
#!pip install git+https://github.com/quantopian/pyfolio

# %%
import pyfolio as pf
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# # 2. Obtendo e tratando os dados

# %%
#tickers = ["ABEV3.SA", "ITSA4.SA", "WEGE3.SA", "USIM5.SA", "VALE3.SA", '^BVSP']
#dados_yahoo = web.get_data_yahoo(tickers, period="5y")["Adj Close"]

tickers = "ABEV3.SA ITSA4.SA WEGE3.SA USIM5.SA VALE3.SA ^BVSP"
dados_yahoo = yf.download(tickers=tickers, period="5y")['Close']


# %%
dados_yahoo

# %%
retorno = dados_yahoo.pct_change()
retorno

# %%
retorno_acumulado = (1 + retorno).cumprod()
retorno_acumulado.iloc[0] = 1
retorno_acumulado

# %%
carteira = 10000 * retorno_acumulado.iloc[:, :5]
carteira["saldo"] = carteira.sum(axis=1)
carteira["retorno"] = carteira["saldo"].pct_change()
carteira

# %% [markdown]
# # 3. Resultados

# %% [markdown]
# O **Sortino Ratio** é uma métrica financeira usada para avaliar o retorno ajustado ao risco de um investimento, focando apenas na volatilidade negativa. Ele é uma variação do **Sharpe Ratio**, mas ao invés de considerar toda a volatilidade (positiva e negativa), o Sortino Ratio leva em conta apenas os retornos negativos, ou seja, os riscos de perda.

# %% [markdown]
# O **Calmar Ratio** é uma métrica financeira usada para avaliar o desempenho de um investimento, comparando seu retorno anual médio com sua máxima perda (**drawdown**). Ele ajuda investidores a entender **quanto retorno podem esperar pelo risco que estão assumindo**.
# 

# %% [markdown]
# O **Omega Ratio** é uma métrica de desempenho financeiro que avalia a relação entre ganhos e perdas de um investimento, considerando um limite de retorno desejado. Diferente de outras métricas como o Sharpe Ratio, ele leva em conta toda a distribuição dos retornos, incluindo assimetria e caudas.

# %% [markdown]
# **Stability**  refere-se à consistência dos retornos ao longo do tempo. Ele mede quão suave é a curva de retorno acumulado, ajudando a avaliar se uma estratégia de investimento tem um desempenho estável ou se apresenta grandes oscilações.

# %% [markdown]
# ### Beta (β) – Medida de Risco
# O Beta mede a sensibilidade de um ativo em relação ao mercado. Ele indica o quanto um ativo tende a se mover em resposta às variações do mercado.
# - **Beta = 1** → O ativo se move exatamente como o mercado.
# - **Beta > 1** → O ativo é mais volátil que o mercado (movimentos amplificados).
# - **Beta < 1** → O ativo é menos volátil que o mercado (movimentos reduzidos).
# - **Beta < 0** → O ativo se move na direção oposta ao mercado.
# 
# ### Alpha (α) – Retorno em Excesso
# O Alpha mede o desempenho ajustado ao risco de um ativo ou fundo de investimento. Ele indica se um ativo está superando ou ficando abaixo do retorno esperado, dado seu nível de risco.
# - **Alpha positivo** → O ativo está gerando retorno acima do esperado.
# - **Alpha negativo** → O ativo está performando abaixo do esperado.
# 
# Esses conceitos são amplamente utilizados na gestão de carteiras e no modelo **CAPM (Capital Asset Pricing Model)** para avaliar o retorno esperado de um investimento.

# %%
#  relatório detalhado de desempenho da sua carteira de investimentos usando a biblioteca Pyfolio
pf.create_full_tear_sheet(carteira["retorno"], benchmark_rets=retorno["^BVSP"])


# %%
fig, ax1 = plt.subplots(figsize=(16,8))
pf.plot_rolling_beta(carteira["retorno"], factor_returns=retorno["^BVSP"], ax=ax1)
plt.ylim((0.5, 1.2));

# %% [markdown]
# * https://www.suno.com.br/artigos/beta/#:~:text=O%C2%A0%C3%8Dndice%20Beta%20%C3%A9%20um%20indicador%20muito%20utilizado
# * https://www.investopedia.com/terms/b/beta.asp
# * https://pt.wikipedia.org/wiki/%C3%8Dndice_beta

# %%


# %%


# %%



