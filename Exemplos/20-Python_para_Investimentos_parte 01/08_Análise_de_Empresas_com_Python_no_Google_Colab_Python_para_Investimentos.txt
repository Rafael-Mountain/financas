# %% [markdown]
# <a href="https://colab.research.google.com/github/codigoquant/python_para_investimentos/blob/master/08_An%C3%A1lise_de_Empresas_com_Python_no_Google_Colab_Python_para_Investimentos.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# #Ricos pelo Acaso

# %% [markdown]
# Link para o vídeo: https://youtu.be/e_ZRDG4F4ZA

# %% [markdown]
# # Importando Bibliotecas

# %%
import numpy as np
import pandas as pd
import string
import warnings
warnings.filterwarnings('ignore')

import requests

# %% [markdown]
# # Obtendo e tratando os dados - Web scraping

# %% [markdown]
# https://www.fundamentus.com.br/resultado.php

# %%
url = 'http://www.fundamentus.com.br/resultado.php'

# %%
header = {
  "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.89 Safari/537.36"  
 }

r = requests.get(url, headers=header)

# %%
# pegando a primeira tabela, ajustando decimais, virgula por ponto
df = pd.read_html(r.text,  decimal=',', thousands='.')[0]

# %%
df

# %%
# substituindo virgula por ponto
for coluna in ['Div.Yield', 'Mrg Ebit', 'Mrg. Líq.', 'ROIC', 'ROE', 'Cresc. Rec.5a']:
  df[coluna] = df[coluna].str.replace('.', '')
  df[coluna] = df[coluna].str.replace(',', '.')
  df[coluna] = df[coluna].str.rstrip('%').astype('float') / 100

# %% [markdown]
# # Analisando os dados - Fórmula Mágica

# %%
# filtrando empresas com liquidez de negociação diária, dos últimos dois meses
df = df[df['Liq.2meses'] > 1000000]

# %%
# melhores empresas, segundo a fórmula mágica.
# https://quantbrasil.com.br/magic-formula/
ranking = pd.DataFrame()
ranking['pos'] = range(1,151)
# valor da firma / geracao de xaixa operacional. Ver empresa que vale pouco em relacao à geração de caixa
ranking['EV/EBIT'] = df[ df['EV/EBIT'] > 0 ].sort_values(by=['EV/EBIT'])['Papel'][:150].values
# retorno sobre o capital investido
ranking['ROIC'] = df.sort_values(by=['ROIC'], ascending=False)['Papel'][:150].values

# %%
# somar os indices, os menores valores são as melhores
ranking

# %%
a = ranking.pivot_table(columns='EV/EBIT', values='pos')

# %%
b = ranking.pivot_table(columns='ROIC', values='pos')

# %%
t = pd.concat([a,b])
t

# %%
# filtrando as que não estão nos dois rankings
rank = t.dropna(axis=1).sum()
rank

# %%
# exclua seguradoras. Comparar com o site:
# https://quantbrasil.com.br/magic-formula/
rank.sort_values()[:15]

# %% [markdown]
# https://insight.economatica.com/existe-formula-magica-para-vencer-o-mercado

# %% [markdown]
# https://quantbrasil.com.br/magic-formula/

# %%


# %%


# %%



