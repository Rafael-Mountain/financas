# %%
import numpy as np
import pandas as pd
import datetime
#import scipy
#import math
from matplotlib.pyplot import figure

# %%
# apagar deposit,withdraw. Carregar no xlsx ou csv, apagar as ordens abertas. Se começar em ordem decrescente, inverter
df = pd.read_csv('Cyborg-Darwinex_deposit_with.csv',index_col='Open Date',parse_dates=True)
# - O operador de negação (~) exclui apenas essas linhas do DataFrame
cyb = df[~df["Action"].str.contains("Deposit|Withdrawal", case=False, na=False)]

# %%
cyb.columns

# %%
cyb.head()
#cyb.tail()

# %%
#  invertendo a ordem dos elementos.
cyb = cyb[::-1]

# %%
cyb.head()

# %%
cyb.index

# %%
print('Numero de retornos')
print(cyb['Gain'].count())

# %%
print('Soma dos Gains (em decimais)')
print("%.2f" %cyb['Gain'].sum())

# %%
LucroTotal = cyb['Profit'].sum()
print('Soma dos lucros')
print("%.2f" %LucroTotal)

# %%
#fig=plt.figure(figsize=(8,6))
#his=fig.add_axes([0,0,1,1])
Gain = cyb['Gain']
Gain.hist(bins = 30, color = 'blue', figsize=(7,3))

# %%
Gain.plot.kde()

# %%
# Gráfico dos gains
Gain.plot(label='Gains',figsize=(12,3),title='Daily Gains')

# %%
Gain.plot.line(x='Open Date',y='Gain',figsize=(12,3),lw=1)

# %%
# Isolate the adjusted closing prices 
adj_profits_px = cyb['Profit']

# Calculate the moving average
Esp = adj_profits_px.rolling(window=30).mean()

# Inspect the result
# print(Esp[-10:])
# print(Esp)
Esp.plot.line(label='Curva',figsize=(12,3),title='Expectancy')

# %%
CurvaDeCapital = cyb['Profit Cumulative'] = cyb['Profit'].cumsum()
CurvaDeCapital.plot.line(x='Open Date',y='Profit',figsize=(12,3),lw=1,title='Curva de Capital')

# %%
Volume = cyb['Units/Lots']
Volume.plot.line(x='Open Date',y='Units/Lots',figsize=(12,3),lw=1 ,title='Volume Negociado')

# %%
DepositoInicial = 15942.61

# %%
print(round(LucroTotal,2))

# %%
Retorno = (LucroTotal / DepositoInicial) * 100
print("O retorno total durante o período foi de %.2f %%" %Retorno)

# %%
data1 = datetime.date(day=3, month=7, year=2017)
data2 = datetime.date(day=21, month=12, year=2023)

diferenca = data2 - data1
n_meses = diferenca.days // 30
print(f"O período em meses foi de {n_meses} meses")

# %%
# Cálculo da taxa mensal
# https://conteudos.xpi.com.br/aprenda-a-investir/relatorios/juros-compostos/
# https://pt.wikipedia.org/wiki/Juro
i = (np.power((1 + LucroTotal/DepositoInicial), (1 / n_meses))) - 1
print("O retorno foi de %.2f%% ao mês" % round(i*100,2))

# %%


# %%


# %%


# %%



