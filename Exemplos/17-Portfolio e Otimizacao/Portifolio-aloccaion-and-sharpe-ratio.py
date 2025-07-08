# %% [markdown]
# # Sharpe Ratio and Portfolio Values

# %%
import pandas as pd

# %%
import yfinance as yf

# %% [markdown]
# ## Create a Portfolio

# %%
start = pd.to_datetime('2012-01-01')
end = pd.to_datetime('2017-01-01')  # não tem negociação nesse dia

# %%
aapl =  yf.download("AAPL", start, end)
msft =  yf.download("MSFT", start, end)
ibm =  yf.download("IBM", start, end)
amzn =  yf.download("AMZN", start, end)

# %%
# Alternative
#aapl = pd.read_csv('AAPL_CLOSE',index_col='Date',parse_dates=True)
#msft = pd.read_csv('MSFT_CLOSE',index_col='Date',parse_dates=True)
#ibm = pd.read_csv('IBM_CLOSE',index_col='Date',parse_dates=True)
#amzn = pd.read_csv('AMZN_CLOSE',index_col='Date',parse_dates=True)

# %%
#aapl.to_csv('AAPL_CLOSE')
#msft.to_csv('MSFT_CLOSE')
#ibm.to_csv('IBM_CLOSE')
#amzn.to_csv('AMZN_CLOSE')

# %%
amzn.head()

# %% [markdown]
# ## Normalize Prices
# 
# This is the same as cumulative daily returns

# %%
# Example
amzn.iloc[0]['Adj Close']

# %%
print(aapl.count())
print(msft.count())
print(ibm.count())
print(amzn.count())

# %%
# criando para cada dataframe e de modo automatico, uma nova coluna. Na hora de imprimir, imprime o último.
for stock_df in (aapl,msft,ibm,amzn):
    stock_df['Normed Return'] = stock_df['Adj Close']/stock_df.iloc[0]['Adj Close']

# %%
aapl.columns

# %%
aapl.head()

# %%
msft.head()

# %%
ibm.head()

# %%
amzn.head()


# %%
ibm.tail()

# %%
amzn.count()

# %%
# para ver que a coluna foi acrescentada
amzn.tail()

# %%
aapl.tail()

# %% [markdown]
# ## Allocations
# 
# Let's pretend we had the following allocations for our total portfolio:
# 
# * 30% in Apple
# * 20% in Microsoft
# * 40% in Amazon
# * 10% in IBM
# 
# Let's have these values be reflected by multiplying our Norme Return by out Allocations

# %%
# Esse loop percorre duas listas ao mesmo tempo usando zip(), que emparelha elementos correspondentes de ambas as listas.
for stock_df,allo in zip([aapl,msft,amzn,ibm],[.3,.2,.4,.1]):
#for stock_df,allo in zip([aapl,msft,ibm,amzn],[.2619,.2076,.0011,.5294]):
    stock_df['Allocation'] = stock_df['Normed Return']*allo

# %%
aapl.head()

# %% [markdown]
# ## Investment
# 
# Let's pretend we invested a million dollars in this portfolio

# %%
# o capital se distribui de acordo com os pesos. Se os pesos fossem todos iguais a um, seria
# como se investir esse valor vezes o número de acoes
for stock_df in [aapl,msft,ibm,amzn]:
    stock_df['Position Values'] = stock_df['Allocation']*1000000

# %% [markdown]
# ## Total Portfolio Value

# %%
portfolio_val = round(pd.concat([aapl['Position Values'],msft['Position Values'],ibm['Position Values'],amzn['Position Values']],axis=1),2)

# %%
portfolio_val.head()

# %%
portfolio_val.columns = ['AAPL Pos','MSFT Pos','IBM Pos','AMZN Pos']

# %%
portfolio_val.head()

# %%
portfolio_val['Total Pos'] = portfolio_val.sum(axis=1)

# %%
portfolio_val.head()

# %%
import matplotlib.pyplot as plt
matplotlib inline

# %%
portfolio_val['Total Pos'].plot(figsize=(10,8))
plt.title('Total Portfolio Value')

# %%
# Cálculo do DD. Fazer
# https://quant.stackexchange.com/questions/18094/how-can-i-calculate-the-maximum-drawdown-mdd-in-python
Roll_Max =       portfolio_val['Total Pos'].cummax()
Daily_Drawdown = portfolio_val['Total Pos']/Roll_Max - 1.0
Max_Daily_Drawdown = Daily_Drawdown.cummin()
# Plot the results
Daily_Drawdown.plot(figsize=(10,8))
Max_Daily_Drawdown.plot()
# Add red dot for max SR
#Max_Daily_Drawdown.plot(Max_Daily_Drawdown.idxmin(),Max_Daily_Drawdown.min(),c='red',s=50,edgecolors='black')
plt.show()


# %%
# AAPL Pos 	MSFT Pos 	IBM Pos 	AMZN Pos 	Total Pos. Cálculo do Maximo DD e respectiva data.
Roll_Max_AAPL =       portfolio_val['AAPL Pos'].cummax()
Daily_Drawdown_AAPL = portfolio_val['AAPL Pos']/Roll_Max_AAPL - 1.0
Max_Daily_Drawdown_AAPL = Daily_Drawdown_AAPL.cummin()
print('Máximo drawdown AAPL:')
print(round(100*Max_Daily_Drawdown_AAPL.min(),2))
print('Data do MDD AAPL:')
print(Max_Daily_Drawdown_AAPL.idxmin())
# AAPL Pos 	MSFT Pos 	IBM Pos 	AMZN Pos 	Total Pos
Roll_Max_MSFT =       portfolio_val['MSFT Pos'].cummax()
Daily_Drawdown_MSFT = portfolio_val['MSFT Pos']/Roll_Max_MSFT - 1.0
Max_Daily_Drawdown_MSFT = Daily_Drawdown_MSFT.cummin()
print('Máximo drawdown MSFT:')
print(round(100*Max_Daily_Drawdown_MSFT.min(),2))
print('Data do MDD MSFT:')
print(Max_Daily_Drawdown_MSFT.idxmin())
# AAPL Pos 	MSFT Pos 	IBM Pos 	AMZN Pos 	Total Pos
Roll_Max_IBM =       portfolio_val['IBM Pos'].cummax()
Daily_Drawdown_IBM = portfolio_val['IBM Pos']/Roll_Max_IBM - 1.0
Max_Daily_Drawdown_IBM = Daily_Drawdown_IBM.cummin()
print('Máximo drawdown IBM:')
print(round(100*Max_Daily_Drawdown_IBM.min(),2))
print('Data do MDD IBM:')
print(Max_Daily_Drawdown_IBM.idxmin())
# AAPL Pos 	CISCO Pos 	IBM Pos 	AMZN Pos 	Total Pos
Roll_Max_AMZN =       portfolio_val['AMZN Pos'].cummax()
Daily_Drawdown_AMZN = portfolio_val['AMZN Pos']/Roll_Max_AMZN - 1.0
Max_Daily_Drawdown_AMZN = Daily_Drawdown_AMZN.cummin()
print('Máximo drawdown AMZN:')
print(round(100*Max_Daily_Drawdown_AMZN.min(),2))
print('Data do MDD AMZN:')
print(Max_Daily_Drawdown_AMZN.idxmin())
print('')
print('Máximo drawdown Port:')
print(round(100*Max_Daily_Drawdown.min(),2))
print('Data do MDD Port:')
print(Max_Daily_Drawdown.idxmin())


# %%
# separando a coluna Total Pos e fazendo o gráfico
portfolio_val.drop('Total Pos',axis=1).plot(kind='line', figsize=(12,8))

# %%
portfolio_val.head()

# %%
portfolio_val.tail()

# %% [markdown]
# # Portfolio Statistics
# ### Daily Returns

# %%
portfolio_val['Daily Return'] = portfolio_val['Total Pos'].pct_change(1)
portfolio_val['Daily Return'].head()

# %% [markdown]
# ### Cumulative Return

# %%
# última posição
portfolio_val.iloc[-1]['Total Pos']

# %%
# primeira posição
portfolio_val.iloc[0]['Total Pos']

# %% [markdown]
# * lembrando que retorno = (Pt/Pt-1) -1 então o retorno total será = (Pfinal / Pinicial) - 1

# %%
# aqui, além de cumulativo, refere-se a todo o período.
cum_ret = round(100 * (portfolio_val.iloc[-1]['Total Pos']/portfolio_val.iloc[0]['Total Pos'] -1 ),2)
print('Our return  was {} percent!'.format(cum_ret))

# %% [markdown]
# ### Avg Daily Return

# %%
# variação diária da soma das alocações das quatro ações. Coluna Total Pos
round(portfolio_val['Daily Return'].mean(),5)

# %% [markdown]
# ### Std Daily Return

# %%
round(portfolio_val['Daily Return'].std(),4)

# %%
portfolio_val['Daily Return'].plot(kind='kde')

# %% [markdown]
# # Sharpe Ratio
# 
# The Sharpe Ratio is a measure for calculating risk-adjusted return, and this ratio has become the industry standard for such calculations. 
# 
# Sharpe ratio = (Mean portfolio return − Risk-free rate)/Standard deviation of portfolio return
# 
# The original Sharpe Ratio
# 
# Annualized Sharpe Ratio = K-value * SR
# 
# K-values for various sampling rates:
# 
# * Daily = sqrt(252)
# * Weekly = sqrt(52)
# * Monthly = sqrt(12)
# 
# 
# Read more: Sharpe Ratio http://www.investopedia.com/terms/s/sharperatio

# %%
SR = round(portfolio_val['Daily Return'].mean()/portfolio_val['Daily Return'].std(),3)


# %%
SR

# %%
# Annualized Sharpe Ratio = K-value * SR
ASR = round((252**0.5)*SR,3)

# %%
ASR

# %%
aapl['Adj Close'].pct_change(1).plot(kind='kde')
ibm['Adj Close'].pct_change(1).plot(kind='kde')
amzn['Adj Close'].pct_change(1).plot(kind='kde')
msft['Adj Close'].pct_change(1).plot(kind='kde')

# %%


# %%


# %%



