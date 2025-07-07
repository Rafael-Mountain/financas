# %% [markdown]
# # Portfolio Optimization

# %% [markdown]
# “A Teoria Moderna de Portfólio (MPT), uma hipótese apresentada por Harry Markowitz em seu artigo  “Portfolio Selection,” (publicado em 1952 pelo Journal of Finance) é uma teoria de investimento baseada na ideia de que investidores avessos ao risco podem construir carteiras para otimizar ou maximizar o retorno esperado com base em um determinado nível de risco de mercado, enfatizando que o risco é uma parte inerente de uma recompensa mais elevada. É uma das teorias econômicas mais importantes e influentes que tratam de finanças e investimentos.

# %% [markdown]
# ## Monte Carlo Simulation for Optimization Search
# 
# 
# We could randomly try to find the optimal portfolio balance using Monte Carlo simulation

# %%
# https://medium.com/turing-talks/an%C3%A1lise-de-um-portf%C3%B3lio-de-a%C3%A7%C3%B5es-em-python-1a5e0b3455fc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
#import yfinance as yf

# %%
# Download and get Daily Returns
# nestes arquivos estao somente as datas e o close. Date,Adj. Close. AAPL_CLOSE é o nome do arquivo.
aapl = pd.read_csv('AAPL_CLOSE',index_col='Date',parse_dates=True)
cisco = pd.read_csv('CISCO_CLOSE',index_col='Date',parse_dates=True)
ibm = pd.read_csv('IBM_CLOSE',index_col='Date',parse_dates=True)
amzn = pd.read_csv('AMZN_CLOSE',index_col='Date',parse_dates=True)
# 2012-01-03 a 2016-12-30

# %%
#start = pd.to_datetime('2012-01-01')
#end = pd.to_datetime('2017-01-01')  # não tem negociação nesse dia

# %%
#aapl =  yf.download("AAPL", start, end)
#msft =  yf.download("MSFT", start, end)
#ibm =  yf.download("IBM", start, end)
#amzn =  yf.download("AMZN", start, end)

# %%
print(aapl.count())
print(cisco.count())
print(ibm.count())
print(amzn.count())

# %%
# concatenação por colunas
stocks = pd.concat([aapl,cisco,ibm,amzn],axis=1)
stocks.columns = ['aapl','cisco','ibm','amzn']

# %%
stocks.tail()

# %%
stocks.pct_change(1)

# %%
# sem alocação, somente os preços. Média dos retornos para todo o período, para cada ação.
mean_daily_ret = stocks.pct_change(1).mean()
mean_daily_ret

# %%
round(stocks.pct_change(1).corr(),2)

# %% [markdown]
# # Simulating Thousands of Possible Allocations

# %%
stocks.head()

# %%
stock_normed = stocks/stocks.iloc[0]
stock_normed.plot()

# %%
stock_normed.head()

# %%
# calcula para todas as acoes, individualmente, sem alocação.
stock_daily_ret = stocks.pct_change(1)
stock_daily_ret.head()

# %% [markdown]
# ## Log Returns vs Arithmetic Returns
# 
# Agora passaremos a usar retornos de log em vez de retornos aritméticos, para muitos de nossos casos de uso eles são quase os mesmos, mas a maioria das análises técnicas exigem redução de tendência/normalização da série temporal e usar retornos de log é uma ótima maneira de fazer isso.
# Os retornos de log são convenientes para trabalhar em muitos dos algoritmos que encontraremos.
# 
# Para uma análise completa de por que usamos retornos de log, consulte [este excelente artigo](https://quantivity.wordpress.com/2011/02/21/why-log-returns/).
# 

# %%
# log da razao dos precos das acoes. Ao inves de usar o change usa o shift e tira o log. Quase igual
log_ret = np.log(stocks/stocks.shift(1))
log_ret.head()

# %%
log_ret.hist(bins=100,figsize=(12,6));
plt.tight_layout()

# %%
log_ret.describe().transpose()

# %%
# sem o log, somente o change
stock_daily_ret.describe().transpose()

# %%
# retorno anual sem log
stock_daily_ret.mean() * 252

# %%
# retorno anual
log_ret.mean() * 252

# %%
# Compute pairwise covariance of columns. Mostra apenas a direção da relação.
log_ret.cov()

# %% [markdown]
# 

# %%
round(log_ret.cov()*252,4) # multiply by days

# %%
# padroniza a covariância e mede a intensidade da relação.[-1,+1]
round(log_ret.corr(),4)

# %%
# ATÉ AQUI SEM ALOCAÇÃO, SOMENTE VARIAÇOES DE PREÇOS

# %% [markdown]
# ## Single Run for Some Random Allocation

# %%
# Set seed (optional). garante que gere a maior quantidade de números aleatórios
# NumPy.Random.Seed (101) Explained - Medium. 101 valor arbitrário
np.random.seed(101)

# Stock Columns
print('Stocks')
print(stocks.columns)
print('\n')

# %%
# Create Random Weights
print('Creating Random Weights')
weights = np.around(np.array(np.random.random(4)),2)
print(weights)
print('\n')

# %%
# Rebalance Weights. Normalização
print('Rebalance to sum to 1.0')
weights = np.around(weights / np.sum(weights),2)
print(weights)
print('\n')

# %%
# Expected Return
print('Expected Portfolio Return')
# fez o produto dos pesos pela média do retorno de cada acao, soma e multiplica pelo n. de dias em uma ano
exp_ret = np.around(np.sum(log_ret.mean() * weights) *252,2)
print(exp_ret)
print('\n')

# %% [markdown]
# $Volatilidade (anualizada) = \sqrt{ w^T \cdot log(Cov) \cdot 252\cdot w }$ 

# %%
# Expected Variance
print('Expected Volatility')
# raiz quadrada do produto dos pesos pelo log da covariancia
exp_vol = np.around(np.sqrt(np.dot(weights.T, np.dot(log_ret.cov() * 252, weights))),2)
print(exp_vol)
print('\n')

# %%
# Sharpe Ratio
SR = np.around(exp_ret/exp_vol,2)
print('Sharpe Ratio')
print(SR)

# %% [markdown]
# Agora fazendo essa operação muitas vezes:

# %%
num_ports = 15000

all_weights = np.zeros((num_ports,len(stocks.columns)))
ret_arr = np.zeros(num_ports)
vol_arr = np.zeros(num_ports)
sharpe_arr = np.zeros(num_ports)

for ind in range(num_ports):

    # Create Random Weights
    weights = np.array(np.random.random(len(stocks.columns)))

    # Rebalance Weights
    weights = weights / np.sum(weights)
    
    # Save Weights
    all_weights[ind,:] = weights

    # Expected Return. Agora são vetores.
    ret_arr[ind] = np.sum((log_ret.mean() * weights) *252)

    # Expected Variance. Vetor também.
    vol_arr[ind] = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov() * 252, weights)))

    # Sharpe Ratio
    sharpe_arr[ind] = ret_arr[ind]/vol_arr[ind]

# %%
max_sr_MC = round(sharpe_arr.max(),3)
max_sr_MC

# %%
sharpe_arr.argmax()

# %%
np.around(all_weights[sharpe_arr.argmax(),:],2)

# %%
max_sr_ret_MC = ret_arr[sharpe_arr.argmax()]
print('Retorno do max SR :')
print(round(max_sr_ret_MC,2))



# %%
max_sr_vol_MC = vol_arr[sharpe_arr.argmax()]
print('Volatilide do max SR:')
print(round(max_sr_vol_MC,2))

# %% [markdown]
# ## Plotting the data

# %%
plt.figure(figsize=(12,8))
plt.scatter(vol_arr,ret_arr,c=sharpe_arr,cmap='plasma')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')

# Add red dot for max SR
plt.scatter(max_sr_vol_MC,max_sr_ret_MC,c='red',s=50,edgecolors='black')

# %% [markdown]
# # Otimização Matemática do Sharpe Ratio
# Existem maneiras muito melhores de encontrar bons pesos de alocação do que apenas adivinhar e verificar. Por isso, serão utilizadas funções de otimização para encontrar matematicamente os pesos ideais. O objetivo é maximizar o Sharpe Ratio.

# %% [markdown]
# ### Functionalize Return and SR operations

# %%
# essa é a função que contém o SR para ser maximizado.
def get_ret_vol_sr(weights):
    """
    Takes in weights, returns array or return,volatility, sharpe ratio. 
    Retorna um vetor de 3 posicoes, 0,1,2
    0 - return
    1 - volatility
    2 - sr
    """
    weights = np.array(weights)
    ret = np.sum(log_ret.mean() * weights) * 252
    vol = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov() * 252, weights)))
    sr = ret/vol
    return np.array([ret,vol,sr])

# %%
from scipy.optimize import minimize

# %% [markdown]
# To fully understand all the parameters, check out:
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

# %%
help(minimize)
# opt_results = minimize(neg_sharpe,init_guess,method='SLSQP',bounds=bounds,constraints=cons)
# By convention of minimize function it should be a function that returns zero for conditions
# cons = ({'type':'eq','fun': check_sum})
# Constraint type: 'eq' for equality, 'ineq' for inequality.
# fun : callable, the function defining the constraint.

# %% [markdown]
# A otimização será feita com uma função de minimização, apesar de querermos maximizar o índice de Sharpe. Para isso precisaremos torná-lo negativo para que possamos minimizar o sharpe negativo (o mesmo que maximizar o sharpe positivo)

# %%
# somente troca o sinal da função do SR. Função a ser minimizada.
# retornando o SR, que está na posicao 2 do vetor de retorno. Por isso precisa trocar o sinal. 
#Queremos o maximo mas ele procura o mínimo, entao temos de inverter.
def neg_sharpe(weights):
    return  get_ret_vol_sr(weights)[2] * -1

# %%
# Constraints (restrições, condições de contorno)
# Equality constraint means that the constraint function result is to be zero whereas inequality 
# means that it is to be non-negative. Restrição: Só serão aceitos pesos cuja soma é 1.
def check_sum(weights):
    '''
    Returns 0 if sum of weights is 1.0
    '''
    return np.sum(weights) - 1

# %%
# By convention of minimize function it should be a function that returns zero for conditions
# check_sum deve retornar zero quando encontrar pesos que satisfaçam a condição de normalização
# olhar o help
# eq quer dizer que o resultado a ser comparado com check_sum deve ser igual
cons = ({'type':'eq','fun': check_sum})

# %%
# 0-1 bounds (limites) for each weight, 0 a 100% em decimal.
bounds = ((0, 1), (0, 1), (0, 1), (0, 1))

# %%
# Pesos iniciais (equal distribution)
#init_guess = [0.25,0.25,0.25,0.25]
init_guess = np.zeros(len(stocks.columns))
init_guess = init_guess + 1/len(stocks.columns)

# %%
# https://pt.wikipedia.org/wiki/Método_dos_mínimos_quadrados
# Sequential Least Squares Programming (SLSQP).
opt_results = minimize(neg_sharpe,init_guess,method='SLSQP',bounds=bounds,constraints=cons)

# %%
opt_results

# %%
np.around(opt_results.x,2)
# A minimização traz somente apenas um vetor de pesos, uma resposta
# resultado das 15.000 iterações acima, no cálculo anteriror:
# pesos por Monte Carlo ([0.26, 0.21, 0.  , 0.53]) 
# Quase igual da minimização aqui, convergência.

# %%
#  return np.array([ret,vol,sr])
# Entrando com os pesos ótimos e obtendo retorno, volatilidade e SR respectivamente.
np.around(get_ret_vol_sr(opt_results.x),2)

# %%
# Retorno do máx SR
max_sr_ret = get_ret_vol_sr(opt_results.x)[0]
print('Retorno do max SR pela minimização :')
print(round(max_sr_ret,2))


# %%
# Volatilidade do máx SR
max_sr_vol = get_ret_vol_sr(opt_results.x)[1]
print('Volatilide do max SR pela minimização :')
print(round(max_sr_vol,2))

# %%
# máx SR
max_sr = get_ret_vol_sr(opt_results.x)[2]
print('Max SR pela minimização :')
print(round(max_sr,2))
# Pelo MC: 1.03

# %% [markdown]
# # Todos os portfólios ideais (fronteira eficiente)
# 
# A fronteira eficiente é o conjunto de carteiras ótimas que oferece o maior retorno esperado para um nível de risco definido ou o menor risco para um determinado nível de retorno esperado. As carteiras que se situam abaixo (vertical da melhor volatilidade) da fronteira eficiente são subótimas, porque não proporcionam retorno suficiente para o nível de risco. As carteiras que se agrupam à direita da fronteira eficiente (horizontal do melhor retorno) também são subótimas, porque apresentam um nível de risco mais elevado para a taxa de retorno definida.
# 
# Fronteira Eficiente http://www.investopedia.com/terms/e/efficientfrontier

# %%
# Esse calculo é somente para encontrar os pontos da fronteira. 
# Os internos vem do Monte Carlo. Para cada valor aqui de retorno, (frontier_y), será 
# minimizada a volatilidade (frontier_volatility), eixo x.
# Crie um número linspace de pontos (retorno) para calcular x (volatilidade)
frontier_y = np.linspace(0,0.3,100) # Altere 100 para um número menor para computadores mais lentos

# %%
# aqui nao precisa trocar o sinal, pois queremos o mínimo de volatilidade posição 1, 
# para cada conjunto de pesos
def minimize_volatility(weights):
    return  get_ret_vol_sr(weights)[1] 

# %%
frontier_volatility = []

for possible_return in frontier_y:
    # determinação dos pesos através de duas funções de restrição:
    cons = ({'type':'eq','fun': check_sum},
            {'type':'eq','fun': lambda w: get_ret_vol_sr(w)[0] - possible_return})
            # queremos aqui os pesos w, cuja operação get_ret_vol_sr(w)[0] - possible_return seja igual a zero. Ou seja,
            # uma combinação de pesos que gerem um retorno igual ao possible_return que está no array frontier_y
            # funções anônimas lambda: veja o notebook "Curso rapido de Python"
    
    result = minimize(minimize_volatility,init_guess,method='SLSQP',bounds=bounds,constraints=cons)
    
    frontier_volatility.append(result['fun'])

# %%
plt.figure(figsize=(12,8))
# Esses arrays vieram do Monte Carlo
plt.scatter(vol_arr,ret_arr,c=sharpe_arr,cmap='plasma')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
# Add frontier line
plt.plot(frontier_volatility,frontier_y,'g--',linewidth=3)
# Add red dot for max SR, pelo Monte Carlo
plt.scatter(max_sr_vol_MC,max_sr_ret_MC,c='blue',s=80,edgecolors='black')

# %%
result

# %%
# frontier_volatility contém os valores da volatilidade da fronteira+
frontier_volatility

# %%
len(frontier_volatility)

# %%
frontier_y

# %%
# Pesos iguais
weights = np.array([0.25,0.25,0.25,0.25])
# Expected Return
print('Expected Portfolio Return')
# fez o produto dos pesos pela média do retorno de cada acao, soma e multiplica pelo n. de dias 
#em uma ano
exp_ret = np.around(np.sum(log_ret.mean() * weights) *252,2)
print(exp_ret)
print('\n')

# %%
# Expected Variance
print('Expected Volatility')
# raiz quadrada do produto dos pesos pelo log da covariancia
exp_vol = np.around(np.sqrt(np.dot(weights.T, np.dot(log_ret.cov() * 252, weights))),2)
print(exp_vol)
print('\n')

# %%
# Sharpe Ratio
SR = np.around(exp_ret/exp_vol,2)
print('Sharpe Ratio')
print(SR)

# %%
# Ótimo: array([0.22, 0.21, 1.03])


