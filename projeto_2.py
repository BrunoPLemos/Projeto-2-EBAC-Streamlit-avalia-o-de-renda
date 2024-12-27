import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

sns.set(context='talk', style='ticks')

st.set_page_config(
     page_title="Análise exploratória de previsão de renda",
     page_icon="https://conceito.de/wp-content/uploads/2012/01/banknotes-159085_1280.png",
     layout="wide",
)

st.write('# Análise exploratória da previsão de renda')

renda = pd.read_csv('./input/previsao_de_renda.csv')

# Converter a coluna de data para datetime

renda['data_ref'] = pd.to_datetime(renda['data_ref'])

# Exibir a data mínima e máxima

min_data = renda['data_ref'].min()

max_data = renda['data_ref'].max()

st.write(f'Data inicial: {min_data}')

st.write(f'Data final: {max_data}')



# Inputs do usuário para escolher o intervalo de datas

data_inicial = st.sidebar.date_input('Data inicial', value=min_data, min_value=min_data, max_value=max_data)

data_final = st.sidebar.date_input('Data final', value=max_data, min_value=min_data, max_value=max_data)



# Exibir as datas selecionadas

#st.sidebar.write('Data inicial = ', data_inicial)

#st.sidebar.write('Data final = ', data_final)



# Filtrar os dados com base no intervalo de datas selecionado

renda_filtrado = renda[(renda['data_ref'] >= pd.to_datetime(data_inicial)) &

(renda['data_ref'] <= pd.to_datetime(data_final))]


#plots
fig, ax = plt.subplots(8,1,figsize=(10,70))
renda[['posse_de_imovel','renda']].plot(kind='hist', ax=ax[0])
st.write('## Gráficos ao longo do tempo')
sns.lineplot(x='data_ref',y='renda', hue='posse_de_imovel',data=renda, ax=ax[1])
ax[1].tick_params(axis='x', rotation=45)
sns.lineplot(x='data_ref',y='renda', hue='posse_de_veiculo',data=renda, ax=ax[2])
ax[2].tick_params(axis='x', rotation=45)
sns.lineplot(x='data_ref',y='renda', hue='qtd_filhos',data=renda, ax=ax[3])
ax[3].tick_params(axis='x', rotation=45)
sns.lineplot(x='data_ref',y='renda', hue='tipo_renda',data=renda, ax=ax[4])
ax[4].tick_params(axis='x', rotation=45)
sns.lineplot(x='data_ref',y='renda', hue='educacao',data=renda, ax=ax[5])
ax[5].tick_params(axis='x', rotation=45)
sns.lineplot(x='data_ref',y='renda', hue='estado_civil',data=renda, ax=ax[6])
ax[6].tick_params(axis='x', rotation=45)
sns.lineplot(x='data_ref',y='renda', hue='tipo_residencia',data=renda, ax=ax[7])
ax[7].tick_params(axis='x', rotation=45)
sns.despine()
st.pyplot(plt)

st.write('## Gráficos bivariada')
fig, ax = plt.subplots(7,1,figsize=(10,50))
sns.barplot(x='posse_de_imovel',y='renda',data=renda, ax=ax[0])
sns.barplot(x='posse_de_veiculo',y='renda',data=renda, ax=ax[1])
sns.barplot(x='qtd_filhos',y='renda',data=renda, ax=ax[2])
sns.barplot(x='tipo_renda',y='renda',data=renda, ax=ax[3])
sns.barplot(x='educacao',y='renda',data=renda, ax=ax[4])
sns.barplot(x='estado_civil',y='renda',data=renda, ax=ax[5])
sns.barplot(x='tipo_residencia',y='renda',data=renda, ax=ax[6])
sns.despine()
st.pyplot(plt)


# Seleção de colunas para os gráficos ao longo do tempo
colunas_tempo = ['posse_de_imovel', 'posse_de_veiculo', 'qtd_filhos', 'tipo_renda', 'educacao', 'estado_civil', 'tipo_residencia']
coluna_selecionada_tempo = st.sidebar.selectbox('Selecione a coluna para análise temporal:', colunas_tempo)

# Seleção de colunas para os gráficos bivariados
colunas_bivariadas = ['posse_de_imovel', 'posse_de_veiculo', 'qtd_filhos', 'tipo_renda', 'educacao', 'estado_civil', 'tipo_residencia']
coluna_selecionada_bivariada = st.sidebar.selectbox('Selecione a coluna para análise bivariada:', colunas_bivariadas)


st.write('## Gráficos ao longo do tempo')
fig, ax = plt.subplots(2,1,figsize=(10,14)) #reduzi para 2 graficos para melhor visualização
renda[['posse_de_imovel','renda']].plot(kind='hist', ax=ax[0]) #mantive o histograma
sns.lineplot(x='data_ref', y='renda', hue=coluna_selecionada_tempo, data=renda_filtrado, ax=ax[1])
ax[1].tick_params(axis='x', rotation=45)
sns.despine()
st.pyplot(fig)

st.write('## Gráficos bivariada')
fig, ax = plt.subplots(1,1,figsize=(10,7)) #reduzi para um grafico para melhor visualização
sns.barplot(x=coluna_selecionada_bivariada, y='renda', data=renda_filtrado, ax=ax[0])
sns.despine()
st.pyplot(fig)


