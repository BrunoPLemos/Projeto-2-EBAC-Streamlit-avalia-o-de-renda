import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np  
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn.tree import DecisionTreeRegressor, plot_tree

sns.set(context='talk', style='ticks')

st.set_page_config(
    page_title="Análise Exploratória de Previsão de Renda",
    page_icon="https://conceito.de/wp-content/uploads/2012/01/banknotes-159085_1280.png",
    layout="wide",
)

st.title('Análise Exploratória da Previsão de Renda')

try:
    renda = pd.read_csv('./input/previsao_de_renda.csv')
    renda['tempo_emprego'].fillna(renda['tempo_emprego'].mean(), inplace=True)
    colunas_para_remover = ["Unnamed: 0", "id_cliente"]
    renda.drop(colunas_para_remover, axis=1, inplace=True)
    renda['data_ref'] = pd.to_datetime(renda['data_ref'])

    min_data = renda['data_ref'].min()
    max_data = renda['data_ref'].max()

    with st.sidebar.expander("Filtros de Data"):
        data_inicial = st.date_input('Data inicial', value=min_data, min_value=min_data, max_value=max_data)
        data_final = st.date_input('Data final', value=max_data, min_value=min_data, max_value=max_data)

    renda_filtrado = renda[(renda['data_ref'] >= pd.to_datetime(data_inicial)) &
                         (renda['data_ref'] <= pd.to_datetime(data_final))]

    with st.sidebar.expander("Seleção de Variáveis"):
        colunas_tempo = ['sexo','posse_de_imovel', 'posse_de_veiculo','qtd_filhos', 'tipo_renda', 'educacao', 'estado_civil', 'tipo_residencia','idade','tempo_emprego', 'qt_pessoas_residencia']
        colunas_bivariadas = colunas_tempo
        colunas_numericas = ['renda', 'idade', 'qtd_filhos', 'tempo_emprego','qt_pessoas_residencia'] 

        colunas_selecionadas_tempo = st.multiselect('Selecione as colunas para análise temporal:', colunas_tempo, default=[colunas_tempo[0]])
        coluna_selecionada_bivariada = st.selectbox('Selecione a coluna para análise bivariada:', colunas_bivariadas)

    # --- Abas ---
    aba_temporal, aba_bivariada, aba_estatisticas, aba_modelagem = st.tabs(["Análise Temporal", "Análise Bivariada", "Estatísticas", "Modelagem"])

    with aba_temporal:
        st.subheader('Gráficos ao longo do tempo')
        if colunas_selecionadas_tempo:
            num_graficos = len(colunas_selecionadas_tempo)
            fig, axes = plt.subplots(num_graficos, 1, figsize=(12, 6 * num_graficos), squeeze=False)

            for i, coluna in enumerate(colunas_selecionadas_tempo):
                ax = axes[i, 0]
                sns.lineplot(x='data_ref', y='renda', hue=coluna, data=renda_filtrado, ax=ax, palette="viridis")
                ax.set_ylabel('Renda (R$)')
                ax.set_xlabel('Data de Referência')
                ax.set_title(f'Variação da Renda por {coluna} ao Longo do Tempo', fontsize=14)
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
                sns.despine(ax=ax) 

            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("Selecione pelo menos uma coluna para o gráfico temporal.")

    with aba_bivariada:
        st.subheader('Gráfico Bivariado')
        if coluna_selecionada_bivariada:
            fig_bivariado, ax_bivariado = plt.subplots(figsize=(8, 5))
            sns.barplot(x=coluna_selecionada_bivariada, y='renda', data=renda_filtrado, ax=ax_bivariado, palette="viridis")
            ax_bivariado.set_ylabel('Renda (R$)')
            ax_bivariado.set_xlabel(coluna_selecionada_bivariada)
            plt.setp(ax_bivariado.get_xticklabels(), rotation=45, ha="right")
            sns.despine(ax=ax_bivariado)
            plt.tight_layout()
            st.pyplot(fig_bivariado)
        else:
            st.warning("Selecione uma coluna para o gráfico bivariado.")

    with aba_estatisticas:
        st.subheader("Estatísticas Descritivas")
        st.write("Resumo estatístico das variáveis numéricas:")
        st.dataframe(renda_filtrado[colunas_numericas].describe().T.style.format("{:.2f}"))

        st.subheader("Correlação entre as variáveis numéricas")
        fig_correlacao, ax_correlacao = plt.subplots()
        sns.heatmap(renda_filtrado[colunas_numericas].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax_correlacao)
        st.pyplot(fig_correlacao)

    with aba_modelagem:
        st.subheader("Modelagem")

        # --- Seleção das variáveis para o modelo ---
        st.write("Selecione as variáveis para o modelo:")
        colunas_modelo = st.multiselect("Variáveis preditoras (X):", renda_filtrado.columns.drop('renda'), 
                                        default=renda_filtrado.columns.drop('renda').tolist()[:3]) 
        variavel_alvo = st.selectbox("Variável alvo (y):", ['renda'])

        if colunas_modelo and variavel_alvo:  # Verifica se as variáveis foram selecionadas
            X = renda_filtrado[colunas_modelo]
            y = renda_filtrado[variavel_alvo]

            # --- Aplicar One-Hot Encoding ---
            X = pd.get_dummies(X, drop_first=True)

            # Converte a coluna 'data_ref' para numérica 
            if 'data_ref' in X.columns:
                data_min = renda_filtrado['data_ref'].min()
                X['data_ref'] = (renda_filtrado['data_ref'] - data_min).dt.days

            # --- Divisão em treino e teste ---
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # --- Modelo de Regressão Linear ---
            st.subheader("Regressão Linear")
            modelo_lr = LinearRegression()
            modelo_lr.fit(X_train, y_train)
            y_pred_lr = modelo_lr.predict(X_test)

            # Avaliação do modelo
            mse_lr = mean_squared_error(y_test, y_pred_lr)
            r2_lr = r2_score(y_test, y_pred_lr)

            st.write(f"**Regressão Linear - Métricas:**")
            st.write(f"- Erro Quadrático Médio (MSE): {mse_lr:.2f}")
            st.write(f"- R-Quadrado (R²): {r2_lr:.2f}")

            # Visualização dos Resíduos
            fig_residuos_lr, ax_residuos_lr = plt.subplots()
            sns.residplot(x=y_test, y=y_pred_lr - y_test, ax=ax_residuos_lr)
            ax_residuos_lr.set_xlabel("Valores Reais")
            ax_residuos_lr.set_ylabel("Resíduos")
            st.pyplot(fig_residuos_lr)

            # --- Modelo de Árvore de Decisão ---
            st.subheader("Árvore de Decisão")
            max_depth = st.slider("Profundidade máxima da árvore:", min_value=1, max_value=20, value=5, step=1)
            modelo_dt = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
            modelo_dt.fit(X_train, y_train)
            y_pred_dt = modelo_dt.predict(X_test)

            # Avaliação do modelo
            mse_dt = mean_squared_error(y_test, y_pred_dt)
            r2_dt = r2_score(y_test, y_pred_dt)

            st.write(f"**Árvore de Decisão - Métricas:**")
            st.write(f"- Erro Quadrático Médio (MSE): {mse_dt:.2f}")
            st.write(f"- R-Quadrado (R²): {r2_dt:.2f}")

            # Visualização da Árvore
            st.subheader("Visualização da Árvore de Decisão")
            fig_arvore, ax_arvore = plt.subplots(figsize=(12, 8))
            plot_tree(modelo_dt, feature_names=X.columns, filled=True, ax=ax_arvore, fontsize=10)
            st.pyplot(fig_arvore)

            # Comparação entre os modelos
            st.subheader("Comparação entre os Modelos")
            comparacao = pd.DataFrame({
                "Modelo": ["Regressão Linear", "Árvore de Decisão"],
                "MSE": [mse_lr, mse_dt],
                "R²": [r2_lr, r2_dt]
            })
            st.dataframe(comparacao)

        else:
            st.warning("Selecione as variáveis para o modelo.")

except FileNotFoundError:
    st.error("Arquivo não encontrado. Certifique-se de que o arquivo 'previsao_de_renda.csv' está no diretório correto.")
except Exception as e:
    st.error(f"Ocorreu um erro ao carregar os dados: {e}")