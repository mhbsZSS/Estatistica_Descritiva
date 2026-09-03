import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde, norm, t

# 1. Configuração da Página
st.set_page_config(page_title="Sistema Estatístico Completo", layout="wide")
st.title("📊 Sistema Completo de Análise Estatística & Inferência")
st.write("Calculadora estatística para dados brutos ou parâmetros sumários (Média, Desvio-Padrão e n).")

# 2. Escolha do Modo de Entrada (O Pulo do Cató da Flexibilidade)
modo_entrada = st.sidebar.radio(
    "Selecione a forma de entrada dos dados:", 
    ("Digitar Dados Brutos", "Informar Média, Desvio-Padrão e n")
)

# Variáveis globais para armazenar os parâmetros calculados ou informados
media = None
desvio_padrao = None
n = None
minimo = None
maximo = None
lista_dados = []
tem_dados_brutos = False

if modo_entrada == "Digitar Dados Brutos":
    dados_padrao = "198.56; 199.44; 201.25; 198.90; 197.75; 198.45; 199.25; 200.05; 199.65"
    entrada_usuario = st.text_area("Insira os dados (separados por ponto e vírgula):", value=dados_padrao)
    
    if entrada_usuario:
        try:
            dados_limpos = entrada_usuario.replace(',', '.')
            lista_dados = [float(x.strip()) for x in entrada_usuario.replace(',', '.').split(';') if x.strip()]
            df_temp = pd.DataFrame({'Variavel': lista_dados})
            
            n = len(df_temp)
            media = df_temp['Variavel'].mean()
            mediana = df_temp['Variavel'].median()
            minimo = df_temp['Variavel'].min()
            maximo = df_temp['Variavel'].max()
            amplitude = maximo - minimo
            q1 = df_temp['Variavel'].quantile(0.25)
            q3 = df_temp['Variavel'].quantile(0.75)
            variancia = df_temp['Variavel'].var(ddof=1)
            desvio_padrao = df_temp['Variavel'].std(ddof=1)
            cv = (desvio_padrao / media) * 100
            tem_dados_brutos = True

            # Moda via KDE
            contagem_freq = df_temp['Variavel'].value_counts()
            if contagem_freq.iloc[0] > 1:
                moda_texto = f"{contagem_freq.index[0]:.4f} (Freq: {contagem_freq.iloc[0]})"
            else:
                kde = gaussian_kde(lista_dados)
                x_vals = np.linspace(minimo, maximo, 1000)
                density = kde(x_vals)
                moda_estimada = x_vals[np.argmax(density)]
                moda_texto = f"{moda_estimada:.4f} (Estimada KDE)"

        except Exception as e:
            st.error(f"Erro na leitura dos dados: {e}")

else:  # Modo Parâmetros Sumários
    st.sidebar.subheader("Parâmetros da Amostra")
    media = st.sidebar.number_input("Média (x̄):", value=199.2556, format="%.4f")
    
    # Novo: Seletor para definir se o dado é Desvio Padrão ou Variância
    tipo_dispersao = st.sidebar.radio(
        "O valor informado abaixo é:",
        ("Desvio Padrão (s)", "Variância (s²)")
    )
    
    valor_dispersao = st.sidebar.number_input("Valor da Dispersão:", value=1.0201, format="%.4f")
    
    # Lógica de conversão
    if tipo_dispersao == "Desvio Padrão (s)":
        desvio_padrao = valor_dispersao
        variancia = desvio_padrao ** 2
    else:
        variancia = valor_dispersao
        desvio_padrao = np.sqrt(variancia) # Calcula a raiz quadrada
        
    n = st.sidebar.number_input("Tamanho da Amostra (n):", value=9, min_value=2, step=1)
    
    # Valores estimados/padrão para exibição caso não haja dados brutos
    mediana, minimo, maximo, amplitude, q1, q3 = media, media-1, media+1, 2, media-0.5, media+0.5
    cv = (desvio_padrao / media) * 100
    moda_texto = "N/A (Moda exige dados brutos)"
    tem_dados_brutos = False
    
# --- EXIBIÇÃO DOS RESULTADOS SE HOUVER DADOS VÁLIDOS ---
if n is not None and media is not None:
    st.success(f"Cálculos efetuados com sucesso! Amostra ativa: $n = {n}$")

    # 1. Resumo Estatístico (Precisão de 4 casas decimais)
    st.subheader("1. Resumo Estatístico (Precisão: 4 Casas Decimais)")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Número de Obs (n)", int(n))
    col1.metric("Mínimo", f"{minimo:.4f}")
    col1.metric("Variância", f"{variancia:.4f}")

    col2.metric("Média", f"{media:.4f}")
    col2.metric("Máximo", f"{maximo:.4f}")
    col2.metric("Desvio-Padrão", f"{desvio_padrao:.4f}")

    col3.metric("Mediana", f"{mediana:.4f}")
    col3.metric("Amplitude", f"{amplitude:.4f}")
    col3.metric("Coef. Variação (CV)", f"{cv:.4f}%")

    col4.metric("1º Quartil (Q1)", f"{q1:.4f}")
    col4.metric("3º Quartil (Q3)", f"{q3:.4f}")
    col4.metric("Moda", moda_texto)

    # 2. Intervalo de Confiança (Z e t) com 4 casas
    st.subheader("2. Intervalos de Confiança para a Média ($\mu$)")
    st.write("Calculado diretamente com base nos parâmetros estatísticos informados.")

    niveis = [0.90, 0.95, 0.955, 0.99]
    ic_dados = []

    ep = desvio_padrao / np.sqrt(n)
    graus_liberdade = n - 1

    for nivel in niveis:
        # Distribuição Normal (Z)
        alpha_z = 1 - nivel
        z_score = norm.ppf(1 - alpha_z / 2)
        margem_z = z_score * ep
        
        # Distribuição t de Student (t)
        alpha_t = 1 - nivel
        t_score = t.ppf(1 - alpha_t / 2, df=graus_liberdade)
        margem_t = t_score * ep

        ic_dados.append({
            "Nível de Confiança": f"{nivel*100:.1f}%".replace('.0%', '%'),
            "IC Normal (Z) - Inferior": f"{media - margem_z:.4f}",
            "IC Normal (Z) - Superior": f"{media + margem_z:.4f}",
            "IC t-Student (t) - Inferior": f"{media - margem_t:.4f}",
            "IC t-Student (t) - Superior": f"{media + margem_t:.4f}"
        })

    df_ic = pd.DataFrame(ic_dados)
    st.dataframe(df_ic, use_container_width=True)

    # 3. Visualização Gráfica (Apenas se houver dados brutos)
    if tem_dados_brutos:
        st.subheader("3. Visualização Gráfica")
        sns.set_theme(style="whitegrid")

        nome_variavel = st.text_input("Nome da variável para os eixos:", value="Valor")
        tab1, tab2, tab3 = st.tabs(["Histograma (KDE)", "Boxplot", "Gráfico de Dispersão"])

        with tab1:
            fig1, ax1 = plt.subplots(figsize=(8, 4))
            sns.histplot(lista_dados, kde=True, color='#2c7fb8', ax=ax1)
            ax1.set_title(f'Histograma com Curva de Densidade - {nome_variavel}')
            ax1.set_xlabel(nome_variavel)
            ax1.set_ylabel('Frequência')
            st.pyplot(fig1)

        with tab2:
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            sns.boxplot(x=lista_dados, color='#7fcdbb', ax=ax2)
            ax2.set_title(f'Boxplot - {nome_variavel}')
            ax2.set_xlabel(nome_variavel)
            st.pyplot(fig2)

        with tab3:
            fig3, ax3 = plt.subplots(figsize=(8, 4))
            sns.scatterplot(x=range(1, len(lista_dados)+1), y=lista_dados, color='#2ca25f', s=60, ax=ax3)
            ax3.set_title('Gráfico de Valores Individuais')
            ax3.set_xlabel('Número da Observação')
            ax3.set_ylabel(nome_variavel)
            st.pyplot(fig3)
    else:
        st.info("ℹ️ Os gráficos visuais ficam disponíveis quando o modo 'Digitar Dados Brutos' está selecionado.")