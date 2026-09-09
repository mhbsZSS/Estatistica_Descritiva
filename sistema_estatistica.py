import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde, norm, t, shapiro, probplot

# 1. Configuração da Página
st.set_page_config(page_title="Sistema Estatístico Completo", layout="wide")
st.title("📊 Sistema Estatístico: Descritiva, Inferência & Normalidade")
st.write("Plataforma para análise descritiva, inferência, testes paramétricos e verificação de normalidade.")

# 2. Navegação Principal por Módulos
modulo = st.sidebar.selectbox(
    "Escolha o Módulo de Análise:",
    (
        "1. Estatística Descritiva & IC (Dados ou Sumário)", 
        "2. Testes de Hipóteses (Z e t)",
        "3. Teste de Normalidade (Shapiro-Wilk)"
    )
)

# ==========================================
# MÓDULO 1: DESCRITIVA E INTERVALO DE CONFIANÇA
# ==========================================
if modulo == "1. Estatística Descritiva & IC (Dados ou Sumário)":
    modo_entrada = st.radio("Forma de entrada dos dados:", ("Digitar Dados Brutos", "Informar Média, Desvio-Padrão e n"), horizontal=True)

    media, desvio_padrao, n = None, None, None
    lista_dados, tem_dados_brutos = [], False

    if modo_entrada == "Digitar Dados Brutos":
        dados_padrao = "198.56; 199.44; 201.25; 198.90; 197.75; 198.45; 199.25; 200.05; 199.65"
        entrada_usuario = st.text_area("Insira os dados (separados por ponto e vírgula):", value=dados_padrao)
        
        if entrada_usuario:
            try:
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

                contagem_freq = df_temp['Variavel'].value_counts()
                if contagem_freq.iloc[0] > 1:
                    moda_texto = f"{contagem_freq.index[0]:.4f} (Freq: {contagem_freq.iloc[0]})"
                else:
                    kde = gaussian_kde(lista_dados)
                    x_vals = np.linspace(minimo, maximo, 1000)
                    moda_estimada = x_vals[np.argmax(kde(x_vals))]
                    moda_texto = f"{moda_estimada:.4f} (Estimada KDE)"
            except Exception as e:
                st.error(f"Erro na leitura: {e}")
    else:
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            media = st.number_input("Média amostral (x̄):", value=199.2556, format="%.4f")
        with col_m2:
            desvio_padrao = st.number_input("Desvio Padrão (s):", value=1.0201, format="%.4f")
        with col_m3:
            n = int(st.number_input("Tamanho da Amostra (n):", value=9, min_value=2, step=1))
        
        mediana, minimo, maximo, amplitude, q1, q3, variancia = media, media-1, media+1, 2, media-0.5, media+0.5, desvio_padrao**2
        cv = (desvio_padrao / media) * 100
        moda_texto = "N/A (Exige dados brutos)"

    if n is not None and media is not None:
        st.success(f"Amostra ativa configurada com n = {n}")
        st.subheader("Resumo Estatístico (Precisão: 4 Casas Decimais)")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Número de Obs (n)", int(n))
        c1.metric("Mínimo", f"{minimo:.4f}" if tem_dados_brutos else "N/A")
        c1.metric("Variância", f"{variancia:.4f}")

        c2.metric("Média", f"{media:.4f}")
        c2.metric("Máximo", f"{maximo:.4f}" if tem_dados_brutos else "N/A")
        c2.metric("Desvio-Padrão", f"{desvio_padrao:.4f}")

        c3.metric("Mediana", f"{mediana:.4f}" if tem_dados_brutos else f"{media:.4f}")
        c3.metric("Amplitude", f"{amplitude:.4f}" if tem_dados_brutos else "N/A")
        c3.metric("Coef. Variação (CV)", f"{cv:.4f}%")

        c4.metric("1º Quartil (Q1)", f"{q1:.4f}" if tem_dados_brutos else "N/A")
        c4.metric("3º Quartil (Q3)", f"{q3:.4f}" if tem_dados_brutos else "N/A")
        c4.metric("Moda", moda_texto)

        st.subheader("Intervalos de Confiança para a Média (μ)")
        ep = desvio_padrao / np.sqrt(n)
        ic_dados = []
        for nivel in [0.90, 0.95, 0.955, 0.99]:
            alpha = 1 - nivel
            z_m = norm.ppf(1 - alpha / 2) * ep
            t_m = t.ppf(1 - alpha / 2, df=n-1) * ep
            ic_dados.append({
                "Nível de Confiança": f"{nivel*100:.1f}%".replace('.0%', '%'),
                "IC Normal (Z) - Inferior": f"{media - z_m:.4f}",
                "IC Normal (Z) - Superior": f"{media + z_m:.4f}",
                "IC t-Student (t) - Inferior": f"{media - t_m:.4f}",
                "IC t-Student (t) - Superior": f"{media + t_m:.4f}"
            })
        st.dataframe(pd.DataFrame(ic_dados), use_container_width=True)

        if tem_dados_brutos:
            st.subheader("Visualização Gráfica")
            sns.set_theme(style="whitegrid")
            nome_variavel = st.text_input("Nome da variável para os eixos:", value="Valor", key="var_mod1")
            
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

# ==========================================
# MÓDULO 2: TESTES DE HIPÓTESES (Z e t)
# ==========================================
elif modulo == "2. Testes de Hipóteses (Z e t)":
    st.subheader("🧪 Módulo de Testes de Hipóteses Paramétricos")
    tipo_teste = st.selectbox(
        "Selecione o Teste Estatístico:",
        ("Teste Z (1 Amostra)", "Teste t (1 Amostra)", "Teste Z (2 Amostras Independentes)", "Teste t (2 Amostras Independentes)")
    )
    
    tipo_cauda = st.selectbox(
        "Tipo de Hipótese Alternativa (H1):",
        ("Bicaudal (≠)", "Unilateral à Esquerda (<)", "Unilateral à Direita (>)")
    )
    
    alpha_nivel = st.selectbox("Nível de Significância (α):", [0.01, 0.05, 0.10], index=1)

    if tipo_teste == "Teste Z (1 Amostra)":
        st.write("---")
        col_z1, col_z2, col_z3, col_z4 = st.columns(4)
        media_amostra = col_z1.number_input("Média Amostral (x̄)", value=199.2556, format="%.4f")
        media_pop = col_z2.number_input("Média Hipótese (μ0)", value=200.0000, format="%.4f")
        sigma_pop = col_z3.number_input("Desvio Padrão Pop. (σ)", value=1.0201, format="%.4f")
        n_amostra = int(col_z4.number_input("Tamanho (n)", value=9, min_value=1))

        if st.button("Executar Teste Z"):
            ep = sigma_pop / np.sqrt(n_amostra)
            z_calc = (media_amostra - media_pop) / ep
            
            if tipo_cauda == "Bicaudal (≠)":
                p_val = 2 * (1 - norm.cdf(abs(z_calc)))
                h1_str = rf"\mu \neq {media_pop}"
            elif tipo_cauda == "Unilateral à Esquerda (<)":
                p_val = norm.cdf(z_calc)
                h1_str = rf"\mu < {media_pop}"
            else:
                p_val = 1 - norm.cdf(z_calc)
                h1_str = rf"\mu > {media_pop}"

            st.markdown("### Hipóteses:")
            st.latex(rf"H_0: \mu = {media_pop}")
            st.latex(rf"H_1: {h1_str}")
            
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Estatística Z Calculada", f"{z_calc:.4f}")
            sc2.metric("P-Valor", f"{p_val:.4f}")
            rejeita = p_val < alpha_nivel
            sc3.metric(f"Conclusão (α = {alpha_nivel})", "Rejeita H0" if rejeita else "Não Rejeita H0")

    elif tipo_teste == "Teste t (1 Amostra)":
        st.write("---")
        col_t1, col_t2, col_t3, col_t4 = st.columns(4)
        media_amostra = col_t1.number_input("Média Amostral (x̄)", value=199.2556, format="%.4f")
        media_pop = col_t2.number_input("Média Hipótese (μ0)", value=200.0000, format="%.4f")
        s_amostra = col_t3.number_input("Desvio Padrão Amostral (s)", value=1.0201, format="%.4f")
        n_amostra = int(col_t4.number_input("Tamanho (n)", value=9, min_value=2))

        if st.button("Executar Teste t"):
            ep = s_amostra / np.sqrt(n_amostra)
            t_calc = (media_amostra - media_pop) / ep
            df_val = n_amostra - 1
            
            if tipo_cauda == "Bicaudal (≠)":
                p_val = 2 * (1 - t.cdf(abs(t_calc), df=df_val))
                h1_str = rf"\mu \neq {media_pop}"
            elif tipo_cauda == "Unilateral à Esquerda (<)":
                p_val = t.cdf(t_calc, df=df_val)
                h1_str = rf"\mu < {media_pop}"
            else:
                p_val = 1 - t.cdf(t_calc, df=df_val)
                h1_str = rf"\mu > {media_pop}"

            st.markdown("### Hipóteses:")
            st.latex(rf"H_0: \mu = {media_pop}")
            st.latex(rf"H_1: {h1_str}")
            
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Estatística t Calculada", f"{t_calc:.4f}")
            sc2.metric("P-Valor", f"{p_val:.4f}")
            rejeita = p_val < alpha_nivel
            sc3.metric(f"Conclusão (α = {alpha_nivel})", "Rejeita H0" if rejeita else "Não Rejeita H0")

    elif tipo_teste == "Teste Z (2 Amostras Independentes)":
        st.write("---")
        col_2z1, col_2z2, col_2z3, col_2z4 = st.columns(4)
        m1 = col_2z1.number_input("Média Amostra 1 (x̄1)", value=10.5, format="%.4f")
        s1 = col_2z2.number_input("Desvio Padrão 1 (σ1)", value=2.1, format="%.4f")
        n1 = int(col_2z3.number_input("Tamanho 1 (n1)", value=30, min_value=1))
        
        col_2z5, col_2z6, col_2z7 = st.columns(3)
        m2 = col_2z5.number_input("Média Amostra 2 (x̄2)", value=9.8, format="%.4f")
        s2 = col_2z6.number_input("Desvio Padrão 2 (σ2)", value=1.9, format="%.4f")
        n2 = int(col_2z7.number_input("Tamanho 2 (n2)", value=30, min_value=1))

        if st.button("Executar Teste Z (2 Amostras)"):
            erro_padrao = np.sqrt((s1**2 / n1) + (s2**2 / n2))
            z_calc = (m1 - m2) / erro_padrao
            
            if tipo_cauda == "Bicaudal (≠)":
                p_val = 2 * (1 - norm.cdf(abs(z_calc)))
                h1_str = r"\mu_1 \neq \mu_2"
            elif tipo_cauda == "Unilateral à Esquerda (<)":
                p_val = norm.cdf(z_calc)
                h1_str = r"\mu_1 < \mu_2"
            else:
                p_val = 1 - norm.cdf(z_calc)
                h1_str = r"\mu_1 > \mu_2"

            st.markdown("### Hipóteses:")
            st.latex(r"H_0: \mu_1 = \mu_2")
            st.latex(rf"H_1: {h1_str}")
            
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Estatística Z", f"{z_calc:.4f}")
            sc2.metric("P-Valor", f"{p_val:.4f}")
            sc3.metric("Conclusão", "Rejeita H0" if p_val < alpha_nivel else "Não Rejeita H0")

    elif tipo_teste == "Teste t (2 Amostras Independentes)":
        st.write("---")
        col_2t1, col_2t2, col_2t3 = st.columns(3)
        m1 = col_2t1.number_input("Média Amostra 1 (x̄1)", value=10.5, format="%.4f", key="tm1")
        s1 = col_2t2.number_input("Desvio Padrão 1 (s1)", value=2.1, format="%.4f", key="ts1")
        n1 = int(col_2t3.number_input("Tamanho 1 (n1)", value=30, min_value=2, key="tn1"))
        
        col_2t5, col_2t6, col_2t7 = st.columns(3)
        m2 = col_2t5.number_input("Média Amostra 2 (x̄2)", value=9.8, format="%.4f", key="tm2")
        s2 = col_2t6.number_input("Desvio Padrão 2 (s2)", value=1.9, format="%.4f", key="ts2")
        n2 = int(col_2t7.number_input("Tamanho 2 (n2)", value=30, min_value=2, key="tn2"))

        if st.button("Executar Teste t (2 Amostras)"):
            se_diff = np.sqrt((s1**2 / n1) + (s2**2 / n2))
            t_calc = (m1 - m2) / se_diff
            df_welch = ((s1**2/n1 + s2**2/n2)**2) / (((s1**2/n1)**2 / (n1-1)) + ((s2**2/n2)**2 / (n2-1)))
            
            if tipo_cauda == "Bicaudal (≠)":
                p_val = 2 * (1 - t.cdf(abs(t_calc), df=df_welch))
                h1_str = r"\mu_1 \neq \mu_2"
            elif tipo_cauda == "Unilateral à Esquerda (<)":
                p_val = t.cdf(t_calc, df=df_welch)
                h1_str = r"\mu_1 < \mu_2"
            else:
                p_val = 1 - t.cdf(t_calc, df=df_welch)
                h1_str = r"\mu_1 > \mu_2"

            st.markdown("### Hipóteses:")
            st.latex(r"H_0: \mu_1 = \mu_2")
            st.latex(rf"H_1: {h1_str}")
            
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Estatística t (Welch)", f"{t_calc:.4f}")
            sc2.metric("P-Valor", f"{p_val:.4f}")
            sc3.metric("Conclusão", "Rejeita H0" if p_val < alpha_nivel else "Não Rejeita H0")

# ==========================================
# MÓDULO 3: TESTE DE NORMALIDADE (SHAPIRO-WILK)
# ==========================================
elif modulo == "3. Teste de Normalidade (Shapiro-Wilk)":
    st.subheader("📈 Verificação de Normalidade (Shapiro-Wilk)")
    st.write("Insira os dados brutos da amostra para verificar se seguem uma distribuição normal, gerando as estatísticas $W$ e $p$, além dos gráficos.")

    entrada_normalidade = st.text_area(
        "Insira os dados da amostra (separados por ponto e vírgula):", 
        value="12.5; 13.1; 12.8; 14.3; 13.7; 15.2; 12.0; 12.9; 13.5; 14.8",
        key="shapiro_input"
    )
    
    alpha_normalidade = st.selectbox("Nível de Significância ($\alpha$):", [0.01, 0.05, 0.10], index=1, key="shapiro_alpha")

    if entrada_normalidade:
        try:
            dados_shapiro = [float(x.strip()) for x in entrada_normalidade.replace(',', '.').split(';') if x.strip()]
            n_shapiro = len(dados_shapiro)
            
            if n_shapiro < 3:
                st.warning("O teste de Shapiro-Wilk requer pelo menos 3 observações.")
            else:
                stat_w, p_valor_sw = shapiro(dados_shapiro)
                
                st.markdown("---")
                st.markdown("### a) Hipóteses do Teste")
                st.latex(r"H_0: \text{Os dados seguem uma distribuição Normal.}")
                st.latex(r"H_1: \text{Os dados NÃO seguem uma distribuição Normal.}")
                
                col_sw1, col_sw2, col_sw3 = st.columns(3)
                col_sw1.metric("b) Estatística W", f"{stat_w:.4f}")
                col_sw2.metric("c) P-Valor", f"{p_valor_sw:.4f}")
                
                rejeita_sw = p_valor_sw < alpha_normalidade
                conclusao = "Não Rejeita $H_0$ (Aprox. Normais)" if not rejeita_sw else "Rejeita $H_0$ (Não Normais)"
                col_sw3.metric("d) Conclusão", conclusao)
                
                if not rejeita_sw:
                    st.success(f"**Resposta (d):** Como o p-valor ({p_valor_sw:.4f}) é maior que $\\alpha$ ({alpha_normalidade}), os dados **podem ser considerados aproximadamente normais**.")
                else:
                    st.error(f"**Resposta (d):** Como o p-valor ({p_valor_sw:.4f}) é menor que $\\alpha$ ({alpha_normalidade}), os dados **NÃO podem ser considerados aproximadamente normais**.")

                st.markdown("---")
                st.subheader("e) Verificação Gráfica")
                sns.set_theme(style="whitegrid")
                
                tab_g1, tab_g2, tab_g3 = st.tabs(["Gráfico Q-Q", "Histograma", "Boxplot"])
                
                with tab_g1:
                    fig_qq, ax_qq = plt.subplots(figsize=(8, 4))
                    probplot(dados_shapiro, dist="norm", plot=ax_qq)
                    ax_qq.set_title("Gráfico Q-Q (Quantile-Quantile)")
                    st.pyplot(fig_qq)
                    st.caption("Se os pontos estiverem próximos à linha vermelha, isso indica normalidade.")
                    
                with tab_g2:
                    fig_hist, ax_hist = plt.subplots(figsize=(8, 4))
                    sns.histplot(dados_shapiro, kde=True, color='#2c7fb8', ax=ax_hist)
                    ax_hist.set_title("Histograma com Curva de Distribuição")
                    st.pyplot(fig_hist)
                    st.caption("Uma curva em forma de 'sino' indica normalidade.")
                    
                with tab_g3:
                    fig_box, ax_box = plt.subplots(figsize=(8, 4))
                    sns.boxplot(x=dados_shapiro, color='#7fcdbb', ax=ax_box)
                    ax_box.set_title("Boxplot")
                    st.pyplot(fig_box)
                    st.caption("Um boxplot simétrico, sem valores atípicos (outliers) severos, apoia a normalidade.")

        except Exception as e:
            st.error(f"Erro no processamento dos dados: {e}")