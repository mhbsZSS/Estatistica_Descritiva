# 📊 Sistema Completo de Análise Estatística & Inferência

Aplicação web interativa desenvolvida em Python com **Streamlit**, projetada para auxiliar em aulas e análises de estatística descritiva e inferencial (seguindo o padrão de softwares como JASP/Minitab).

## 🚀 Funcionalidades Atuais
* **Resumo Estatístico Completo**: Média, mediana, moda (via estimativa KDE para dados contínuos), mínimo, máximo, amplitude, quartis $Q_1$ e $Q_3$, variância, desvio-padrão e coeficiente de variação (CV) com precisão de 4 casas decimais.
* **Inferência Estatística**: Cálculo automático de **Intervalos de Confiança (IC)** para a média ($\mu$) utilizando a **Distribuição Normal ($Z$)** e a **Distribuição t de Student ($t$)** para os níveis de 90%, 95%, 95,5% e 99%.
* **Entrada Flexível**: Permite alternar entre a digitação de dados brutos ou a inserção direta de parâmetros sumários ($n$, $\bar{x}$, $s$).
* **Visualização Gráfica**: Geração dinâmica de histogramas com curva de densidade (KDE), boxplots e gráficos de valores individuais.

## 🛠️ Como Executar o Projeto Localmente
1. Clone este repositório:
   ```bash
   git clone [https://github.com/SEU-USUARIO/nome-do-repositorio.git](https://github.com/SEU-USUARIO/nome-do-repositorio.git)

2. Crie e ative um ambiente virtual (.venv):
   ```bash
   python -m venv .venv
  .venv\Scripts\activate  # No Windows

3. Instale as dependências:
   ```bash
  pip install -r requirements.txt

4. Execute o sistema:
  ```bash
  streamlit run sistema_estatistica.py

