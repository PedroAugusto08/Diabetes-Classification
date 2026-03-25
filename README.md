<p align="center"> 
  <img src="Diabetes-Classification/imgs/logo_azul.png" alt="CEFET-MG" width="100px" height="100px">
</p>

<h1 align="center">
Diabetes Classification
</h1>

<h3 align="center">
Desenvolvimento de um experimento que contempla análise e compreensão dos dados, desde pré-processamento à definição de uma metodologia experimental adequada para a classificação de pacientes com Diabete.
</h3>

<div align="center">

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

</div>

---

<div align="justify">
<p><strong>Disciplina:</strong> Inteligência Computacional<br>
<strong>Instituição:</strong> Centro Federal de Educação Tecnológica de Minas Gerais (CEFET-MG) - Campus V Divinópolis<br>
<strong>Professor:</strong> Alisson Marques da Silva<br>
<strong>Projeto:</strong> "Atividade Prática 01 - Metodologia Experimental"<br>
</div>


---


## Projeto e Base de Dados

Segundo José Antonio Miguel Marcondes (2003), o diabete afeta aproximadamente dez milhões de brasileiros e sua incidência vem aumentando, sobretudo no diabetes tipo 2. Nesse contexto, a classificação de pacientes é relevante para suporte a ações de prevenção e acompanhamento.

Este trabalho, desenvolvido na disciplina de Inteligência Computacional, utiliza a base pública **Diabetes Health Indicators Dataset** (Kaggle), com `100.000` registros e `31` atributos, para um problema de **classificação binária** cujo alvo é `diagnosed_diabetes` (`0` ou `1`). A base reúne variáveis sociodemográficas, hábitos de vida, histórico clínico e biomarcadores, o que a torna adequada ao enunciado por conter atributos heterogêneos e permitir uma metodologia experimental realista.


## Metodologia Adotada (Pré-processamento + Modelagem)

As implementações principais estão em `Diabetes-Classification/src/preprocessing.py`, `Diabetes-Classification/src/models.py` e `Diabetes-Classification/src/main.py`.

No pré-processamento, adotou-se: leitura da base, amostragem estratificada opcional, separação entre variáveis preditoras e alvo, divisão treino/teste estratificada (`80/20`) e codificação categórica via `OneHotEncoder` apenas nas colunas não numéricas. As colunas `diabetes_stage` e `diabetes_risk_score` foram removidas por representarem informação fortemente ligada ao próprio desfecho, reduzindo risco de enviesamento da avaliação.

```python
if n_samples < len(df):
    df, _ = train_test_split(
        df,
        train_size=n_samples,
        stratify=df["diagnosed_diabetes"],
        random_state=42,
    )
```

Na modelagem, optou-se por `Pipeline` para padronizar o fluxo e preservar reprodutibilidade. A seleção de atributos é embarcada com `SelectFromModel` + `RandomForestClassifier` (`threshold="1.2*median"`), seguida de padronização quando necessário. Foram definidos três classificadores (`KNN`, `SVM` e `DecisionTree`), em consonância com o enunciado.

```python
def get_knn_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("feature_selection", _get_feature_selector()),
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier()),
        ]
    )
```

As decisões acima foram justificadas por critérios técnicos e empíricos: preservação da proporção de classes, prevenção de vazamento de informação, comparabilidade entre algoritmos e redução de dimensionalidade para favorecer generalização.


## Como Executar

Instalação de dependências:

```bash
pip install pandas numpy scikit-learn
```

Execução do pipeline atual:

```bash
cd Diabetes-Classification
python src/main.py
```

Observação: `src/main.py` é interativo e solicita a quantidade de amostras e o modelo a ser avaliado.

Teste rápido do pré-processamento:

```bash
cd Diabetes-Classification
python - <<'PY'
from src.preprocessing import load_and_prepare_data

X_train, X_test, y_train, y_test, feature_names = load_and_prepare_data(
    path="data/diabetes_dataset.csv",
    n_samples=20000,
)

print("Treino:", X_train.shape)
print("Teste:", X_test.shape)
print("Atributos finais:", len(feature_names))
PY
```



## Referências

MARCONDES, José Antonio Miguel. Diabete melito: fisiopatologia e tratamento. Revista da Faculdade de Ciências Médicas de Sorocaba, [S. l.], v. 5, n. 1, p. 18–26, 2007. Disponível em: https://revistas.pucsp.br/index.php/RFCMS/article/view/117. Acesso em: 24 mar. 2026.

Rakesh Kolipaka, and Ranjith Kumar Digutla. (2025). Diabetes Health Indicators Dataset [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/13128284


