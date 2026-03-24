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


## Sobre o Projeto

Segundo José Antonio Miguel Marcondes (2003), o diabete afeta, aproximadamente, dez milhões de brasileiros e sua incidência está aumentando não só no Brasil, mas em todo o mundo, principalmente o diabetes tipo 2, cuja prevalência tem aumentado significamente em adolescentes e adultos jovens. Tendo isto em vista, a classificação de pacientes (neste contexto, dentro um grupo selecionado de dados) torna-se de suma importância para previsão e profilaxia do distúrbio.

Ademais, utilizando-se dos conhecimentos adquiridos na disciplina de Inteligência Computacional acerca da análise e compreensão destes dados, almejando-se um bom experimento devido à sua reprodutibilidade, transparência e solidez científica. A base de dados utilizada foi retirada do Kaggle, sintetizada por Mohan Krishna Thalla seguindo relações médicas e de saúde pública comumente aceitas e amplamente relatadas em diversos estudos e fontes públicas. A base, por si só, foi tratada posteriormente de acordo com as etapas previstas pelas orientações do trabalho, seguindo a estrutura resumida a seguir:

- **Escolha do Conjunto de Dados**: O conjunto de dados escolhido enquadra-se numa tarefa de classificação binária, contém atributos de tipos distintos (como idade que é um número inteiro e gênero que é uma string), com presença de um desafio condizente com a realidade (excesso de atributos que serão selecionados posteriormente) e com documentação pública no Kaggle.
- **Análise e Compreensão dos Dados**: Identificou-se os problemas à serem resolvidos quando tange os obstáculos da ciência dos dados, além do problema ser em essência de classificação (entre ter ou não diabetes). Além do pré-processamento e metodologia experimental.


## Estrutura do Repositório

```text
Diabetes-Classification/
├── data/
│   └── diabetes_dataset.csv
├── imgs/
└── src/
    ├── preprocessing.py
    └── models.py
```


## Base de Dados Utilizada

A seguir tem-se a descrição das características essencias da base de dados utilizada:

- *Fonte:* Kaggle — *Diabetes Health Indicators Dataset*.
- *Volume:* `100.002` registros e `31` atributos.
- *Alvo da classificação:* `diagnosed_diabetes` (binário: `0` ou `1`).

Houve colunas auxiliares removidas no experimento atual (sendo elas `diabetes_stage` e `diabetes_risk_score`), devido ao que estas representam, pois elas auxiliam diretamente na classificação do algoritmo através da representação de estágios da diabetes e de uma pontuação que representa o risco de estar com tal. Além disso, tem-se alguns outros atributos presentes na base a seguir:

- Sociodemográficos: `age`, `gender`, `ethnicity`, `education_level`, `income_level`.
- Hábitos de vida: `smoking_status`, `alcohol_consumption_per_week`, `physical_activity_minutes_per_week`, `diet_score`, `sleep_hours_per_day`, `screen_time_hours_per_day`.
- Histórico clínico: `family_history_diabetes`, `hypertension_history`, `cardiovascular_history`.
- Biomarcadores: `bmi`, `waist_to_hip_ratio`, `systolic_bp`, `diastolic_bp`, `cholesterol_total`, `glucose_fasting`, `hba1c`.


## Etapas de Pré-processamento 

As etapas atuais estão implementadas em `Diabetes-Classification/src/preprocessing.py`, por meio da função `load_and_prepare_data(path, n_samples)`, esta segue o fluxo apresentado:

1 - Leitura da base com a biblioteca `pandas`.

2 - Amostragem estratificada opcional para reduzir o volume mantendo proporção de classes (visto que os dados são volumosos).

3 - Separação entre variáveis preditoras e variável alvo.

4 - Divisão treino/teste estratificada (`80/20`) com reprodutibilidade (`random_state=42`), tal divisão foi adotada de acordo com a literatura.

5 - Verificação da distribuição das classes em treino e teste.

O trecho abaixo garante **amostragem com estratificação**, importante para evitar viés de classe durante a redução da base.

```python
df = pd.read_csv(path)

if n_samples < len(df):
  df, _ = train_test_split(
    df,
    train_size=n_samples,
    stratify=df["diagnosed_diabetes"],
    random_state=42
  )
```

Adicionalmente, o trecho abaixo define explicitamente o alvo do problema, removendo colunas não utilizadas e aplica *particionamento estratificado para avaliação mais consistente.


```python
X = df.drop(columns=[
  "diagnosed_diabetes",
  "diabetes_stage",
  "diabetes_risk_score"
]).values

y = df["diagnosed_diabetes"].values

X_train, X_test, y_train, y_test = train_test_split(
  X, y,
  test_size=0.2,
  stratify=y,
  random_state=42
)
```


## Como Executar 



## Referências

MARCONDES, José Antonio Miguel. Diabete melito: fisiopatologia e tratamento. Revista da Faculdade de Ciências Médicas de Sorocaba, [S. l.], v. 5, n. 1, p. 18–26, 2007. Disponível em: https://revistas.pucsp.br/index.php/RFCMS/article/view/117. Acesso em: 24 mar. 2026.

Rakesh Kolipaka, and Ranjith Kumar Digutla. (2025). Diabetes Health Indicators Dataset [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/13128284


