import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
def load_and_prepare_data(path, n_samples):
    df = pd.read_csv(path) 
    print(f"Dataset original: {len(df)} linhas")
    # amostragem estratificada
    if n_samples < len(df):
        df, _ = train_test_split(
            df,
            train_size=n_samples,
            stratify=df["diagnosed_diabetes"],
            random_state=42
        )
    df = df.reset_index(drop=True)
    print(f"Dataset reduzido: {len(df)} linhas")
    # separar atributos e alvo
    X = df.drop(columns=[
        "diagnosed_diabetes",
        "diabetes_stage",
        "diabetes_risk_score"
    ]).values
    y = df["diagnosed_diabetes"].values
    # split treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )
    # codificação de atributos categóricos
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_train = encoder.fit_transform(X_train)
    X_test = encoder.transform(X_test)    

    print("\nDistribuição das classes:")
    print("Treino:", pd.Series(y_train).value_counts(normalize=True))
    print("Teste :", pd.Series(y_test).value_counts(normalize=True))
    return X_train, X_test, y_train, y_test

load_and_prepare_data("Diabetes-Classification/data/diabetes_dataset.csv", n_samples=1000)