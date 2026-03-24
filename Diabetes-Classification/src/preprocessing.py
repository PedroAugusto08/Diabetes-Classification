import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_prepare_data(path, n_samples):
    """
    Carrega o dataset, aplica amostragem estratificada e divide em treino/teste.
    """

    # Carregar dataset
    df = pd.read_csv(path)

    print(f"Dataset original: {len(df)} linhas")

    # Amostragem estratificada
    if n_samples < len(df):
        df, _ = train_test_split(
            df,
            train_size=n_samples,
            stratify=df["diagnosed_diabetes"],
            random_state=42
        )

    df = df.reset_index(drop=True)

    print(f"Dataset reduzido: {len(df)} linhas")

    # Separar features e alvo
    X = df.drop(columns=[
        "diagnosed_diabetes",
        "diabetes_stage",
        "diabetes_risk_score"
    ]).values

    y = df["diagnosed_diabetes"].values

    # Split treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # Verificação (opcional)
    print("\nDistribuição das classes:")
    print("Treino:", pd.Series(y_train).value_counts(normalize=True))
    print("Teste :", pd.Series(y_test).value_counts(normalize=True))

    return X_train, X_test, y_train, y_test