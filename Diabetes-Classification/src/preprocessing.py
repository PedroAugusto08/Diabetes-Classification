import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def sample_stratified_df(
    df: pd.DataFrame,
    n_samples: int,
    target_col: str = "diagnosed_diabetes",
    random_state: int = 42,
) -> pd.DataFrame:
    if n_samples < len(df):
        df, _ = train_test_split(
            df,
            train_size=n_samples,
            stratify=df[target_col],
            random_state=random_state,
        )
    return df.reset_index(drop=True)

def load_raw_data(path: str, n_samples: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Dataset original: {len(df)} linhas")
    sampled_df = sample_stratified_df(df=df, n_samples=n_samples)
    print(f"Dataset reduzido: {len(sampled_df)} linhas")
    return sampled_df

def prepare_features(df: pd.DataFrame):
    # Separar atributos e alvo
    X = df.drop(columns=[
        "diagnosed_diabetes",
        "diabetes_stage",
        "diabetes_risk_score"
    ])
    y = df["diagnosed_diabetes"].values
    # Split treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )
    # Separa colunas categóricas e numéricas para evitar one-hot em variáveis contínuas.
    categorical_cols = X_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    numeric_cols = [
        col for col in X_train.columns
        if col not in categorical_cols
    ]

    X_train_num = X_train[numeric_cols].reset_index(drop=True)
    X_test_num = X_test[numeric_cols].reset_index(drop=True)

    if categorical_cols:
        # One-hot somente nas colunas categóricas.
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        X_train_cat = encoder.fit_transform(X_train[categorical_cols])
        X_test_cat = encoder.transform(X_test[categorical_cols])

        cat_feature_names = encoder.get_feature_names_out(categorical_cols).tolist()
        X_train_cat_df = pd.DataFrame(X_train_cat, columns=cat_feature_names)
        X_test_cat_df = pd.DataFrame(X_test_cat, columns=cat_feature_names)

        X_train_final = pd.concat([X_train_num, X_train_cat_df], axis=1)
        X_test_final = pd.concat([X_test_num, X_test_cat_df], axis=1)
        feature_names = X_train_final.columns.tolist()
    else:
        X_train_final = X_train_num
        X_test_final = X_test_num
        feature_names = numeric_cols

    return (
        X_train_final.to_numpy(),
        X_test_final.to_numpy(),
        y_train,
        y_test,
        feature_names,
    )


def load_and_prepare_data(path, n_samples):
    df = load_raw_data(path=path, n_samples=n_samples)
    return prepare_features(df)