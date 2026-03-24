"""
Este módulo define os modelos e seus passos de pré-processamento
( seleção de atributos e, quando necessário, padronização ).
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def _get_feature_selector() -> SelectFromModel:
    # Cria o seletor de atributos baseado em importância de features.
    selector_estimator = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    return SelectFromModel(
        estimator=selector_estimator,
        threshold="1.2*median",
    )


def get_knn_pipeline() -> Pipeline:
    # Retorna o pipeline de classificação com KNN.
    return Pipeline(
        steps=[
            ("feature_selection", _get_feature_selector()),
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier()),
        ]
    )


def get_svm_pipeline() -> Pipeline:
    # Retorna o pipeline de classificação com SVM.
    return Pipeline(
        steps=[
            ("feature_selection", _get_feature_selector()),
            ("scaler", StandardScaler()),
            ("model", SVC(probability=True, random_state=42)),
        ]
    )


def get_dt_pipeline() -> Pipeline:
    # Retorna o pipeline de classificação com Árvore de Decisão.
    return Pipeline(
        steps=[
            ("feature_selection", _get_feature_selector()),
            (
                "model",
                DecisionTreeClassifier(random_state=42),
            ),
        ]
    )


def get_all_models() -> dict[str, Pipeline]:
    # Retorna todos os pipelines de classificação disponíveis.
    return {
        "KNN": get_knn_pipeline(),
        "SVM": get_svm_pipeline(),
        "DecisionTree": get_dt_pipeline(),
    }


def get_selected_feature_names(
    trained_pipeline: Pipeline,
    feature_names: list[str],
) -> list[str]:
    # Retorna os nomes das features escolhidas pelo seletor do pipeline.
    # Garante que o pipeline tenha a etapa esperada.
    if "feature_selection" not in trained_pipeline.named_steps:
        raise ValueError("O pipeline não possui a etapa 'feature_selection'.")

    selector = trained_pipeline.named_steps["feature_selection"]

    if not hasattr(selector, "get_support"):
        raise TypeError(
            "A etapa 'feature_selection' não suporta get_support()."
        )

    try:
        support_mask = selector.get_support()
    except Exception as exc:
        raise ValueError(
            "A etapa de seleção ainda não foi treinada. Execute fit antes."
        ) from exc

    if len(feature_names) != len(support_mask):
        raise ValueError(
            "A quantidade de feature_names não corresponde ao número de atributos vistos pelo pipeline."
        )

    # Mantém apenas os nomes cuja posição foi marcada como selecionada.
    return [
        feature_name
        for feature_name, was_selected in zip(feature_names, support_mask)
        if was_selected
    ]


def get_selected_feature_importances(
    trained_pipeline: Pipeline,
    feature_names: list[str],
) -> list[tuple[str, float]]:
    # Retorna (feature, importância) para as features selecionadas.
    
    # Valida se a etapa de seleção existe no pipeline treinado.
    if "feature_selection" not in trained_pipeline.named_steps:
        raise ValueError("O pipeline não possui a etapa 'feature_selection'.")

    selector = trained_pipeline.named_steps["feature_selection"]

    # A máscara booleana indica quais colunas foram mantidas.
    try:
        support_mask = selector.get_support()
    except Exception as exc:
        raise ValueError(
            "A etapa de seleção ainda não foi treinada. Execute fit antes."
        ) from exc

    if len(feature_names) != len(support_mask):
        raise ValueError(
            "A quantidade de feature_names não corresponde ao número de atributos vistos pelo pipeline."
        )

    if not hasattr(selector, "estimator_"):
        raise ValueError(
            "O estimador interno do seletor não foi ajustado. Execute fit antes."
        )

    selector_estimator = selector.estimator_

    if hasattr(selector_estimator, "feature_importances_"):
        # Caminho principal para modelos baseados em árvore.
        full_importances = selector_estimator.feature_importances_
    elif hasattr(selector_estimator, "coef_"):
        # Fallback para estimadores lineares; média do valor absoluto dos coeficientes.
        coef = selector_estimator.coef_
        full_importances = np.mean(np.abs(coef), axis=0)
    else:
        raise TypeError(
            "O estimador do seletor não expõe feature_importances_ nem coef_."
        )

    selected_importances = [
        float(importance)
        for importance, was_selected in zip(full_importances, support_mask)
        if was_selected
    ]
    selected_names = [
        feature_name
        for feature_name, was_selected in zip(feature_names, support_mask)
        if was_selected
    ]

    # Ordena do maior peso para o menor para facilitar análise.
    ranked = sorted(
        zip(selected_names, selected_importances),
        key=lambda item: item[1],
        reverse=True,
    )

    return ranked


def get_feature_importances_df(
    trained_pipeline: Pipeline,
    feature_names: list[str],
) -> pd.DataFrame:
    # Retorna um DataFrame com as features selecionadas e suas importâncias.
    ranked = get_selected_feature_importances(
        trained_pipeline=trained_pipeline,
        feature_names=feature_names,
    )

    return pd.DataFrame(
        ranked,
        columns=["feature", "importance"],
    )