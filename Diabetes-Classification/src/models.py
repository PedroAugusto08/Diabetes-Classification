"""
Este módulo define os modelos e seus passos de pré-processamento
( seleção de atributos e, quando necessário, padronização ).
"""

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
        threshold="median",
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

    return [
        feature_name
        for feature_name, was_selected in zip(feature_names, support_mask)
        if was_selected
    ]