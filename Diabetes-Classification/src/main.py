from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import (
	ConfusionMatrixDisplay,
	RocCurveDisplay,
	accuracy_score,
	recall_score,
	roc_auc_score,
)

from models import (
	get_all_models,
	get_selected_feature_importances,
)
from preprocessing import load_and_prepare_data


def _select_model_name(available_models: list[str]) -> str:
	# Permite ao usuário escolher um modelo por número ou por nome.
	print("\nModelos disponíveis:")
	for index, model_name in enumerate(available_models, start=1):
		print(f"{index}. {model_name}")

	user_choice = input("\nEscolha o modelo (número ou nome): ").strip()

	# Escolha por índice (ex.: 1, 2, 3)
	if user_choice.isdigit():
		selected_index = int(user_choice) - 1
		if 0 <= selected_index < len(available_models):
			return available_models[selected_index]

	# Escolha por nome, sem diferenciar maiúsculas/minúsculas.
	normalized_map = {name.lower(): name for name in available_models}
	if user_choice.lower() in normalized_map:
		return normalized_map[user_choice.lower()]

	raise ValueError("Modelo inválido. Execute novamente e escolha uma opção válida.")


def main() -> None:
	# Monta o caminho absoluto do dataset com base na pasta do projeto.
	project_root = Path(__file__).resolve().parents[1]
	dataset_path = project_root / "data" / "diabetes_dataset.csv"

	n_samples_input = input("Quantidade de amostras (Enter para 10000): ").strip()
	n_samples = int(n_samples_input) if n_samples_input else 10000

	X_train, X_test, y_train, y_test, feature_names = load_and_prepare_data(
		path=str(dataset_path),
		n_samples=n_samples,
	)

	models = get_all_models()
	model_names = list(models.keys())
	selected_model_name = _select_model_name(model_names)
	pipeline = models[selected_model_name]

	print(f"\nTreinando modelo: {selected_model_name}")
	pipeline.fit(X_train, y_train)

	y_pred = pipeline.predict(X_test)
	y_proba = pipeline.predict_proba(X_test)[:, 1]

	accuracy = accuracy_score(y_test, y_pred)
	recall = recall_score(y_test, y_pred)

	try:
		roc_auc = roc_auc_score(y_test, y_proba)
	except ValueError:
		roc_auc = float("nan")

	print("\n=== Métricas de Avaliação ===")
	print(f"Modelo: {selected_model_name}")
	print(f"Accuracy: {accuracy:.4f}")
	print(f"Recall: {recall:.4f}")
	print(f"ROC-AUC: {roc_auc:.4f}")

	selected_feature_importances = get_selected_feature_importances(
		pipeline,
		feature_names,
	)

	print(
		f"\n=== Features Selecionadas e Importância ({len(selected_feature_importances)}) ==="
	)
	for feature_name, importance in selected_feature_importances:
		print(f"- {feature_name}: {importance:.6f}")

	# Figura 1: Curva ROC.
	fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
	RocCurveDisplay.from_predictions(
		y_test,
		y_proba,
		name=selected_model_name,
		ax=ax_roc,
	)
	ax_roc.plot(
		[0, 1],
		[0, 1],
		linestyle="--",
		color="gray",
		label="Baseline",
	)
	ax_roc.legend(loc="lower right")
	ax_roc.set_title("Curva ROC")
	fig_roc.tight_layout()
	plt.show()

	# Figura 2: Matriz de confusão.
	fig_cm, ax_cm = plt.subplots(figsize=(7, 6))
	ConfusionMatrixDisplay.from_predictions(
		y_test,
		y_pred,
		ax=ax_cm,
		cmap="Blues",
	)
	ax_cm.set_title("Matriz de Confusão")
	fig_cm.tight_layout()
	plt.show()


if __name__ == "__main__":
	main()