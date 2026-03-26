from __future__ import annotations
from collections import Counter
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
	ConfusionMatrixDisplay,
	accuracy_score,
	confusion_matrix,
	f1_score,
	precision_score,
	recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from models import get_all_models, get_selected_feature_importances
from preprocessing import load_raw_data, prepare_features


def _build_output_dirs(output_root: Path) -> dict[str, Path]:
	tables_dir = output_root / "tables"
	plots_dir = output_root / "plots"

	tables_dir.mkdir(parents=True, exist_ok=True)
	plots_dir.mkdir(parents=True, exist_ok=True)
	discussion_dir = output_root / "discussion"

	# Limpa artefatos anteriores para manter saída objetiva.
	for csv_file in tables_dir.glob("*.csv"):
		csv_file.unlink(missing_ok=True)
	for png_file in plots_dir.glob("*.png"):
		png_file.unlink(missing_ok=True)
	if discussion_dir.exists():
		for md_file in discussion_dir.glob("*.md"):
			md_file.unlink(missing_ok=True)

	return {
		"root": output_root,
		"tables": tables_dir,
		"plots": plots_dir,
	}


def _run_minimal_exploratory_analysis(
	df: pd.DataFrame,
	output_dirs: dict[str, Path],
) -> None:
	resumo_exploratorio = pd.DataFrame(
		[
			{
				"total_registros": int(len(df)),
				"total_atributos": int(df.shape[1]),
				"atributos_numericos": int(df.select_dtypes(include=[np.number]).shape[1]),
				"atributos_categoricos": int(df.select_dtypes(exclude=[np.number]).shape[1]),
				"classe_positiva_pct": float((df["diagnosed_diabetes"] == 1).mean() * 100),
			}
		]
	)
	resumo_exploratorio.to_csv(
		output_dirs["tables"] / "resumo_exploratorio.csv",
		index=False,
	)

	class_counts = df["diagnosed_diabetes"].value_counts().sort_index()
	labels = ["Não Diabetes", "Diabetes"]
	plt.figure(figsize=(6, 4))
	plt.bar(labels, class_counts.values, color=["#4C78A8", "#F58518"])
	plt.title("Distribuição da Classe-Alvo")
	plt.xlabel("Classe")
	plt.ylabel("Quantidade")
	plt.tight_layout()
	plt.savefig(output_dirs["plots"] / "distribuicao_classe_alvo.png", dpi=150)
	plt.close()


def _run_data_quality_assessment(
	df: pd.DataFrame,
	output_dirs: dict[str, Path],
) -> dict[str, float | str]:
	missing_total = int(df.isna().sum().sum())
	duplicate_count = int(df.duplicated().sum())
	positive_class_ratio = float((df["diagnosed_diabetes"] == 1).mean() * 100)

	numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
	max_outlier_pct = 0.0
	feature_with_max_outlier = ""

	for col in numeric_cols:
		q1 = float(df[col].quantile(0.25))
		q3 = float(df[col].quantile(0.75))
		iqr = q3 - q1
		lower_bound = q1 - 1.5 * iqr
		upper_bound = q3 + 1.5 * iqr
		outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
		outlier_count = int(outlier_mask.sum())
		outlier_pct = (outlier_count / len(df)) * 100

		if outlier_pct > max_outlier_pct:
			max_outlier_pct = outlier_pct
			feature_with_max_outlier = col

	quality_summary = {
		"total_registros": float(len(df)),
		"total_ausentes": float(missing_total),
		"total_duplicados": float(duplicate_count),
		"proporcao_classe_positiva_pct": float(positive_class_ratio),
		"atributo_maior_outlier": feature_with_max_outlier,
		"maior_outlier_pct_iqr": float(max_outlier_pct),
	}

	pd.DataFrame([quality_summary]).to_csv(
		output_dirs["tables"] / "resumo_qualidade_dados.csv",
		index=False,
	)

	return quality_summary


def _evaluate_models(
	raw_df: pd.DataFrame,
	n_repeats: int,
	output_dirs: dict[str, Path],
) -> pd.DataFrame:
	models = get_all_models()
	model_stats: dict[str, dict[str, object]] = {}

	for model_name in models.keys():
		model_stats[model_name] = {
			"accuracy": [],
			"precision": [],
			"recall": [],
			"f1_score": [],
			"f1_cv_media": [],
			"f1_cv_desvio": [],
			"cm_sum": np.zeros((2, 2), dtype=int),
			"top3_list": [],
		}

	for repeat_index in range(n_repeats):
		repeat_seed = 42 + repeat_index
		print(f"  Repetição {repeat_index + 1}/{n_repeats} (seed={repeat_seed})")

		X_train, X_test, y_train, y_test, feature_names = prepare_features(
			raw_df,
			random_state=repeat_seed,
		)

		for model_name, pipeline in get_all_models().items():
			cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=repeat_seed)
			cv_scores = cross_val_score(
				pipeline,
				X_train,
				y_train,
				cv=cv,
				scoring="f1",
				n_jobs=None,
			)

			pipeline.fit(X_train, y_train)
			y_pred = pipeline.predict(X_test)
			cm = confusion_matrix(y_test, y_pred)

			feature_importances = get_selected_feature_importances(
				trained_pipeline=pipeline,
				feature_names=feature_names,
			)
			top3_features = ", ".join(
				feature_name
				for feature_name, _ in feature_importances[:3]
			)

			stats = model_stats[model_name]
			stats["accuracy"].append(float(accuracy_score(y_test, y_pred)))
			stats["precision"].append(float(precision_score(y_test, y_pred, zero_division=0)))
			stats["recall"].append(float(recall_score(y_test, y_pred, zero_division=0)))
			stats["f1_score"].append(float(f1_score(y_test, y_pred, zero_division=0)))
			stats["f1_cv_media"].append(float(np.mean(cv_scores)))
			stats["f1_cv_desvio"].append(float(np.std(cv_scores)))
			stats["cm_sum"] += cm
			stats["top3_list"].append(top3_features)

	rows: list[dict[str, float | str]] = []
	for model_name, stats in model_stats.items():
		most_common_top3 = Counter(stats["top3_list"]).most_common(1)[0][0]
		cm_sum = stats["cm_sum"]

		rows.append(
			{
				"modelo": model_name,
				"top3_atributos": most_common_top3,
				"f1_cv_media": float(np.mean(stats["f1_cv_media"])),
				"f1_cv_desvio": float(np.mean(stats["f1_cv_desvio"])),
				"accuracy": float(np.mean(stats["accuracy"])),
				"precision": float(np.mean(stats["precision"])),
				"recall": float(np.mean(stats["recall"])),
				"f1_score": float(np.mean(stats["f1_score"])),
				"verdadeiro_negativo": int(cm_sum[0, 0]),
				"falso_positivo": int(cm_sum[0, 1]),
				"falso_negativo": int(cm_sum[1, 0]),
				"verdadeiro_positivo": int(cm_sum[1, 1]),
			}
		)

		fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
		disp = ConfusionMatrixDisplay(
			confusion_matrix=cm_sum,
			display_labels=["Não Diabetes", "Diabetes"],
		)
		disp.plot(ax=ax_cm, cmap="Blues", colorbar=False)
		ax_cm.set_title(f"Matriz de Confusão Acumulada - {model_name}")
		ax_cm.set_xlabel("Predição")
		ax_cm.set_ylabel("Valor Real")
		fig_cm.tight_layout()
		fig_cm.savefig(output_dirs["plots"] / f"cm_{model_name}.png", dpi=150)
		plt.close(fig_cm)

	comparison_df = pd.DataFrame(rows).sort_values("f1_score", ascending=False)
	comparison_df.to_csv(output_dirs["tables"] / "comparacao_modelos.csv", index=False)

	return comparison_df


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Pipeline simplificado de EDA, qualidade de dados e comparação de modelos.",
	)
	parser.add_argument(
		"--n-samples",
		type=int,
		default=10000,
		help="Quantidade de amostras para o experimento (estratificado).",
	)
	parser.add_argument(
		"--output-dir",
		type=str,
		default="outputs",
		help="Diretório para salvar tabelas e gráficos.",
	)
	parser.add_argument(
		"--n-repeats",
		type=int,
		default=3,
		help="Quantidade de repetições do experimento para média das métricas.",
	)
	args = parser.parse_args()

	project_root = Path(__file__).resolve().parents[1]
	dataset_path = project_root / "data" / "diabetes_dataset.csv"
	output_dirs = _build_output_dirs(project_root / args.output_dir)

	print("\n[1/4] Carregando dados...")
	raw_df = load_raw_data(path=str(dataset_path), n_samples=args.n_samples)

	print("[2/4] Executando EDA mínimo (1 tabela + 1 gráfico)...")
	_run_minimal_exploratory_analysis(df=raw_df, output_dirs=output_dirs)

	print("[3/4] Avaliando qualidade dos dados e pré-processamento...")
	_run_data_quality_assessment(df=raw_df, output_dirs=output_dirs)

	print(
		f"[4/4] Treinando modelos e avaliando métricas (com validação cruzada e {args.n_repeats} repetições)..."
	)
	comparison_df = _evaluate_models(
		raw_df=raw_df,
		n_repeats=args.n_repeats,
		output_dirs=output_dirs,
	)

	best_model = comparison_df.iloc[0]["modelo"]
	best_f1 = comparison_df.iloc[0]["f1_score"]

	print("\nExecução concluída com sucesso.")
	print(f"Melhor modelo por F1-score: {best_model} ({best_f1:.4f})")
	print(f"Resultados salvos em: {output_dirs['root']}")


if __name__ == "__main__":
	main()