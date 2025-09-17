import pandas as pd

class SystemsOverview:
    def __init__(self):
        self.results = []

    def add_system_results(self, system_name, metrics_original, metrics_dedup):
        self.results.append({
            "System": system_name,
            "Accuracy Origin": metrics_original["accuracy"],
            "Accuracy Deduplication": metrics_dedup["accuracy"],
            "Precision Origin": metrics_original["precision_macro"],   # or precision_weighted
            "Precision Deduplication": metrics_dedup["precision_macro"],
            "Recall Origin": metrics_original["recall_macro"],
            "Recall Deduplication": metrics_dedup["recall_macro"]
        })

    def print_results_table(self):
        df_results = pd.DataFrame(self.results)
        print(df_results.to_string(index=False))

def get_stats(report):
        return {
            "accuracy": report["accuracy"],
            "precision_macro": report["macro avg"]["precision"],
            "precision_weighted": report["weighted avg"]["precision"],
            "recall_macro": report["macro avg"]["recall"],
            "recall_weighted": report["weighted avg"]["recall"],
            "f1_macro": report["macro avg"]["f1-score"],
            "f1_weighted": report["weighted avg"]["f1-score"]
        }