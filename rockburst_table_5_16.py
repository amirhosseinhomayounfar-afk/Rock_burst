"""Build the thesis-style Table 5-16 for the rock-burst database.

The script is designed for Google Colab and also runs with the Python standard
library only. It reads the uploaded Excel file, fits five multinomial logistic
regression (MLR) models on training data, predicts the holdout rows that were
not used for fitting each relation, and exports the count of correct predictions
by observed rock-burst intensity class.

Default input file: 1.main.xlsx
Default output files: table_5_16_counts.md and table_5_16_counts.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
from xml.etree import ElementTree as ET

# Edit these names if your Excel headers are changed.
DEFAULT_FEATURES = ["Wet", "sc/st", "sq/sc"]
DEFAULT_TARGET = "Actual intensity level"

# The reference table has two relations tested on a 102/153 holdout and three
# relations tested on a 51/153 holdout. Keeping the same ratios makes the table
# automatically scale to the current database size.
DEFAULT_RELATIONS = [
    ("(5-41)", 102 / 153, 1401),
    ("(5-42)", 102 / 153, 1402),
    ("(5-43)", 51 / 153, 1403),
    ("(5-44)", 51 / 153, 1404),
    ("(5-45)", 51 / 153, 1405),
]

CLASS_LABELS = {
    1: "عدم وقوع",
    2: "وقوع ضعیف",
    3: "وقوع متوسط",
    4: "وقوع قوی",
}

PERSIAN_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")


def fa_num(value: object) -> str:
    """Convert Western digits in a value to Persian digits for display."""
    return str(value).translate(PERSIAN_DIGITS)


def excel_column_index(cell_ref: str) -> int:
    letters = "".join(re.findall(r"[A-Z]+", cell_ref))
    index = 0
    for char in letters:
        index = index * 26 + ord(char) - ord("A") + 1
    return index - 1


def read_xlsx_first_sheet(path: Path) -> List[Dict[str, str]]:
    """Read the first worksheet of an .xlsx file using only stdlib modules."""
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        shared_strings: List[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("m:si", ns):
                text = "".join((node.text or "") for node in item.findall(".//m:t", ns))
                shared_strings.append(text)

        sheet_root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        rows: List[List[str]] = []
        for row in sheet_root.findall(".//m:sheetData/m:row", ns):
            values: Dict[int, str] = {}
            for cell in row.findall("m:c", ns):
                ref = cell.attrib.get("r", "")
                value_node = cell.find("m:v", ns)
                value = ""
                if value_node is not None and value_node.text is not None:
                    value = value_node.text
                    if cell.attrib.get("t") == "s":
                        value = shared_strings[int(value)]
                values[excel_column_index(ref)] = value
            if values:
                rows.append([values.get(i, "") for i in range(max(values) + 1)])

    if not rows:
        raise ValueError(f"No rows were found in {path}")

    headers = [header.strip() for header in rows[0]]
    records = []
    for row in rows[1:]:
        records.append({header: (row[i].strip() if i < len(row) else "") for i, header in enumerate(headers)})
    return records


def to_float(value: str) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() == "N/A":
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def prepare_dataset(
    records: Iterable[Dict[str, str]], features: Sequence[str], target: str
) -> Tuple[List[List[float]], List[int]]:
    x_rows: List[List[float]] = []
    y_rows: List[int] = []
    for record in records:
        x = [to_float(record.get(feature, "")) for feature in features]
        y_value = to_float(record.get(target, ""))
        if any(value is None for value in x) or y_value is None:
            continue
        y = int(round(y_value))
        if y not in CLASS_LABELS:
            continue
        x_rows.append([float(value) for value in x if value is not None])
        y_rows.append(y)
    if not x_rows:
        raise ValueError("No complete modelling rows were found. Check feature and target column names.")
    return x_rows, y_rows


def standardize(train_x: Sequence[Sequence[float]], all_x: Sequence[Sequence[float]]) -> Tuple[List[List[float]], List[List[float]]]:
    means = [sum(row[j] for row in train_x) / len(train_x) for j in range(len(train_x[0]))]
    scales = []
    for j, mean in enumerate(means):
        variance = sum((row[j] - mean) ** 2 for row in train_x) / max(1, len(train_x) - 1)
        scales.append(math.sqrt(variance) or 1.0)

    def transform(rows: Sequence[Sequence[float]]) -> List[List[float]]:
        return [[(row[j] - means[j]) / scales[j] for j in range(len(means))] for row in rows]

    return transform(train_x), transform(all_x)


def softmax(scores: Sequence[float]) -> List[float]:
    max_score = max(scores)
    exps = [math.exp(score - max_score) for score in scores]
    total = sum(exps)
    return [value / total for value in exps]


def fit_multinomial_logistic_regression(
    x_train: Sequence[Sequence[float]],
    y_train: Sequence[int],
    classes: Sequence[int],
    epochs: int = 2500,
    learning_rate: float = 0.08,
    l2: float = 0.001,
) -> List[List[float]]:
    """Fit a small softmax regression model with batch gradient descent."""
    n_features = len(x_train[0]) + 1  # + intercept
    weights = [[0.0] * n_features for _ in classes]
    class_to_index = {label: index for index, label in enumerate(classes)}
    design = [[1.0] + list(row) for row in x_train]

    for _ in range(epochs):
        gradients = [[0.0] * n_features for _ in classes]
        for x, y in zip(design, y_train):
            probabilities = softmax([sum(w * value for w, value in zip(row_w, x)) for row_w in weights])
            for class_index in range(len(classes)):
                target_value = 1.0 if class_index == class_to_index[y] else 0.0
                error = probabilities[class_index] - target_value
                for feature_index, value in enumerate(x):
                    gradients[class_index][feature_index] += error * value
        row_count = len(design)
        for class_index in range(len(classes)):
            for feature_index in range(n_features):
                penalty = 0.0 if feature_index == 0 else l2 * weights[class_index][feature_index]
                weights[class_index][feature_index] -= learning_rate * (gradients[class_index][feature_index] / row_count + penalty)
    return weights


def predict(weights: Sequence[Sequence[float]], x_rows: Sequence[Sequence[float]], classes: Sequence[int]) -> List[int]:
    predictions = []
    for row in x_rows:
        x = [1.0] + list(row)
        scores = [sum(weight * value for weight, value in zip(class_weights, x)) for class_weights in weights]
        predictions.append(classes[max(range(len(scores)), key=lambda index: scores[index])])
    return predictions


def stratified_holdout_indices(y_rows: Sequence[int], fraction: float, seed: int) -> List[int]:
    rng = random.Random(seed)
    by_class: Dict[int, List[int]] = defaultdict(list)
    for index, label in enumerate(y_rows):
        by_class[label].append(index)

    selected: List[int] = []
    for label, indices in sorted(by_class.items()):
        shuffled = indices[:]
        rng.shuffle(shuffled)
        count = max(1, int(round(len(indices) * fraction)))
        selected.extend(shuffled[:count])
    selected.sort()
    return selected


@dataclass
class RelationResult:
    name: str
    holdout_size: int
    population_counts: Counter
    correct_counts: Counter


def evaluate_relations(x_rows: List[List[float]], y_rows: List[int], relation_specs=DEFAULT_RELATIONS) -> List[RelationResult]:
    classes = sorted(CLASS_LABELS)
    results = []
    all_indices = set(range(len(y_rows)))
    for name, fraction, seed in relation_specs:
        holdout = stratified_holdout_indices(y_rows, fraction, seed)
        train = sorted(all_indices - set(holdout))
        train_x_raw = [x_rows[index] for index in train]
        holdout_x_raw = [x_rows[index] for index in holdout]
        train_x, combined_x = standardize(train_x_raw, train_x_raw + holdout_x_raw)
        holdout_x = combined_x[len(train_x_raw) :]
        train_y = [y_rows[index] for index in train]
        holdout_y = [y_rows[index] for index in holdout]

        weights = fit_multinomial_logistic_regression(train_x, train_y, classes)
        predictions = predict(weights, holdout_x, classes)
        correct = Counter()
        for observed, predicted in zip(holdout_y, predictions):
            if observed == predicted:
                correct[observed] += 1
        results.append(
            RelationResult(
                name=name,
                holdout_size=len(holdout),
                population_counts=Counter(holdout_y),
                correct_counts=correct,
            )
        )
    return results


def build_table(results: Sequence[RelationResult]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for holdout_size in sorted({result.holdout_size for result in results}):
        group_results = [result for result in results if result.holdout_size == holdout_size]
        population = Counter()
        for result in group_results:
            population = result.population_counts
            break
        for class_id in sorted(CLASS_LABELS):
            row: Dict[str, object] = {
                "داده‌های اعتبارسنجی": f"{fa_num(holdout_size)} سری داده",
                "جامعه آماری": CLASS_LABELS[class_id],
                "تعداد داده‌های جامعه": population[class_id],
            }
            for result in results:
                row[result.name] = result.correct_counts[class_id] if result.holdout_size == holdout_size else "—"
            rows.append(row)
        total_row: Dict[str, object] = {
            "داده‌های اعتبارسنجی": f"{fa_num(holdout_size)} سری داده",
            "جامعه آماری": "مجموع",
            "تعداد داده‌های جامعه": sum(population.values()),
        }
        for result in results:
            total_row[result.name] = sum(result.correct_counts.values()) if result.holdout_size == holdout_size else "—"
        rows.append(total_row)
    return rows


def write_csv(rows: Sequence[Dict[str, object]], path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: Sequence[Dict[str, object]], path: Path) -> None:
    headers = list(rows[0])
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fa_num(row[header]) for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Table 5-16 style correct-prediction counts for the rock-burst database.")
    parser.add_argument("--excel", default="1.main.xlsx", help="Path to the Excel database.")
    parser.add_argument("--features", nargs="+", default=DEFAULT_FEATURES, help="Feature columns used in MLR.")
    parser.add_argument("--target", default=DEFAULT_TARGET, help="Observed intensity-level column.")
    parser.add_argument("--output-prefix", default="table_5_16_counts", help="Prefix for .md and .csv outputs.")
    args = parser.parse_args()

    records = read_xlsx_first_sheet(Path(args.excel))
    x_rows, y_rows = prepare_dataset(records, args.features, args.target)
    results = evaluate_relations(x_rows, y_rows)
    rows = build_table(results)

    prefix = Path(args.output_prefix)
    write_markdown(rows, prefix.with_suffix(".md"))
    write_csv(rows, prefix.with_suffix(".csv"))

    print(f"Complete modelling rows: {len(y_rows)}")
    print(f"Class counts: {dict(sorted(Counter(y_rows).items()))}")
    print(f"Features: {', '.join(args.features)}")
    print(f"Saved: {prefix.with_suffix('.md')} and {prefix.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
