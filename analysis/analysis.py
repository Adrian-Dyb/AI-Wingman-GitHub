#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-wingman — finalna analiza statystyczna
=========================================

Cel:
Odtworzenie finalnych analiz pracy magisterskiej bez pseudoreplikacji.

Wejście:
- judge_ratings.csv
- trials.csv

Główna procedura:
1. Wczytanie ocen sędziów.
2. Agregacja 3 ocen do poziomu trial_id.
3. Agregacja prób do participant_id × assigned_condition.
4. H1, H2a, H2b, H3: test Wilcoxona dla par.
5. H4: osobny test Wilcoxona.
6. Analizy eksploracyjne cech tekstowych.
7. Wilcoxon:
   - dwustronny,
   - asymptotyczny,
   - zero_method="wilcox" (zera pomijane),
   - średnie rangi dla remisów,
   - bez poprawki ciągłości.
8. Holm dla czterech głównych wyników percepcyjnych:
   H1, H2a, H2b, H3.
9. Matched-pairs rank-biserial correlation (r_rb).
10. Raportuje:
   N, N_eff, n_zero, W, p, p_Holm, r_rb, średnie.

Uruchomienie:
    python analysis.py

lub:
    python analysis.py \
        --judge-ratings judge_ratings.csv \
        --trials trials.csv \
        --out-dir analysis_outputs

Wymagania:
    numpy
    scipy
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, wilcoxon


PRIMARY = {
    "H1_quality": "quality",
    "H2a_naturalness": "naturalness",
    "H2b_attractiveness": "attractiveness",
    "H3_continuation": "continuation",
}

H4 = {
    "H4_open_question_present": "open_question_present",
}

EXPLORATORY = {
    "EXP_word_count": "word_count",
    "EXP_char_count": "char_count",
    "EXP_emoji_count": "emoji_count",
    "EXP_question_present": "question_present",
}

RATING_COLUMNS = {
    "quality": "quality_rating",
    "naturalness": "naturalness_rating",
    "attractiveness": "attractiveness_rating",
    "continuation": "continuation_rating",
}

CONDITIONS = ("manual", "ai_assisted")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_bool(value: str) -> int:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return 1
    if text in {"false", "0", "no"}:
        return 0
    raise ValueError(f"Nie można zinterpretować jako bool: {value!r}")


def mean(values) -> float:
    return float(np.mean(np.asarray(list(values), dtype=float)))


def holm_adjust(p_values: list[float]) -> list[float]:
    """
    Korekta Holma bez użycia zewnętrznego pakietu statystycznego.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)

    running_max = 0.0
    for rank, idx in enumerate(order):
        candidate = (m - rank) * p[idx]
        running_max = max(running_max, candidate)
        adjusted[idx] = min(1.0, running_max)

    return adjusted.tolist()


def rank_biserial_paired(manual, ai) -> tuple[float, int, int]:
    """
    Matched-pairs rank-biserial correlation.

    Kierunek:
        dodatni  -> AI-assisted wyżej
        ujemny   -> manual wyżej

    Zera są pomijane.
    Remisy |difference| otrzymują średnie rangi.
    """
    manual = np.asarray(manual, dtype=float)
    ai = np.asarray(ai, dtype=float)
    diff = ai - manual

    zero_mask = diff == 0.0
    nonzero = diff[~zero_mask]

    n_zero = int(zero_mask.sum())
    n_eff = int(nonzero.size)

    if n_eff == 0:
        return 0.0, n_eff, n_zero

    ranks = rankdata(np.abs(nonzero), method="average")
    r_plus = float(ranks[nonzero > 0].sum())
    r_minus = float(ranks[nonzero < 0].sum())

    denominator = r_plus + r_minus
    r_rb = 0.0 if denominator == 0 else (r_plus - r_minus) / denominator

    return float(r_rb), n_eff, n_zero


def wilcoxon_paired(manual, ai) -> dict:
    """
    Finalna specyfikacja testu:
    - paired,
    - two-sided,
    - asymptotic approximation,
    - zero_method='wilcox': zerowe różnice są usuwane,
    - tied absolute differences: average ranks (standard Wilcoxona/SciPy),
    - correction=False: bez poprawki ciągłości.
    """
    manual = np.asarray(manual, dtype=float)
    ai = np.asarray(ai, dtype=float)

    if manual.shape != ai.shape:
        raise ValueError("Wektory manual i AI mają różne długości.")

    diff = ai - manual
    n = int(len(diff))
    n_zero = int(np.sum(diff == 0.0))
    n_eff = n - n_zero

    if n_eff == 0:
        return {
            "N": n,
            "N_eff": 0,
            "n_zero": n_zero,
            "W": 0.0,
            "p": 1.0,
            "r_rb": 0.0,
            "manual_mean": float(np.mean(manual)),
            "ai_mean": float(np.mean(ai)),
            "delta_ai_minus_manual": 0.0,
        }

    result = wilcoxon(
        manual,
        ai,
        alternative="two-sided",
        zero_method="wilcox",
        method="approx",
        correction=False,
    )

    r_rb, n_eff_rb, n_zero_rb = rank_biserial_paired(manual, ai)

    if n_eff_rb != n_eff or n_zero_rb != n_zero:
        raise RuntimeError("Niespójność N_eff / n_zero.")

    return {
        "N": n,
        "N_eff": n_eff,
        "n_zero": n_zero,
        "W": float(result.statistic),
        "p": float(result.pvalue),
        "r_rb": float(r_rb),
        "manual_mean": float(np.mean(manual)),
        "ai_mean": float(np.mean(ai)),
        "delta_ai_minus_manual": float(np.mean(ai - manual)),
    }


def validate_inputs(ratings, trials):
    required_rating = {
        "judge_id",
        "trial_id",
        *RATING_COLUMNS.values(),
    }
    required_trial = {
        "trial_id",
        "participant_id",
        "assigned_condition",
        "word_count",
        "char_count",
        "emoji_count",
        "question_present",
        "open_question_present",
        "sentiment",
    }

    if not ratings:
        raise ValueError("judge_ratings.csv jest pusty.")
    if not trials:
        raise ValueError("trials.csv jest pusty.")

    missing_r = required_rating - set(ratings[0])
    missing_t = required_trial - set(trials[0])

    if missing_r:
        raise ValueError(
            "Brak kolumn w judge_ratings.csv: "
            + ", ".join(sorted(missing_r))
        )
    if missing_t:
        raise ValueError(
            "Brak kolumn w trials.csv: "
            + ", ".join(sorted(missing_t))
        )

    trial_ids = [t["trial_id"] for t in trials]
    if len(trial_ids) != len(set(trial_ids)):
        raise ValueError("trials.csv zawiera duplikaty trial_id.")

    by_trial = Counter(r["trial_id"] for r in ratings)
    invalid = {tid: n for tid, n in by_trial.items() if n != 3}
    if invalid:
        raise ValueError(
            "Każdy trial_id powinien mieć dokładnie 3 oceny. "
            f"Nieprawidłowe: {invalid}"
        )

    rating_trial_ids = set(by_trial)
    trial_set = set(trial_ids)

    missing_trials = rating_trial_ids - trial_set
    missing_ratings = trial_set - rating_trial_ids

    if missing_trials:
        raise ValueError(
            "Oceny odnoszą się do nieznanych trial_id: "
            + ", ".join(sorted(missing_trials))
        )

    if missing_ratings:
        raise ValueError(
            "Brak ocen sędziów dla trial_id: "
            + ", ".join(sorted(missing_ratings))
        )


def aggregate_to_trial(ratings, trials):
    """
    3 sędziów -> jedna średnia na trial_id dla każdego wymiaru.
    """
    ratings_by_trial = defaultdict(list)
    for row in ratings:
        ratings_by_trial[row["trial_id"]].append(row)

    trial_by_id = {row["trial_id"]: row for row in trials}

    output = []

    for trial_id in sorted(trial_by_id):
        trial = trial_by_id[trial_id]
        judge_rows = ratings_by_trial[trial_id]

        out = {
            "trial_id": trial_id,
            "participant_id": trial["participant_id"],
            "assigned_condition": trial["assigned_condition"],
            "scenario_id": trial.get("scenario_id", ""),
            "word_count": float(trial["word_count"]),
            "char_count": float(trial["char_count"]),
            "emoji_count": float(trial["emoji_count"]),
            "question_present": parse_bool(trial["question_present"]),
            "open_question_present": parse_bool(
                trial["open_question_present"]
            ),
            "sentiment": trial["sentiment"],
        }

        for metric, col in RATING_COLUMNS.items():
            out[metric] = mean(float(r[col]) for r in judge_rows)

        output.append(out)

    return output


def aggregate_to_participant_condition(trial_level):
    """
    trial_id -> participant_id × assigned_condition.

    Każdy uczestnik ma własną średnią w warunku manualnym
    i własną średnią w warunku AI-assisted.
    """
    by_participant_condition = defaultdict(list)

    for row in trial_level:
        key = (row["participant_id"], row["assigned_condition"])
        by_participant_condition[key].append(row)

    participant_ids = sorted({r["participant_id"] for r in trial_level})
    output = []

    all_metrics = [
        *RATING_COLUMNS.keys(),
        "word_count",
        "char_count",
        "emoji_count",
        "question_present",
        "open_question_present",
    ]

    for participant_id in participant_ids:
        out = {"participant_id": participant_id}

        for condition in CONDITIONS:
            rows = by_participant_condition.get(
                (participant_id, condition), []
            )

            if not rows:
                raise ValueError(
                    f"Uczestnik {participant_id} nie ma danych "
                    f"dla warunku {condition}."
                )

            out[f"n_trials_{condition}"] = len(rows)

            for metric in all_metrics:
                out[f"{metric}_{condition}"] = mean(
                    float(r[metric]) for r in rows
                )

        output.append(out)

    return output


def vectors(participant_level, metric):
    manual = [
        float(row[f"{metric}_manual"])
        for row in participant_level
    ]
    ai = [
        float(row[f"{metric}_ai_assisted"])
        for row in participant_level
    ]
    return manual, ai


def run_inferential_tests(participant_level):
    rows = []

    # H1, H2a, H2b, H3
    primary_rows = []
    for hypothesis, metric in PRIMARY.items():
        manual, ai = vectors(participant_level, metric)
        result = wilcoxon_paired(manual, ai)

        row = {
            "analysis": "primary",
            "hypothesis": hypothesis,
            "metric": metric,
            **result,
            "p_Holm": None,
        }
        primary_rows.append(row)

    adjusted = holm_adjust([row["p"] for row in primary_rows])
    for row, p_holm in zip(primary_rows, adjusted):
        row["p_Holm"] = float(p_holm)

    rows.extend(primary_rows)

    # H4 — osobna rodzina
    for hypothesis, metric in H4.items():
        manual, ai = vectors(participant_level, metric)
        result = wilcoxon_paired(manual, ai)

        rows.append(
            {
                "analysis": "H4_separate_family",
                "hypothesis": hypothesis,
                "metric": metric,
                **result,
                "p_Holm": None,
            }
        )

    # Eksploracyjne
    for label, metric in EXPLORATORY.items():
        manual, ai = vectors(participant_level, metric)
        result = wilcoxon_paired(manual, ai)

        rows.append(
            {
                "analysis": "exploratory",
                "hypothesis": label,
                "metric": metric,
                **result,
                "p_Holm": None,
            }
        )

    return rows


def sentiment_descriptives(trial_level):
    rows = []
    categories = sorted({r["sentiment"] for r in trial_level})

    for condition in CONDITIONS:
        subset = [
            r for r in trial_level
            if r["assigned_condition"] == condition
        ]
        counts = Counter(r["sentiment"] for r in subset)
        total = len(subset)

        for category in categories:
            rows.append(
                {
                    "assigned_condition": condition,
                    "sentiment": category,
                    "n": counts[category],
                    "proportion": counts[category] / total if total else 0,
                }
            )

    return rows


def print_results(rows):
    print()
    print("=" * 118)
    print("AI-WINGMAN — FINALNE WYNIKI")
    print("=" * 118)
    print(
        f"{'Analiza':<20} {'Metryka':<22} "
        f"{'N':>3} {'N_eff':>6} {'zera':>5} "
        f"{'M_manual':>10} {'M_AI':>10} "
        f"{'W':>8} {'p':>11} {'p_Holm':>11} {'r_rb':>9}"
    )
    print("-" * 118)

    for row in rows:
        p_holm = (
            "-"
            if row["p_Holm"] is None
            else f"{row['p_Holm']:.6f}"
        )

        print(
            f"{row['hypothesis']:<20} "
            f"{row['metric']:<22} "
            f"{row['N']:>3} "
            f"{row['N_eff']:>6} "
            f"{row['n_zero']:>5} "
            f"{row['manual_mean']:>10.6f} "
            f"{row['ai_mean']:>10.6f} "
            f"{row['W']:>8.3f} "
            f"{row['p']:>11.6f} "
            f"{p_holm:>11} "
            f"{row['r_rb']:>9.6f}"
        )

    print("=" * 118)
    print(
        "Ustawienia Wilcoxona: two-sided, method='approx', "
        "zero_method='wilcox', correction=False."
    )
    print(
        "Remisy wartości bezwzględnych różnic otrzymują średnie rangi. "
        "r_rb > 0 oznacza wyższy wynik AI-assisted."
    )
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Finalna analiza badania AI-wingman."
    )
    parser.add_argument(
        "--judge-ratings",
        type=Path,
        default=Path("judge_ratings.csv"),
        help="Ujednolicony plik ocen trzech sędziów.",
    )
    parser.add_argument(
        "--trials",
        type=Path,
        default=Path("trials.csv"),
        help="Finalny plik prób eksperymentalnych.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("analysis_outputs"),
        help="Katalog wynikowy.",
    )
    args = parser.parse_args()

    ratings = read_csv(args.judge_ratings)
    trials = read_csv(args.trials)

    # Jeżeli pliki zawierają fazę badania, analizujemy wyłącznie main.
    if ratings and "study_phase" in ratings[0]:
        ratings = [
            r for r in ratings
            if r.get("study_phase", "main") == "main"
        ]

    if trials and "study_phase" in trials[0]:
        trials = [
            r for r in trials
            if r.get("study_phase", "main") == "main"
        ]

    validate_inputs(ratings, trials)

    trial_level = aggregate_to_trial(ratings, trials)
    participant_level = aggregate_to_participant_condition(trial_level)
    inferential_results = run_inferential_tests(participant_level)
    sentiment_rows = sentiment_descriptives(trial_level)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(
        args.out_dir / "trial_level_aggregates.csv",
        trial_level,
    )
    write_csv(
        args.out_dir / "participant_condition_aggregates.csv",
        participant_level,
    )

    result_fields = [
        "analysis",
        "hypothesis",
        "metric",
        "N",
        "N_eff",
        "n_zero",
        "manual_mean",
        "ai_mean",
        "delta_ai_minus_manual",
        "W",
        "p",
        "p_Holm",
        "r_rb",
    ]

    write_csv(
        args.out_dir / "final_statistical_results.csv",
        inferential_results,
        result_fields,
    )
    write_csv(
        args.out_dir / "sentiment_descriptives.csv",
        sentiment_rows,
    )

    print_results(inferential_results)

    print("Kontrola danych:")
    print(f"- oceny sędziów: {len(ratings)}")
    print(f"- triale: {len(trial_level)}")
    print(f"- uczestnicy: {len(participant_level)}")
    print(f"- wyniki zapisano do: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
