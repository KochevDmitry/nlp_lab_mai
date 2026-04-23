import json
import time
import requests
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

SERVICE_URL = "http://localhost:8000"
MODEL = "qwen2.5:0.5b"
DATASET_PATH = "data/spam.csv"
N_SAMPLES = 100  # кол-во сэмплов для оценки

PROMPTS = {
    "zero_shot": """You are a spam detection assistant.
Classify the SMS message as spam or not spam.

Respond ONLY with a valid JSON object, no markdown, no extra text:
{"reasoning": "<your reasoning>", "verdict": <0 or 1>}

verdict 1 = spam, verdict 0 = ham (not spam).""",

    "cot": """You are a spam detection assistant.
Classify the SMS message as spam or not spam.

Think step by step:
1. Identify suspicious keywords or patterns (urgency, prizes, links, money).
2. Assess the tone and intent of the message.
3. Make a final decision.

Respond ONLY with a valid JSON object, no markdown, no extra text:
{"reasoning": "<step-by-step reasoning>", "verdict": <0 or 1>}

verdict 1 = spam, verdict 0 = ham (not spam).""",

    "few_shot": """You are a spam detection assistant.
Classify the SMS message as spam or not spam.

Examples:
Message: "FREE entry in 2 a wkly comp to win FA Cup final tkts! Text FA to 87121"
{"reasoning": "Contains prize offer, urgency, and short code — typical spam pattern.", "verdict": 1}

Message: "Ok lar... Joking wif u oni..."
{"reasoning": "Casual conversational message between friends, no promotional content.", "verdict": 0}

Message: "WINNER! You have been selected to receive a £900 prize. Call 09061701461 now."
{"reasoning": "Prize claim with phone number and urgency — clear spam.", "verdict": 1}

Message: "Nah I don't think he goes to usf, he lives around here though"
{"reasoning": "Informal personal conversation, no spam indicators.", "verdict": 0}

Respond ONLY with a valid JSON object, no markdown, no extra text:
{"reasoning": "<your reasoning>", "verdict": <0 or 1>}

verdict 1 = spam, verdict 0 = ham (not spam).""",

    "cot_few_shot": """You are a spam detection assistant.
Classify the SMS message as spam or not spam.

Examples of step-by-step reasoning:
Message: "FREE entry in 2 a wkly comp to win FA Cup final tkts! Text FA to 87121"
{"reasoning": "1. Keywords: FREE, win, prize — strong spam signals. 2. Contains short code to text — typical spam call to action. 3. No personal context. Decision: spam.", "verdict": 1}

Message: "Ok lar... Joking wif u oni..."
{"reasoning": "1. No suspicious keywords. 2. Informal slang, casual tone. 3. Clearly a personal message. Decision: not spam.", "verdict": 0}

Message: "WINNER! You have been selected to receive a £900 prize. Call 09061701461 now."
{"reasoning": "1. WINNER, prize, phone number — all red flags. 2. Urgent call to action. 3. No personal context. Decision: spam.", "verdict": 1}

Now think step by step for the new message:
1. Identify suspicious keywords or patterns.
2. Assess tone and intent.
3. Make a final decision.

Respond ONLY with a valid JSON object, no markdown, no extra text:
{"reasoning": "<step-by-step reasoning>", "verdict": <0 or 1>}

verdict 1 = spam, verdict 0 = ham (not spam).""",
}


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------

def load_dataset(path: str, n_samples: int) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1", usecols=[0, 1], header=0)
    df.columns = ["label_str", "text"]
    df["label"] = df["label_str"].map({"ham": 0, "spam": 1})
    df = df.dropna(subset=["label", "text"])

    spam_df = df[df["label"] == 1].sample(n=int(n_samples * 0.2), random_state=42)
    ham_df = df[df["label"] == 0].sample(n=int(n_samples * 0.8), random_state=42)
    sample = pd.concat([spam_df, ham_df]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Датасет загружен: {len(sample)} сэмплов "
          f"(ham={len(ham_df)}, spam={len(spam_df)})")
    return sample


def query_llm(system_prompt: str, message: str) -> dict:
    full_prompt = f"{system_prompt}\n\nMessage: \"{message}\""
    try:
        response = requests.post(
            f"{SERVICE_URL}/generate",
            json={"prompt": full_prompt, "model": MODEL},
            timeout=120,
        )
        response.raise_for_status()
        raw = response.json()["response"].strip()

        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("JSON not found in response")
        parsed = json.loads(raw[start:end])
        parsed["verdict"] = int(parsed.get("verdict", -1))
        return parsed

    except Exception as e:
        return {"reasoning": f"parse error: {e}", "verdict": -1}


def evaluate_technique(name: str, system_prompt: str, df: pd.DataFrame) -> dict:
    y_true, y_pred = [], []
    errors = 0

    print(f"\n{'=' * 50}")
    print(f"Техника: {name} ({len(df)} сэмплов)")
    print("=" * 50)

    for i, row in df.iterrows():
        result = query_llm(system_prompt, row["text"])
        verdict = result["verdict"]

        if verdict == -1:
            errors += 1
            verdict = 0 

        y_true.append(int(row["label"]))
        y_pred.append(verdict)

        if (i + 1) % 10 == 0:
            print(f"  Обработано: {i + 1}/{len(df)}")

        time.sleep(0.1) 

    metrics = {
        "technique": name,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "parse_errors": errors,
        "n_samples": len(df),
    }
    return metrics


def print_report(all_metrics: list[dict]) -> None:
    print("ОТЧЁТ")
    header = f"{'Техника':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}"
    print(header)
    print("-" * 65)
    for m in all_metrics:
        print(
            f"{m['technique']:<20} "
            f"{m['accuracy']:>10.4f} "
            f"{m['precision']:>10.4f} "
            f"{m['recall']:>10.4f} "
            f"{m['f1']:>10.4f}"
        )
    print("=" * 65)

    best = max(all_metrics, key=lambda x: x["f1"])
    print(f"\nЛучшая техника по F1: {best['technique']} (F1={best['f1']})")


def save_report(all_metrics: list[dict], path: str = "results/research_report.json") -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    print(f"\nОтчёт сохранён в {path}")


if __name__ == "__main__":
    try:
        r = requests.get(f"{SERVICE_URL}/health", timeout=5)
        r.raise_for_status()
        print("LLM-сервис доступен.")
    except Exception as e:
        print(f"Сервис недоступен: {e}")
        print("Убедитесь, что контейнер запущен: docker compose up")
        exit(1)

    df = load_dataset(DATASET_PATH, N_SAMPLES)

    all_metrics = []
    for technique_name, system_prompt in PROMPTS.items():
        metrics = evaluate_technique(technique_name, system_prompt, df)
        all_metrics.append(metrics)
        print(f"  → F1: {metrics['f1']}, Accuracy: {metrics['accuracy']}, "
              f"Parse errors: {metrics['parse_errors']}")

    print_report(all_metrics)
    save_report(all_metrics)