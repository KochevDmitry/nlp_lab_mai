import requests
import json

SERVICE_URL = "http://localhost:8000"


def send_prompt(prompt: str, model: str = "qwen2.5:0.5b") -> str:
    response = requests.post(
        f"{SERVICE_URL}/generate",
        json={"prompt": prompt, "model": model},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"]


def check_health() -> bool:
    try:
        r = requests.get(f"{SERVICE_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def run_inference(prompts: list[str]) -> list[dict]:
    results = []
    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] Отправляю запрос...")
        response = send_prompt(prompt)
        results.append({"prompt": prompt, "response": response})
        print(f"  Ответ: {response[:120]}...\n")
    return results


def save_report(results: list[dict], filename: str = "results/report.json") -> None:
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Отчёт сохранён в {filename}")


if __name__ == "__main__":
    if not check_health():
        print("Сервис недоступен")
        exit(1)

    prompts = [
        "Что такое машинное обучение?",
        "Объясни квантовые вычисления простыми словами.",
        "Какова столица Франции?",
        "Как работает фотосинтез?",
        "Напиши короткое стихотворение про осень.",
        "Каковы преимущества занятий спортом?",
        "Что такое искусственный интеллект?",
        "Как дела?",
        "Назови три популярных языка программирования.",
        "В чём смысл жизни?",
    ]

    results = run_inference(prompts)
    save_report(results)
    print("Готово!")