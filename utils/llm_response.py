
from pathlib import Path
from llama_cpp import Llama

PROJECT_DIR = Path(__file__).resolve().parent.parent

BASE_MODEL_PATH = (
    PROJECT_DIR / "models" / "Qwen3-4B-Q4_K_M.gguf"
)

ADAPTER_PATH = (
    PROJECT_DIR / "models" / "plant_disease_adapter.gguf"
)

_llm = None


def load_llm():
    global _llm

    if _llm is not None:
        return _llm

    if not BASE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Base model not found: {BASE_MODEL_PATH}"
        )

    if not ADAPTER_PATH.exists():
        raise FileNotFoundError(
            f"LoRA adapter not found: {ADAPTER_PATH}"
        )

    _llm = Llama(
        model_path=str(BASE_MODEL_PATH),
        lora_path=str(ADAPTER_PATH),
        n_ctx=4096,
        n_threads=6,
        n_gpu_layers=0,
        verbose=False
    )

    return _llm



def generate_disease_explanation(disease_name, knowledge_card):
    llm = load_llm()

    system_prompt = """You are a plant disease knowledge assistant.
Answer using reliable plant pathology information.
Use the provided disease information as your factual reference.
Do not claim to have inspected an image.
Do not invent symptoms, causes, or treatments.
State when information is unavailable.
Use clear, technically accurate language.
Avoid repeating information.

Return exactly these six sections:

1. Disease Overview
2. Cause and Pathogen
3. Symptoms
4. Favorable Conditions
5. Spread
6. Management and Prevention

Write complete sentences under each heading.
Keep the answer concise but informative.
Do not repeat headings or generate a second answer.
Complete all six sections.
Never end a sentence or paragraph midway.
Finish the Management and Prevention section completely.
Do not include internal reasoning or thinking tags in the answer.
"""

    user_prompt = f"""Plant disease class: {disease_name}

Verified disease information:
{knowledge_card}

Question: Provide a complete, structured explanation of this plant disease using the six required sections.
"""

    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        max_tokens=750,
        temperature=0.0,
        top_p=1.0,
        repeat_penalty=1.1,
        seed=42
    )

    answer = response["choices"][0]["message"]["content"].strip()

    answer = answer.replace("<think>", "")
    answer = answer.replace("</think>", "")

    return answer.strip()
