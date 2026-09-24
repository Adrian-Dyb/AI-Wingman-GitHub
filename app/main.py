from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path
from datetime import datetime, timezone
from threading import Lock
from uuid import uuid4
import csv
import json
import os
import re
import time

load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

PROMPT_VERSION = os.getenv("PROMPT_VERSION", "experiment_prompt_v1.4")
GUIDELINES_VERSION = os.getenv("GUIDELINES_VERSION", "guidelines_v1.4")
PROTOCOL_VERSION = os.getenv("PROTOCOL_VERSION", "protocol_v1.0")
DATA_SCHEMA_VERSION = os.getenv("DATA_SCHEMA_VERSION", "data_schema_v1.0")
GUARDRAIL_VERSION = os.getenv("GUARDRAIL_VERSION", "output_guardrail_v1.2")

TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "300"))

EXPERIMENT_MODE = os.getenv("EXPERIMENT_MODE", "true").lower() == "true"

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

PARTICIPANTS_FILE = DATA_DIR / "participants.csv"
TRIALS_FILE = DATA_DIR / "trials.csv"
SUGGESTIONS_FILE = DATA_DIR / "suggestions.csv"
JUDGE_RATINGS_FILE = DATA_DIR / "judge_ratings.csv"
GUARDRAIL_EVENTS_FILE = DATA_DIR / "guardrail_events.csv"
GENERATION_ERRORS_FILE = DATA_DIR / "generation_errors.csv"

csv_lock = Lock()

client = (
    OpenAI(api_key=OPENAI_API_KEY, timeout=20.0)
    if OPENAI_API_KEY
    else None
)

app = FastAPI(
    title="AI Wingman API",
    version="2.1.3",
    description="Experiment-ready backend for the AI-wingman master thesis prototype.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # development only; restrict in production
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

EXPERIMENT_INSTRUCTION = (
    "Wyobraź sobie, że prowadzisz tę rozmowę w aplikacji randkowej. "
    "Napisz jedną wiadomość jako odpowiedź na ostatni komunikat rozmówcy. "
    "Odpowiedź powinna być naturalna i adekwatna do kontekstu oraz "
    "pozostawiać możliwość dalszej rozmowy. Nie ma jednej poprawnej odpowiedzi."
)

SCENARIOS = {
    "scenario_hiking": {
        "title": "Zainteresowania / góry",
        "context": [
            {
                "sender": "match",
                "text": (
                    "Ostatnio coraz częściej chodzę po górach. "
                    "Najbardziej lubię spokojniejsze trasy i dobre widoki."
                ),
            }
        ],
    },
    "scenario_travel": {
        "title": "Podróże",
        "context": [
            {
                "sender": "match",
                "text": (
                    "Gdybym jutro mogła polecieć gdziekolwiek, chyba wybrałabym Portugalię. "
                    "Najbardziej kuszą mnie ocean i małe miasta."
                ),
            }
        ],
    },
    "scenario_movies": {
        "title": "Filmy",
        "context": [
            {
                "sender": "match",
                "text": (
                    "Ostatnio trafił mi się naprawdę dobry film science fiction. "
                    "Lubię takie klimaty."
                ),
            }
        ],
    },
    "scenario_fading_conversation": {
        "title": "Wygasająca rozmowa",
        "context": [
            {
                "sender": "user",
                "text": "W sumie najlepsze plany i tak wychodzą spontanicznie.",
            },
            {
                "sender": "match",
                "text": "Haha, dokładnie 😂",
            },
        ],
    },
    "scenario_emotional": {
        "title": "Komunikat emocjonalny",
        "context": [
            {
                "sender": "match",
                "text": (
                    "Ostatnio mam trochę cięższy okres i chyba dlatego "
                    "rzadziej tu zaglądam."
                ),
            }
        ],
    },
}

GROUP_CONDITIONS = {
    "A": {
        "scenario_hiking": "manual",
        "scenario_travel": "ai_assisted",
        "scenario_movies": "manual",
        "scenario_fading_conversation": "ai_assisted",
        "scenario_emotional": "manual",
    },
    "B": {
        "scenario_hiking": "ai_assisted",
        "scenario_travel": "manual",
        "scenario_movies": "ai_assisted",
        "scenario_fading_conversation": "manual",
        "scenario_emotional": "ai_assisted",
    },
}

ORDER_VERSIONS = {
    "O1": [
        "scenario_hiking",
        "scenario_travel",
        "scenario_movies",
        "scenario_fading_conversation",
        "scenario_emotional",
    ],
    "O2": [
        "scenario_travel",
        "scenario_movies",
        "scenario_fading_conversation",
        "scenario_emotional",
        "scenario_hiking",
    ],
    "O3": [
        "scenario_movies",
        "scenario_fading_conversation",
        "scenario_emotional",
        "scenario_hiking",
        "scenario_travel",
    ],
    "O4": [
        "scenario_fading_conversation",
        "scenario_emotional",
        "scenario_hiking",
        "scenario_travel",
        "scenario_movies",
    ],
    "O5": [
        "scenario_emotional",
        "scenario_hiking",
        "scenario_travel",
        "scenario_movies",
        "scenario_fading_conversation",
    ],
}

RESPONSE_STRATEGIES = ["neutral", "warm", "engaging"]

# ============================================================
# CSV SCHEMAS
# ============================================================

PARTICIPANT_FIELDS = [
    "participant_id",
    "study_phase",
    "experimental_group",
    "order_version",
    "age",
    "dating_app_experience",
    "llm_experience",
    "registered_at",
    "completed_all_trials",
]

TRIAL_FIELDS = [
    "trial_id",
    "participant_id",
    "study_phase",
    "protocol_version",
    "data_schema_version",
    "session_id",
    "trial_number",
    "experimental_group",
    "order_version",
    "scenario_id",
    "assigned_condition",
    "suggestion_exposure",
    "final_action",
    "generation_id",
    "selected_suggestion_id",
    "final_text",
    "author_helpfulness_rating",
    "started_at",
    "completed_at",
    "duration_ms",
    "model_provider",
    "model_name",
    "prompt_version",
    "guidelines_version",
    "temperature",
    "max_tokens",
    "api_latency_ms",
    "api_error",
    "generation_status",
    "word_count",
    "char_count",
    "question_present",
    "emoji_count",
    "sentiment",
    "open_question_present",
]

SUGGESTION_FIELDS = [
    "suggestion_id",
    "generation_id",
    "trial_id",
    "study_phase",
    "participant_id",
    "scenario_id",
    "suggestion_position",
    "response_strategy",
    "suggestion_text",
    "was_selected",
    "was_edit_base",
    "generated_at",
    "model_provider",
    "model_name",
    "prompt_version",
    "guidelines_version",
    "temperature",
    "max_tokens",
    "api_latency_ms",
]

JUDGE_RATING_FIELDS = [
    "rating_id",
    "judge_id",
    "trial_id",
    "presentation_order",
    "quality_rating",
    "naturalness_rating",
    "attractiveness_rating",
    "continuation_rating",
    "started_at",
    "rated_at",
    "rating_duration_ms",
]


GUARDRAIL_EVENT_FIELDS = [
    "event_id",
    "generation_id",
    "trial_id",
    "participant_id",
    "scenario_id",
    "response_strategy",
    "original_text",
    "initial_flags",
    "repair_attempted",
    "repaired_text",
    "final_flags",
    "final_status",
    "created_at",
    "model_name",
    "prompt_version",
    "guidelines_version",
    "guardrail_version",
]


GENERATION_ERROR_FIELDS = [
    "event_id",
    "participant_id",
    "session_id",
    "trial_id",
    "trial_number",
    "experimental_group",
    "order_version",
    "scenario_id",
    "error_type",
    "error_message",
    "api_latency_ms",
    "created_at",
    "model_name",
    "prompt_version",
    "guidelines_version",
    "guardrail_version",
    "protocol_version",
    "data_schema_version",
]

# ============================================================
# PYDANTIC MODELS
# ============================================================

class ParticipantCreate(BaseModel):
    participant_id: str = Field(min_length=2, max_length=32)
    study_phase: Literal["pilot", "main"] = "main"
    experimental_group: Literal["A", "B"]
    order_version: Literal["O1", "O2", "O3", "O4", "O5"]
    age: int = Field(ge=18, le=99)
    dating_app_experience: Literal[
        "current", "previous", "none"
    ]
    llm_experience: Literal[
        "none", "occasional", "regular"
    ]


class SuggestionRequest(BaseModel):
    participant_id: str
    session_id: str
    trial_id: str
    trial_number: int = Field(ge=1, le=5)
    experimental_group: Literal["A", "B"]
    order_version: Literal["O1", "O2", "O3", "O4", "O5"]
    scenario_id: str


class TrialSubmitRequest(BaseModel):
    participant_id: str
    session_id: str
    trial_id: str
    trial_number: int = Field(ge=1, le=5)
    experimental_group: Literal["A", "B"]
    order_version: Literal["O1", "O2", "O3", "O4", "O5"]
    scenario_id: str
    final_action: Literal[
        "manual", "use", "edit", "reject_and_write_own"
    ]
    generation_id: Optional[str] = None
    selected_suggestion_id: Optional[str] = None
    final_text: str = Field(min_length=1, max_length=2000)
    author_helpfulness_rating: Optional[int] = Field(
        default=None, ge=1, le=5
    )
    started_at: datetime


class AnalyzeRequest(BaseModel):
    text: str


class JudgeRatingRequest(BaseModel):
    judge_id: str = Field(min_length=2, max_length=32)
    trial_id: str
    presentation_order: int = Field(ge=1)
    quality_rating: int = Field(ge=1, le=5)
    naturalness_rating: int = Field(ge=1, le=5)
    attractiveness_rating: int = Field(ge=1, le=5)
    continuation_rating: int = Field(ge=1, le=5)
    started_at: datetime


# ============================================================
# CSV HELPERS
# ============================================================

def append_csv_row(path: Path, fieldnames: List[str], row: dict) -> None:
    with csv_lock:
        file_exists = path.exists() and path.stat().st_size > 0

        with path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerow({
                field: row.get(field, "")
                for field in fieldnames
            })


def read_csv_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []

    with csv_lock:
        with path.open("r", newline="", encoding="utf-8") as file:
            return list(csv.DictReader(file))


def rewrite_csv_rows(
    path: Path,
    fieldnames: List[str],
    rows: List[dict],
) -> None:
    with csv_lock:
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def find_row(path: Path, field: str, value: str) -> Optional[dict]:
    for row in read_csv_rows(path):
        if row.get(field) == value:
            return row
    return None


# ============================================================
# EXPERIMENT HELPERS
# ============================================================

def get_assigned_condition(
    experimental_group: str,
    scenario_id: str,
) -> str:
    if scenario_id not in SCENARIOS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown scenario_id: {scenario_id}",
        )

    return GROUP_CONDITIONS[experimental_group][scenario_id]


def validate_trial_position(
    order_version: str,
    trial_number: int,
    scenario_id: str,
) -> None:
    expected_scenario = ORDER_VERSIONS[order_version][trial_number - 1]

    if expected_scenario != scenario_id:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid scenario for trial {trial_number}. "
                f"Expected {expected_scenario}, got {scenario_id}."
            ),
        )


def participant_matches_request(
    participant_id: str,
    experimental_group: str,
    order_version: str,
) -> None:
    participant = find_row(
        PARTICIPANTS_FILE,
        "participant_id",
        participant_id,
    )

    if participant is None:
        raise HTTPException(
            status_code=404,
            detail="Participant is not registered.",
        )

    if participant["experimental_group"] != experimental_group:
        raise HTTPException(
            status_code=400,
            detail="Participant group does not match registration.",
        )

    if participant["order_version"] != order_version:
        raise HTTPException(
            status_code=400,
            detail="Order version does not match registration.",
        )


def scenario_context_as_text(scenario_id: str) -> str:
    context = SCENARIOS[scenario_id]["context"]
    labels = {"match": "Match", "user": "User"}

    return "\n".join(
        f"{labels.get(item['sender'], item['sender'])}: {item['text']}"
        for item in context
    )


# ============================================================
# PROMPT + LLM
# ============================================================

def load_guidelines() -> str:
    # Frozen experiment guardrails. These rules were refined after
    # the v1.1 development stress test. They explicitly distinguish
    # facts stated by the USER from facts stated by the MATCH.
    if EXPERIMENT_MODE:
        return """
Zasady generowania sugestii AI-wingman — v1.4:

1. System wspiera użytkownika, ale nie zastępuje go.
2. Sugestie mają być krótkie, naturalne i możliwe do edycji.
3. Nie generuj treści manipulacyjnych, presyjnych ani seksualnych.

4. ŹRÓDŁO FAKTÓW O UŻYTKOWNIKU:
   Za prawdziwe informacje o użytkowniku wolno uznać wyłącznie to,
   co użytkownik sam wcześniej podał w wypowiedziach oznaczonych
   jako User. Wypowiedzi Match opisują rozmówcę, nie użytkownika.

5. ZAKAZ FABRYKOWANIA AUTOBIOGRAFII:
   Nie przypisuj użytkownikowi niewskazanych:
   - uczuć i stanów emocjonalnych,
   - intencji interpersonalnych,
   - ocen relacji,
   - preferencji,
   - doświadczeń,
   - planów,
   - wiedzy i przekonań,
   - faktów biograficznych,
   - ani BRAKU powyższych informacji.

6. TWARDY ZAKAZ NIEUZASADNIONEJ PIERWSZEJ OSOBY:
   Jeżeli dana treść nie wynika bezpośrednio z wypowiedzi User,
   nie używaj w imieniu użytkownika sformułowań takich jak:
   „przykro mi”, „cieszę się”, „mam nadzieję”, „martwię się”,
   „jestem tutaj”, „będę tutaj”, „chętnie”, „lubię”, „uwielbiam”,
   „też lubię”, „zależy mi”, „tęsknię”, „nie mogę się doczekać”,
   „dużo słyszałem”, „słyszałem”, „nie mam planów”,
   „jeszcze tam nie byłem”, „nie wiem, gdzie pojadę”
   ani innych równoważnych deklaracji w pierwszej osobie.

7. PYTANIA AUTOBIOGRAFICZNE OD MATCH:
   Jeżeli Match pyta o osobiste doświadczenie, preferencję lub plan
   użytkownika, których system nie zna z wypowiedzi User, NIE odpowiadaj
   fikcyjnym faktem i NIE wymyślaj nawet braku takiego faktu.
   Zamiast tego możesz:
   - odnieść się neutralnie do wypowiedzi Match,
   - zapytać o perspektywę Match,
   - użyć konstrukcji warunkowej,
   o ile nie występuje odmowa lub granica.

8. ZAKAZ PRESUPOZYCJI:
   Nie formułuj pytań zakładających istnienie niepodanych doświadczeń,
   preferencji lub planów użytkownika. Unikaj m.in.:
   „jakie jest Twoje ulubione...?”,
   „który wyjazd wspominasz najlepiej?”,
   „gdzie planujesz urlop?”,
   „co zawsze chciałeś odwiedzić?”,
   jeśli odpowiednia informacja nie wynika z wypowiedzi User.

9. ODMOWA / GRANICA — REGUŁA NADRZĘDNA:
   Jeżeli ostatnia wypowiedź Match wyraża:
   - odmowę,
   - brak zainteresowania,
   - potrzebę przestrzeni,
   - chęć zakończenia kontaktu,
   - brak zgody na spotkanie lub eskalację,
   wtedy WSZYSTKIE trzy strategie mają wyłącznie krótko
   zaakceptować granicę.

10. W sytuacji z punktu 9:
    - nie zadawaj żadnego pytania,
    - nie używaj znaku zapytania,
    - nie pytaj o uzasadnienie,
    - nie proponuj kompromisu,
    - nie proponuj innego spotkania lub terminu,
    - nie próbuj ponownie otwierać rozmowy,
    - nie deklaruj przyszłej dostępności użytkownika
      („jestem tutaj”, „będę tutaj”, „daj znać”),
    - nie sugeruj, że rozmówca powinien zmienić zdanie.

11. GRANICA SEKSUALNA / INTYMNA:
    Jeżeli rozmówca zaznacza brak zainteresowania treściami seksualnymi
    lub inną granicę intymności, nie seksualizuj odpowiedzi i nie eskaluj
    bliskości.

12. BRAK PRESJI I MANIPULACJI:
    Nie sugeruj, że rozmówca ma obowiązek odpowiedzieć, wyjaśnić się,
    przeprosić, zmienić zdanie lub podjąć określone działanie.
    Nie wzbudzaj winy i nie używaj emocjonalnego szantażu.

13. STRATEGIE:
    - neutral: najprostsza i najbardziej zachowawcza;
    - warm: życzliwa, ale bez tworzenia emocji, intencji lub deklaracji
      dostępności w imieniu użytkownika;
    - engaging: może rozwijać rozmowę tylko wtedy, gdy kontekst na to
      pozwala. Respektowanie granic ma zawsze pierwszeństwo.

14. PYTANIA:
    Pytanie otwarte jest opcjonalne. Może pojawić się tylko wtedy,
    gdy nie zakłada niepodanych faktów i nie narusza granic rozmówcy.

15. KONTROLA PRZED ZWROTEM:
    Przed zwróceniem każdej sugestii sprawdź kolejno:
    a) Czy zawiera jakąkolwiek nieuzasadnioną deklarację w pierwszej osobie?
    b) Czy wymyśla fakt o użytkowniku lub brak takiego faktu?
    c) Czy pytanie presuponuje nieznane doświadczenie/preferencję/plan?
    d) Czy ostatnia wypowiedź Match ustanawia granicę?
    e) Jeśli tak, czy sugestia jest wyłącznie krótkim acknowledgement
       bez pytania i bez próby kontynuacji?
    Jeżeli którykolwiek warunek nie jest spełniony, przepisz sugestię.
""".strip()

    path = Path("guidelines.txt")

    if path.exists():
        return path.read_text(encoding="utf-8")

    return """
Zasady generowania sugestii AI-wingman:
1. System wspiera użytkownika, ale nie zastępuje go.
2. Sugestie mają być krótkie, naturalne i możliwe do edycji.
3. Nie generuj treści manipulacyjnych, presyjnych ani seksualnych.
4. Nie deklaruj uczuć, preferencji, doświadczeń ani planów za użytkownika.
5. Respektuj odmowy i granice rozmówcy.
6. Zachowaj adekwatność do kontekstu.
""".strip()


def build_prompt(scenario_id: str) -> str:
    guidelines = load_guidelines()
    context_text = scenario_context_as_text(scenario_id)

    return f"""
Jesteś komponentem systemu AI-wingman wspierającego użytkownika
w formułowaniu odpowiedzi w aplikacji randkowej.

WERSJA PROMPTU:
{PROMPT_VERSION}

ZASADY:
{guidelines}

KONTEKST ROZMOWY:
{context_text}

CEL EKSPERYMENTALNY:
Utwórz propozycje odpowiedzi na ostatni komunikat rozmówcy.
Odpowiedzi powinny być naturalne i adekwatne do kontekstu.
Dalszą rozmowę podtrzymuj tylko wtedy, gdy ostatnia wypowiedź Match
nie zawiera odmowy, potrzeby przestrzeni ani innej granicy.

REGUŁY KRYTYCZNE:
1. Match i User to dwie różne osoby.
2. Fakty o użytkowniku mogą pochodzić wyłącznie z wypowiedzi User.
3. Nie twórz w imieniu użytkownika uczuć, intencji, preferencji,
   doświadczeń, planów, wiedzy ani ich braku.
4. Bez podstawy w User nie używaj m.in.:
   „przykro mi”, „cieszę się”, „mam nadzieję”, „jestem tutaj”,
   „będę tutaj”, „chętnie”, „lubię”, „uwielbiam”, „też lubię”,
   „dużo słyszałem”, „nie mam planów”, „jeszcze tam nie byłem”.
5. Jeżeli Match zadaje pytanie autobiograficzne, na które brak danych
   w wypowiedziach User, nie odpowiadaj fikcyjnym faktem. Odnieś się
   neutralnie do Match albo zapytaj o jego/jej perspektywę.
6. Nie zadawaj pytań, które presuponują nieznane fakty o użytkowniku.
7. Jeżeli ostatnia wypowiedź Match jest odmową lub prośbą o przestrzeń,
   wszystkie trzy odpowiedzi mają być krótkim zaakceptowaniem granicy:
   bez pytania, bez znaku „?”, bez negocjacji, bez „daj znać”,
   bez „jestem/będę tutaj” i bez próby ponownego otwarcia rozmowy.
8. Reguła granic ma pierwszeństwo przed strategią engaging.
9. Nie stosuj presji, manipulacji ani eskalacji seksualnej.

AUTOKONTROLA KAŻDEJ SUGESTII:
- Czy zawiera pierwszoosobową deklarację niepodaną przez User?
- Czy wymyśla fakt o użytkowniku albo brak faktu?
- Czy pytanie zakłada nieznane doświadczenie/preferencję/plan?
- Czy ostatnia wypowiedź Match ustanawia granicę?
- Jeśli ustanawia granicę: czy sugestia nie zawiera żadnego pytania
  ani zachęty do kontynuacji?
Jeśli wykryjesz problem, przepisz sugestię przed zwróceniem JSON.

Wygeneruj dokładnie trzy strategie:
1. neutral
2. warm
3. engaging

Zwróć WYŁĄCZNIE poprawny JSON:

{{
  "suggestions": [
    {{"response_strategy": "neutral", "text": "..."}},
    {{"response_strategy": "warm", "text": "..."}},
    {{"response_strategy": "engaging", "text": "..."}}
  ]
}}
""".strip()


def parse_model_response(raw_text: str) -> List[dict]:
    cleaned = raw_text.strip()

    if cleaned.startswith("```"):
        cleaned = (
            cleaned
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

    data = json.loads(cleaned)
    suggestions = data.get("suggestions")

    if not isinstance(suggestions, list):
        raise ValueError("Missing suggestions list.")

    parsed = []

    for strategy in RESPONSE_STRATEGIES:
        found = next(
            (
                item
                for item in suggestions
                if item.get("response_strategy") == strategy
                and isinstance(item.get("text"), str)
                and item["text"].strip()
            ),
            None,
        )

        if found is None:
            raise ValueError(
                f"Missing valid strategy: {strategy}"
            )

        parsed.append({
            "response_strategy": strategy,
            "text": found["text"].strip(),
        })

    return parsed


def dev_fallback_suggestions() -> List[dict]:
    return [
        {
            "response_strategy": "neutral",
            "text": "Brzmi ciekawie. Co najbardziej podoba Ci się w tym pomyśle?",
        },
        {
            "response_strategy": "warm",
            "text": "To brzmi naprawdę przyjemnie.",
        },
        {
            "response_strategy": "engaging",
            "text": "Co najbardziej Cię w tym przekonuje?",
        },
    ]


def find_existing_generation(trial_id: str) -> Optional[dict]:
    rows = [
        row
        for row in read_csv_rows(SUGGESTIONS_FILE)
        if row.get("trial_id") == trial_id
    ]

    if not rows:
        return None

    generation_id = rows[0]["generation_id"]
    same_generation = [
        row for row in rows
        if row.get("generation_id") == generation_id
    ]

    if len(same_generation) != 3:
        return None

    same_generation.sort(
        key=lambda row: int(row["suggestion_position"])
    )

    return {
        "generation_id": generation_id,
        "suggestions": [
            {
                "suggestion_id": row["suggestion_id"],
                "response_strategy": row["response_strategy"],
                "text": row["suggestion_text"],
            }
            for row in same_generation
        ],
        "model_provider": same_generation[0]["model_provider"],
        "model_name": same_generation[0]["model_name"],
        "prompt_version": same_generation[0]["prompt_version"],
        "guidelines_version": same_generation[0]["guidelines_version"],
        "temperature": float(same_generation[0]["temperature"]),
        "max_tokens": int(same_generation[0]["max_tokens"]),
        "api_latency_ms": int(same_generation[0]["api_latency_ms"]),
        "guardrail_triggered": False,
        "regenerated_strategies": [],
        "guardrail_version": GUARDRAIL_VERSION,
        "cached": True,
    }


def get_generation_rows(generation_id: str) -> List[dict]:
    return [
        row
        for row in read_csv_rows(SUGGESTIONS_FILE)
        if row.get("generation_id") == generation_id
    ]


def update_suggestion_usage(
    generation_id: str,
    selected_suggestion_id: Optional[str],
    final_action: str,
) -> None:
    rows = read_csv_rows(SUGGESTIONS_FILE)

    changed = False

    for row in rows:
        if row.get("generation_id") != generation_id:
            continue

        is_selected = (
            selected_suggestion_id is not None
            and row.get("suggestion_id") == selected_suggestion_id
        )

        row["was_selected"] = str(is_selected)
        row["was_edit_base"] = str(
            is_selected and final_action == "edit"
        )
        changed = True

    if changed:
        rewrite_csv_rows(
            SUGGESTIONS_FILE,
            SUGGESTION_FIELDS,
            rows,
        )



# ============================================================
# DETERMINISTIC OUTPUT GUARDRAILS
# ============================================================
#
# Ta warstwa NIE próbuje rozwiązać ogólnego problemu moderacji.
# Kontroluje wyłącznie klasy błędów wykryte empirycznie podczas
# development audit / stress testów v1.0-v1.3.
#
# Zasada działania:
# 1. LLM generuje 3 sugestie.
# 2. Poniższy validator wykrywa obserwowane wcześniej wzorce.
# 3. Jeżeli co najmniej jedna sugestia jest oznaczona flagą,
#    wszystkie wadliwe strategie są naprawiane w JEDNYM dodatkowym
#    wywołaniu LLM.
# 4. Naprawione sugestie są ponownie sprawdzane.
# 5. Jeżeli nadal występuje flaga, wynik nie jest zwracany
#    uczestnikowi w EXPERIMENT_MODE.

BLOCKED_FIRST_PERSON_PATTERNS = [
    r"\bprzykro mi\b",
    r"\bcieszę się\b",
    r"\bmam nadzieję\b",
    r"\bmartwię się\b",
    r"\bjestem tutaj\b",
    r"\bbędę tutaj\b",
    r"\bjestem otwart[ya]\b",
    r"\bbędę otwart[ya]\b",
    r"\bchętnie\b",
    r"\bteż lubię\b",
    r"\blubię\b",
    r"\buwielbiam\b",
    r"\bzależy mi\b",
    r"\btęsknię\b",
    r"\bnie mogę się doczekać\b",
    r"\bciekaw jestem\b",
    r"\bciekawa jestem\b",
    r"\bjestem ciekaw[ay]?\b",
    r"\bdużo słyszałem\b",
    r"\bdużo słyszałam\b",
    r"\bsłyszałem\b",
    r"\bsłyszałam\b",
    r"\bnie planuję\b",
    r"\bnie mam planów\b",
    r"\bnie mam jeszcze planów\b",
    r"\bjeszcze tam nie byłem\b",
    r"\bjeszcze tam nie byłam\b",
    r"\bnie mogę powiedzieć\b",
    r"\bżyczę ci\b",
    r"\btrzymam kciuki\b",

    # output_guardrail_v1.2 — false negatives observed in v1.4.1:
    # "Nie mam jeszcze ulubionego kraju."
    r"\bnie mam (?:jeszcze )?ulubion(?:ego|ej|ych|ą|y)\b",

    # "Nie ma jeszcze planów urlopowych."
    # Deliberately narrow: catches autobiographical/impersonal claims
    # about lack of plans without broadly blocking every "nie ma".
    r"\bnie ma (?:jeszcze )?(?:konkretn(?:ych|ego|ej) )?plan(?:ów|u|y)\b",

    # Related paraphrase of the same unsupported lack-of-plan claim.
    r"\bbrak (?:jeszcze )?(?:konkretn(?:ych|ego|ej) )?plan(?:ów|u|y)\b",

    # output_guardrail_v1.2 — semantic paraphrases observed in v1.4.2.
    # These remain declarations of the user's internal state/intention
    # under the annotation codebook, even when they avoid earlier wording.
    r"\bżyczę\b",
    r"\bciekawi mnie\b",
    r"\binteresuje mnie\b",
    r"\bwezmę to pod uwagę\b",
    r"\bpostaram się\b",
]

HARD_BOUNDARY_PATTERNS = [
    r"\bpotrzebuję (trochę )?przestrzeni\b",
    r"\bwolę teraz nie pisać\b",
    r"\bnie chcę teraz pisać\b",
    r"\bnie chcę dalej pisać\b",
    r"\bnie chcę kontynuować\b",
    r"\bproszę[,]? nie pisz\b",
    r"\bzakończmy (tę )?(rozmowę|kontakt)\b",
    r"\bskończmy (tę )?(rozmowę|kontakt)\b",
]

BOUNDARY_REOPENING_PATTERNS = [
    r"\bdaj znać\b",
    r"\bgdy będziesz gotow",
    r"\bkiedy będziesz gotow",
    r"\bjeśli będziesz gotow",
    r"\bjestem tutaj\b",
    r"\bbędę tutaj\b",
    r"\bmoże kiedyś\b",
]

def _normalize_guardrail_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _last_match_text(context_text: str) -> str:
    matches = []

    for line in context_text.splitlines():
        stripped = line.strip()

        if stripped.lower().startswith("match:"):
            matches.append(stripped.split(":", 1)[1].strip())

    return matches[-1] if matches else context_text.strip()


def is_hard_boundary_context(context_text: str) -> bool:
    last_match = _normalize_guardrail_text(
        _last_match_text(context_text)
    )

    if any(
        re.search(pattern, last_match)
        for pattern in HARD_BOUNDARY_PATTERNS
    ):
        return True

    # Development audit pokazał, że kombinacja zamknięcia rozmowy
    # z odmową spotkania powinna być traktowana jako zakończenie kontaktu.
    if "dzięki za rozmowę" in last_match and any(
        phrase in last_match
        for phrase in [
            "nie czuję",
            "nie chcę",
            "nie jestem zainteres",
            "nie powinniśmy się spotkać",
        ]
    ):
        return True

    return False


def validate_suggestion_text(
    text: str,
    context_text: str,
) -> List[str]:
    normalized = _normalize_guardrail_text(text)
    flags = []

    # 1. Empirycznie wykrywane nieuzasadnione deklaracje
    #    pierwszoosobowe / autobiograficzne.
    for pattern in BLOCKED_FIRST_PERSON_PATTERNS:
        if re.search(pattern, normalized):
            flags.append("unsupported_first_person")
            break

    # 2. Po twardej granicy odpowiedź nie może ponownie
    #    otwierać rozmowy.
    if is_hard_boundary_context(context_text):
        if "?" in text:
            flags.append("boundary_question")

        if any(
            re.search(pattern, normalized)
            for pattern in BOUNDARY_REOPENING_PATTERNS
        ):
            flags.append("boundary_reopening")

    return sorted(set(flags))


class GuardrailValidationError(RuntimeError):
    def __init__(self, message: str, trace: List[dict]):
        super().__init__(message)
        self.trace = trace


def repair_flagged_suggestions(
    context_text: str,
    flagged_items: List[dict],
) -> List[dict]:
    if client is None:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured."
        )

    guidelines = load_guidelines()

    unsafe_json = json.dumps(
        [
            {
                "response_strategy": item["response_strategy"],
                "text": item["text"],
                "flags": item["flags"],
            }
            for item in flagged_items
        ],
        ensure_ascii=False,
        indent=2,
    )

    repair_prompt = f"""
Jesteś modułem naprawczym systemu AI-wingman.

ZASADY:
{guidelines}

KONTEKST:
{context_text}

Poniższe sugestie zostały odrzucone przez deterministyczny
validator wyjścia:

{unsafe_json}

Wygeneruj poprawione wersje WYŁĄCZNIE dla wskazanych strategii.

WYMAGANIA BEZWZGLĘDNE:
- nie używaj nieuzasadnionych deklaracji pierwszoosobowych;
- nie wymyślaj uczuć, intencji, preferencji, doświadczeń,
  planów, wiedzy ani ich braku za użytkownika;
- nie używaj równoważnych parafraz zakazanych deklaracji;
- jeśli ostatnia wypowiedź Match stanowi twardą odmowę,
  potrzebę przestrzeni lub zakończenie kontaktu:
  odpowiedź ma być krótkim zaakceptowaniem granicy,
  bez pytania, bez znaku zapytania, bez „daj znać”,
  bez deklaracji przyszłej dostępności i bez próby kontynuacji.

Zachowaj dokładnie te same wartości response_strategy.

Zwróć WYŁĄCZNIE JSON:
{{
  "suggestions": [
    {{"response_strategy": "...", "text": "..."}}
  ]
}}
""".strip()

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Naprawiasz wyłącznie sugestie odrzucone przez "
                    "deterministyczny validator bezpieczeństwa. "
                    "Nie dodawaj komentarzy. Zwróć wyłącznie JSON."
                ),
            },
            {
                "role": "user",
                "content": repair_prompt,
            },
        ],
        temperature=0.2,
        max_tokens=MAX_TOKENS,
        response_format={"type": "json_object"},
    )

    raw_text = response.choices[0].message.content or ""
    cleaned = raw_text.strip()

    if cleaned.startswith("```"):
        cleaned = (
            cleaned
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

    data = json.loads(cleaned)
    repaired = data.get("suggestions")

    if not isinstance(repaired, list):
        raise ValueError(
            "Repair response does not contain suggestions list."
        )

    expected_strategies = {
        item["response_strategy"]
        for item in flagged_items
    }

    result = []

    for strategy in expected_strategies:
        found = next(
            (
                item
                for item in repaired
                if item.get("response_strategy") == strategy
                and isinstance(item.get("text"), str)
                and item["text"].strip()
            ),
            None,
        )

        if found is None:
            raise ValueError(
                f"Repair response missing strategy: {strategy}"
            )

        result.append({
            "response_strategy": strategy,
            "text": found["text"].strip(),
        })

    return result


def apply_output_guardrails(
    context_text: str,
    suggestions: List[dict],
) -> tuple[List[dict], List[dict]]:
    initial_by_strategy = {
        item["response_strategy"]: {
            "response_strategy": item["response_strategy"],
            "text": item["text"],
            "flags": validate_suggestion_text(
                item["text"],
                context_text,
            ),
        }
        for item in suggestions
    }

    flagged_items = [
        item
        for item in initial_by_strategy.values()
        if item["flags"]
    ]

    if not flagged_items:
        return suggestions, []

    # Jedna zbiorcza regeneracja wszystkich wadliwych strategii.
    repaired = repair_flagged_suggestions(
        context_text,
        flagged_items,
    )

    repaired_map = {
        item["response_strategy"]: item["text"]
        for item in repaired
    }

    final_suggestions = []
    trace = []
    remaining_unsafe = []

    for item in suggestions:
        strategy = item["response_strategy"]
        initial = initial_by_strategy[strategy]

        if not initial["flags"]:
            final_suggestions.append(item)
            continue

        repaired_text = repaired_map[strategy]
        final_flags = validate_suggestion_text(
            repaired_text,
            context_text,
        )

        trace_item = {
            "response_strategy": strategy,
            "original_text": item["text"],
            "initial_flags": initial["flags"],
            "repair_attempted": True,
            "repaired_text": repaired_text,
            "final_flags": final_flags,
            "final_status": (
                "passed_after_repair"
                if not final_flags
                else "failed_after_repair"
            ),
        }
        trace.append(trace_item)

        if final_flags:
            remaining_unsafe.append(trace_item)
        else:
            final_suggestions.append({
                "response_strategy": strategy,
                "text": repaired_text,
            })

    if remaining_unsafe:
        raise GuardrailValidationError(
            "Suggestion remained unsafe after one repair attempt.",
            trace,
        )

    # Przywróć oryginalną kolejność neutral/warm/engaging.
    final_suggestions.sort(
        key=lambda item: RESPONSE_STRATEGIES.index(
            item["response_strategy"]
        )
    )

    return final_suggestions, trace


def log_guardrail_trace(
    trace: List[dict],
    generation_id: str,
    trial_id: str,
    participant_id: str,
    scenario_id: str,
) -> None:
    created_at = datetime.now(timezone.utc).isoformat()

    for item in trace:
        append_csv_row(
            GUARDRAIL_EVENTS_FILE,
            GUARDRAIL_EVENT_FIELDS,
            {
                "event_id": f"GR_{uuid4().hex[:12]}",
                "generation_id": generation_id,
                "trial_id": trial_id,
                "participant_id": participant_id,
                "scenario_id": scenario_id,
                "response_strategy": item[
                    "response_strategy"
                ],
                "original_text": item["original_text"],
                "initial_flags": "|".join(
                    item["initial_flags"]
                ),
                "repair_attempted": item[
                    "repair_attempted"
                ],
                "repaired_text": item["repaired_text"],
                "final_flags": "|".join(
                    item["final_flags"]
                ),
                "final_status": item["final_status"],
                "created_at": created_at,
                "model_name": OPENAI_MODEL,
                "prompt_version": PROMPT_VERSION,
                "guidelines_version": GUIDELINES_VERSION,
                "guardrail_version": GUARDRAIL_VERSION,
            },
        )



def log_generation_error(
    request: SuggestionRequest,
    exc: Exception,
    api_latency_ms: int,
) -> None:
    append_csv_row(
        GENERATION_ERRORS_FILE,
        GENERATION_ERROR_FIELDS,
        {
            "event_id": f"GE_{uuid4().hex[:12]}",
            "participant_id": request.participant_id,
            "session_id": request.session_id,
            "trial_id": request.trial_id,
            "trial_number": request.trial_number,
            "experimental_group": request.experimental_group,
            "order_version": request.order_version,
            "scenario_id": request.scenario_id,
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:1000],
            "api_latency_ms": api_latency_ms,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_name": OPENAI_MODEL,
            "prompt_version": PROMPT_VERSION,
            "guidelines_version": GUIDELINES_VERSION,
            "guardrail_version": GUARDRAIL_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "data_schema_version": DATA_SCHEMA_VERSION,
        },
    )


# ============================================================
# TEXT ANALYSIS
# ============================================================

OPEN_QUESTION_WORDS = {
    "co",
    "czym",
    "czego",
    "jak",
    "jaka",
    "jaki",
    "jakie",
    "jakiego",
    "gdzie",
    "dokąd",
    "skąd",
    "kiedy",
    "dlaczego",
    "czemu",
    "który",
    "która",
    "które",
    "ile",
}


def has_open_question(text: str) -> bool:
    text_lower = text.lower()

    # Only analyse spans that actually end with a question mark.
    question_spans = re.findall(r"([^?]+)\?", text_lower)

    for span in question_spans:
        words = re.findall(r"\b[\wąćęłńóśźż]+\b", span)

        if any(word in OPEN_QUESTION_WORDS for word in words):
            return True

    return False


def analyze_text(text: str) -> dict:
    text = text.strip()
    words = re.findall(r"\b[\wąćęłńóśźż-]+\b", text.lower())
    text_lower = text.lower()

    positive_phrases = [
        "super",
        "świetnie",
        "fajnie",
        "dobrze",
        "ciekawie",
        "miło",
        "brzmi dobrze",
        "chętnie",
    ]

    negative_phrases = [
        "nie lubię",
        "źle",
        "nudno",
        "słabo",
        "problem",
        "dziwnie",
    ]

    emoji_pattern = re.compile(
        "["
        "\U0001F1E0-\U0001F1FF"
        "\U0001F300-\U0001F5FF"
        "\U0001F600-\U0001F64F"
        "\U0001F680-\U0001F6FF"
        "\U0001F900-\U0001F9FF"
        "]+",
        flags=re.UNICODE,
    )

    emoji_count = sum(
        len(match)
        for match in emoji_pattern.findall(text)
    )

    negative_score = sum(
        1 for phrase in negative_phrases
        if phrase in text_lower
    )

    positive_score = sum(
        1 for phrase in positive_phrases
        if phrase in text_lower
    )

    if positive_score > negative_score:
        sentiment = "positive"
    elif negative_score > positive_score:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return {
        "word_count": len(words),
        "char_count": len(text),
        "question_present": "?" in text,
        "emoji_count": emoji_count,
        "sentiment": sentiment,
        "open_question_present": has_open_question(text),
    }


# ============================================================
# PARTICIPANT HELPERS
# ============================================================

def update_participant_completion(participant_id: str) -> None:
    participant_trials = {
        row["trial_id"]
        for row in read_csv_rows(TRIALS_FILE)
        if row.get("participant_id") == participant_id
    }

    if len(participant_trials) < 5:
        return

    participants = read_csv_rows(PARTICIPANTS_FILE)

    changed = False

    for participant in participants:
        if participant.get("participant_id") == participant_id:
            participant["completed_all_trials"] = "True"
            changed = True

    if changed:
        rewrite_csv_rows(
            PARTICIPANTS_FILE,
            PARTICIPANT_FIELDS,
            participants,
        )


# ============================================================
# ROUTES: HEALTH / CONFIG
# ============================================================

@app.get("/")
def root():
    return {
        "message": "AI Wingman backend is running",
        "version": "2.1.3",
        "experiment_mode": EXPERIMENT_MODE,
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "experiment_mode": EXPERIMENT_MODE,
        "openai_configured": client is not None,
        "model": OPENAI_MODEL,
        "prompt_version": PROMPT_VERSION,
        "guidelines_version": GUIDELINES_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "data_schema_version": DATA_SCHEMA_VERSION,
        "guardrail_version": GUARDRAIL_VERSION,
    }


@app.get("/api/experiment/scenarios")
def get_scenarios():
    return {
        "instruction": EXPERIMENT_INSTRUCTION,
        "scenarios": SCENARIOS,
    }


# ============================================================
# ROUTES: PARTICIPANTS / EXPERIMENT CONFIG
# ============================================================

@app.post("/api/participants")
def register_participant(data: ParticipantCreate):
    if find_row(
        PARTICIPANTS_FILE,
        "participant_id",
        data.participant_id,
    ):
        raise HTTPException(
            status_code=409,
            detail="Participant already exists.",
        )

    append_csv_row(
        PARTICIPANTS_FILE,
        PARTICIPANT_FIELDS,
        {
            "participant_id": data.participant_id,
            "study_phase": data.study_phase,
            "experimental_group": data.experimental_group,
            "order_version": data.order_version,
            "age": data.age,
            "dating_app_experience": data.dating_app_experience,
            "llm_experience": data.llm_experience,
            "registered_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "completed_all_trials": False,
        },
    )

    return {
        "status": "ok",
        "participant_id": data.participant_id,
    }


@app.get("/api/experiment/config/{participant_id}")
def get_participant_experiment_config(participant_id: str):
    participant = find_row(
        PARTICIPANTS_FILE,
        "participant_id",
        participant_id,
    )

    if participant is None:
        raise HTTPException(
            status_code=404,
            detail="Participant is not registered.",
        )

    group = participant["experimental_group"]
    order_version = participant["order_version"]
    order = ORDER_VERSIONS[order_version]

    trials = []

    for index, scenario_id in enumerate(order, start=1):
        trials.append({
            "trial_number": index,
            "scenario_id": scenario_id,
            "title": SCENARIOS[scenario_id]["title"],
            "context": SCENARIOS[scenario_id]["context"],
            "assigned_condition": GROUP_CONDITIONS[group][scenario_id],
        })

    return {
        "participant_id": participant_id,
        "study_phase": participant.get("study_phase", "main"),
        "experimental_group": group,
        "order_version": order_version,
        "instruction": EXPERIMENT_INSTRUCTION,
        "trials": trials,
    }



@app.get("/api/experiment/progress/{participant_id}")
def get_participant_progress(participant_id: str):
    participant = find_row(
        PARTICIPANTS_FILE,
        "participant_id",
        participant_id,
    )

    if participant is None:
        raise HTTPException(
            status_code=404,
            detail="Participant is not registered.",
        )

    order_version = participant["order_version"]
    expected_order = ORDER_VERSIONS[order_version]

    completed_rows = [
        row
        for row in read_csv_rows(TRIALS_FILE)
        if row.get("participant_id") == participant_id
    ]

    completed_scenarios = {
        row.get("scenario_id")
        for row in completed_rows
    }

    completed_trial_numbers = sorted({
        int(row["trial_number"])
        for row in completed_rows
        if row.get("trial_number", "").isdigit()
    })

    next_trial_index = 5

    for index, scenario_id in enumerate(expected_order):
        if scenario_id not in completed_scenarios:
            next_trial_index = index
            break

    return {
        "participant_id": participant_id,
        "completed_trial_numbers": completed_trial_numbers,
        "completed_scenarios": sorted(completed_scenarios),
        "completed_count": len(completed_scenarios),
        "next_trial_index": next_trial_index,
        "completed_all_trials": next_trial_index >= 5,
    }


# ============================================================
# ROUTES: SUGGESTIONS
# ============================================================

@app.post("/api/suggestions")
def generate_suggestions(request: SuggestionRequest):
    participant_matches_request(
        request.participant_id,
        request.experimental_group,
        request.order_version,
    )

    validate_trial_position(
        request.order_version,
        request.trial_number,
        request.scenario_id,
    )

    assigned_condition = get_assigned_condition(
        request.experimental_group,
        request.scenario_id,
    )

    if assigned_condition != "ai_assisted":
        raise HTTPException(
            status_code=400,
            detail=(
                "Suggestions can only be generated "
                "for an AI-assisted trial."
            ),
        )

    # One successful generation per trial.
    existing = find_existing_generation(request.trial_id)

    if existing is not None:
        return existing

    prompt = build_prompt(request.scenario_id)

    generation_id = f"G_{uuid4().hex[:12]}"
    generated_at = datetime.now(timezone.utc).isoformat()

    start = time.perf_counter()
    guardrail_trace = []

    try:
        if client is None:
            raise RuntimeError(
                "OPENAI_API_KEY is not configured."
            )

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Jesteś komponentem generowania systemu AI-wingman. "
                        "Traktuj Match i User jako dwie różne osoby. Fakty o użytkowniku "
                        "mogą pochodzić wyłącznie z wypowiedzi User. Nie twórz za użytkownika "
                        "uczuć, intencji, preferencji, doświadczeń, planów, wiedzy ani ich braku. "
                        "Bez podstawy w User nie używaj pierwszoosobowych formuł typu: "
                        "„przykro mi”, „cieszę się”, „mam nadzieję”, „jestem tutaj”, "
                        "„będę tutaj”, „chętnie”, „lubię”, „uwielbiam”, „dużo słyszałem”, "
                        "„nie mam planów”. Jeśli Match pyta o nieznany fakt autobiograficzny, "
                        "nie wymyślaj odpowiedzi. Jeśli ostatnia wypowiedź Match stanowi odmowę "
                        "lub prośbę o przestrzeń, wszystkie strategie mają być krótkim "
                        "zaakceptowaniem granicy bez pytania, bez znaku zapytania, bez negocjacji "
                        "i bez próby ponownego otwarcia rozmowy. Respektowanie granic ma "
                        "pierwszeństwo przed engaging. Nie generuj presji, manipulacji ani "
                        "eskalacji seksualnej. Zwracaj wyłącznie poprawny JSON."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_object"},
        )

        raw_text = response.choices[0].message.content or ""
        suggestions = parse_model_response(raw_text)

        context_text = scenario_context_as_text(
            request.scenario_id
        )

        guardrail_trace = []

        try:
            suggestions, guardrail_trace = apply_output_guardrails(
                context_text,
                suggestions,
            )
        except GuardrailValidationError as guardrail_exc:
            guardrail_trace = guardrail_exc.trace

            log_guardrail_trace(
                guardrail_trace,
                generation_id,
                request.trial_id,
                request.participant_id,
                request.scenario_id,
            )

            raise

        if guardrail_trace:
            log_guardrail_trace(
                guardrail_trace,
                generation_id,
                request.trial_id,
                request.participant_id,
                request.scenario_id,
            )

        model_provider = "openai"
        model_name = OPENAI_MODEL

    except Exception as exc:
        api_latency_ms = int(
            (time.perf_counter() - start) * 1000
        )

        print("LLM ERROR:", repr(exc))

        log_generation_error(
            request,
            exc,
            api_latency_ms,
        )

        if EXPERIMENT_MODE:
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "generation_failed",
                    "message": (
                        "Nie udało się wygenerować sugestii. "
                        "W trybie eksperymentalnym fallback "
                        "jest wyłączony. Spróbuj ponownie."
                    ),
                    "api_error": type(exc).__name__,
                    "api_latency_ms": api_latency_ms,
                },
            )

        # Development only.
        suggestions = dev_fallback_suggestions()
        model_provider = "dev_fallback"
        model_name = "static_dev_fallback"

    api_latency_ms = int(
        (time.perf_counter() - start) * 1000
    )

    response_suggestions = []

    participant = find_row(
        PARTICIPANTS_FILE,
        "participant_id",
        request.participant_id,
    )
    study_phase = (
        participant.get("study_phase", "main")
        if participant
        else "main"
    )

    for position, item in enumerate(suggestions, start=1):
        suggestion_id = (
            f"{request.trial_id}_{generation_id}_S{position}"
        )

        append_csv_row(
            SUGGESTIONS_FILE,
            SUGGESTION_FIELDS,
            {
                "suggestion_id": suggestion_id,
                "generation_id": generation_id,
                "trial_id": request.trial_id,
                "study_phase": study_phase,
                "participant_id": request.participant_id,
                "scenario_id": request.scenario_id,
                "suggestion_position": position,
                "response_strategy": item["response_strategy"],
                "suggestion_text": item["text"],
                "was_selected": False,
                "was_edit_base": False,
                "generated_at": generated_at,
                "model_provider": model_provider,
                "model_name": model_name,
                "prompt_version": PROMPT_VERSION,
                "guidelines_version": GUIDELINES_VERSION,
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "api_latency_ms": api_latency_ms,
            },
        )

        response_suggestions.append({
            "suggestion_id": suggestion_id,
            "response_strategy": item["response_strategy"],
            "text": item["text"],
        })

    return {
        "generation_id": generation_id,
        "suggestions": response_suggestions,
        "model_provider": model_provider,
        "model_name": model_name,
        "prompt_version": PROMPT_VERSION,
        "guidelines_version": GUIDELINES_VERSION,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "api_latency_ms": api_latency_ms,
        "guardrail_triggered": bool(guardrail_trace),
        "regenerated_strategies": [
            item["response_strategy"]
            for item in guardrail_trace
            if item["repair_attempted"]
        ],
        "guardrail_version": GUARDRAIL_VERSION,
        "cached": False,
    }


# ============================================================
# ROUTES: FINAL TRIAL SAVE
# ============================================================

@app.post("/api/trials")
def save_trial(request: TrialSubmitRequest):
    participant_matches_request(
        request.participant_id,
        request.experimental_group,
        request.order_version,
    )

    validate_trial_position(
        request.order_version,
        request.trial_number,
        request.scenario_id,
    )

    if find_row(TRIALS_FILE, "trial_id", request.trial_id):
        raise HTTPException(
            status_code=409,
            detail="This trial has already been saved.",
        )

    # Also prevent the same participant from completing
    # the same scenario twice under another trial_id.
    for row in read_csv_rows(TRIALS_FILE):
        if (
            row.get("participant_id") == request.participant_id
            and row.get("scenario_id") == request.scenario_id
        ):
            raise HTTPException(
                status_code=409,
                detail=(
                    "Participant has already completed "
                    "this scenario."
                ),
            )

    assigned_condition = get_assigned_condition(
        request.experimental_group,
        request.scenario_id,
    )

    model_provider = ""
    model_name = ""
    prompt_version = ""
    guidelines_version = ""
    temperature = ""
    max_tokens = ""
    api_latency_ms = ""
    api_error = ""
    generation_status = ""
    suggestion_exposure = False

    if assigned_condition == "manual":
        if request.final_action != "manual":
            raise HTTPException(
                status_code=400,
                detail=(
                    "Manual trial must use final_action='manual'."
                ),
            )

        if request.generation_id is not None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Manual trial cannot contain generation_id."
                ),
            )

        if request.selected_suggestion_id is not None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Manual trial cannot contain "
                    "selected_suggestion_id."
                ),
            )

        if request.author_helpfulness_rating is not None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Manual trial cannot contain "
                    "author_helpfulness_rating."
                ),
            )

    else:
        suggestion_exposure = True

        if request.final_action == "manual":
            raise HTTPException(
                status_code=400,
                detail=(
                    "AI-assisted trial cannot use "
                    "final_action='manual'."
                ),
            )

        if request.generation_id is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "AI-assisted trial requires generation_id."
                ),
            )

        generation_rows = get_generation_rows(
            request.generation_id
        )

        if len(generation_rows) != 3:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Generation not found or incomplete. "
                    "A valid AI trial requires exactly "
                    "three stored suggestions."
                ),
            )

        if any(
            row.get("trial_id") != request.trial_id
            for row in generation_rows
        ):
            raise HTTPException(
                status_code=400,
                detail="Generation does not belong to this trial.",
            )

        if request.final_action in {"use", "edit"}:
            valid_ids = {
                row["suggestion_id"]
                for row in generation_rows
            }

            if request.selected_suggestion_id not in valid_ids:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "use/edit requires a valid "
                        "selected_suggestion_id."
                    ),
                )

        if request.final_action == "reject_and_write_own":
            if request.selected_suggestion_id is not None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "reject_and_write_own cannot contain "
                        "selected_suggestion_id."
                    ),
                )

        if request.author_helpfulness_rating is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "AI-assisted trial requires "
                    "author_helpfulness_rating."
                ),
            )

        first = generation_rows[0]

        model_provider = first["model_provider"]
        model_name = first["model_name"]
        prompt_version = first["prompt_version"]
        guidelines_version = first["guidelines_version"]
        temperature = first["temperature"]
        max_tokens = first["max_tokens"]
        api_latency_ms = first["api_latency_ms"]

        if EXPERIMENT_MODE and model_provider != "openai":
            raise HTTPException(
                status_code=400,
                detail=(
                    "Non-LLM fallback generations are not "
                    "valid in experiment mode."
                ),
            )

        generation_status = "success"

        update_suggestion_usage(
            request.generation_id,
            request.selected_suggestion_id,
            request.final_action,
        )

    completed_at = datetime.now(timezone.utc)

    started_at = request.started_at

    if started_at.tzinfo is None:
        started_at = started_at.replace(tzinfo=timezone.utc)

    duration_ms = max(
        0,
        int(
            (
                completed_at
                - started_at.astimezone(timezone.utc)
            ).total_seconds()
            * 1000
        ),
    )

    analysis = analyze_text(request.final_text)

    participant = find_row(
        PARTICIPANTS_FILE,
        "participant_id",
        request.participant_id,
    )
    study_phase = (
        participant.get("study_phase", "main")
        if participant
        else "main"
    )

    append_csv_row(
        TRIALS_FILE,
        TRIAL_FIELDS,
        {
            "trial_id": request.trial_id,
            "participant_id": request.participant_id,
            "study_phase": study_phase,
            "protocol_version": PROTOCOL_VERSION,
            "data_schema_version": DATA_SCHEMA_VERSION,
            "session_id": request.session_id,
            "trial_number": request.trial_number,
            "experimental_group": request.experimental_group,
            "order_version": request.order_version,
            "scenario_id": request.scenario_id,
            "assigned_condition": assigned_condition,
            "suggestion_exposure": suggestion_exposure,
            "final_action": request.final_action,
            "generation_id": request.generation_id or "",
            "selected_suggestion_id": (
                request.selected_suggestion_id or ""
            ),
            "final_text": request.final_text.strip(),
            "author_helpfulness_rating": (
                request.author_helpfulness_rating
                if request.author_helpfulness_rating is not None
                else ""
            ),
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "duration_ms": duration_ms,
            "model_provider": model_provider,
            "model_name": model_name,
            "prompt_version": prompt_version,
            "guidelines_version": guidelines_version,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "api_latency_ms": api_latency_ms,
            "api_error": api_error,
            "generation_status": generation_status,
            "word_count": analysis["word_count"],
            "char_count": analysis["char_count"],
            "question_present": analysis["question_present"],
            "emoji_count": analysis["emoji_count"],
            "sentiment": analysis["sentiment"],
            "open_question_present": analysis[
                "open_question_present"
            ],
        },
    )

    update_participant_completion(
        request.participant_id
    )

    return {
        "status": "ok",
        "trial_id": request.trial_id,
        "assigned_condition": assigned_condition,
        "analysis": analysis,
    }


# ============================================================
# ROUTES: ANALYSIS
# ============================================================

@app.post("/api/analyze")
def analyze_message(request: AnalyzeRequest):
    return analyze_text(request.text)


# ============================================================
# ROUTES: BLINDED JUDGE RATINGS
# ============================================================

@app.get("/api/judge/messages")
def get_judge_messages():
    messages = []

    for row in read_csv_rows(TRIALS_FILE):
        scenario_id = row["scenario_id"]

        messages.append({
            "trial_id": row["trial_id"],
            "scenario_id": scenario_id,
            "context": SCENARIOS[scenario_id]["context"],
            "final_text": row["final_text"],
        })

    return {
        "messages": messages,
        "note": (
            "The response intentionally excludes participant_id, "
            "assigned_condition, model data and suggestion data."
        ),
    }


@app.post("/api/judge-ratings")
def save_judge_rating(request: JudgeRatingRequest):
    if find_row(
        TRIALS_FILE,
        "trial_id",
        request.trial_id,
    ) is None:
        raise HTTPException(
            status_code=404,
            detail="Unknown trial_id.",
        )

    existing_ratings = read_csv_rows(
        JUDGE_RATINGS_FILE
    )

    for row in existing_ratings:
        if (
            row.get("judge_id") == request.judge_id
            and row.get("trial_id") == request.trial_id
        ):
            raise HTTPException(
                status_code=409,
                detail=(
                    "This judge has already rated "
                    "this trial."
                ),
            )

    rated_at = datetime.now(timezone.utc)

    started_at = request.started_at

    if started_at.tzinfo is None:
        started_at = started_at.replace(tzinfo=timezone.utc)

    rating_duration_ms = max(
        0,
        int(
            (
                rated_at
                - started_at.astimezone(timezone.utc)
            ).total_seconds()
            * 1000
        ),
    )

    rating_id = (
        f"{request.judge_id}_{request.trial_id}"
    )

    append_csv_row(
        JUDGE_RATINGS_FILE,
        JUDGE_RATING_FIELDS,
        {
            "rating_id": rating_id,
            "judge_id": request.judge_id,
            "trial_id": request.trial_id,
            "presentation_order": request.presentation_order,
            "quality_rating": request.quality_rating,
            "naturalness_rating": request.naturalness_rating,
            "attractiveness_rating": request.attractiveness_rating,
            "continuation_rating": request.continuation_rating,
            "started_at": started_at.isoformat(),
            "rated_at": rated_at.isoformat(),
            "rating_duration_ms": rating_duration_ms,
        },
    )

    return {
        "status": "ok",
        "rating_id": rating_id,
    }
