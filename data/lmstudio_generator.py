"""
Генератор ПАРНОГО корпуса названий через LM Studio.

Каждое название принадлежит сразу 2 категориям (гибридный товар, который
покупатель нашёл бы в обеих категориях — например, кокосовое масло это и
"food", и "cosmetics").

    Модель:   unsloth/gemma-4-12B-it-qat-GGUF (quant UD-Q4_K_XL)
    Эндпоинт: http://127.0.0.1:8888/v1

Использование:
    1) Запустить LM Studio, загрузить модель, включить Local Server (порт 8888)
    2) python data/lmstudio_generator.py                    # все 15 пар x 200
       python data/lmstudio_generator.py --pairs cosmetics+food,electronics+toys
       python data/lmstudio_generator.py --per-pair 300 --force

Результат: data/corpus_pairs.csv (столбцы category_a,category_b,name)
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

LMSTUDIO_BASE = "http://127.0.0.1:8888/v1"
MODEL = "unsloth/gemma-4-12B-it-qat-GGUF"

PER_PAIR = 200
BATCH_SIZE = 250  # контекст gemma ~118k: одна пара влезает в 1 запрос
MAX_RETRIES = 3
RETRY_DELAY_SEC = 5

DATA_DIR = Path(__file__).resolve().parent
BATCHES_DIR = DATA_DIR / "batches_pairs"
BATCHES_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = DATA_DIR / "corpus_pairs.csv"

# Базовые категории (6). Ключ пары — отсортированная пара через "+".
BASE_CATEGORIES: dict[str, str] = {
    "electronics": "электроника",
    "clothing": "одежда и обувь",
    "food": "продукты питания",
    "cosmetics": "косметика и парфюмерия",
    "toys": "игрушки",
    "household": "бытовая техника и товары для дома",
}

# 15 пар (C(6,2)): концепция гибрида + бренды обеих сторон + примеры.
PAIRS: dict[str, dict[str, str]] = {
    "clothing+cosmetics": {
        "concept": "товары на стыке образа и ухода: косметички, несессеры, бьюти-кейсы, повязки для умывания, халаты, дорожные наборы",
        "products": "косметички, несессеры, бьюти-кейсы, органайзеры для косметики, повязки для умывания, полотенца для лица, халаты, кисти в чехлах, дорожные наборы",
        "brands": "Zara, H&M, Mango, Guess, Samsonite, Nivea, Dove, Garnier, L'Oreal, Maybelline, Librederm, Чистая Линия, Natura Siberica, Vivienne Sabo",
        "examples": "Косметичка тканевая, Несессер дорожный, Повязка для умывания, Халат махровый, Набор кистей в чехле",
    },
    "clothing+electronics": {
        "concept": "носимая электроника и умная одежда: смарт-часы, фитнес-браслеты, куртки с подогревом, светящиеся кроссовки",
        "products": "смарт-часы, фитнес-браслеты, умные кольца, куртки с подогревом, жилеты с подогревом, кроссовки с подсветкой, наушники спортивные, пульсометры",
        "brands": "Xiaomi, Huawei, Samsung, Honor, Amazfit, Nothing, Nike, Adidas, Puma, Columbia, The North Face, JBL, Sony",
        "examples": "Xiaomi Smart Band 9, Huawei Watch Fit 3, Samsung Galaxy Watch 6, Куртка с подогревом, Кроссовки с подсветкой",
    },
    "clothing+food": {
        "concept": "одежда для кухни и фуд-мерч: фартуки, поварская форма, колпаки, футболки с принтами еды",
        "products": "фартуки кухонные, кители поварские, колпаки поварские, футболки с принтами, худи с принтами, кепки, перчатки кухонные",
        "brands": "Film, Luxary, поварская форма Pro, Coca-Cola, Mars, Nestle, Простоквашино, Чибо, McDonalds merch, Zara, H&M, Ostin",
        "examples": "Фартук кухонный льняной, Китель поварской белый, Колпак поварской, Футболка с принтом, Худи oversize",
    },
    "clothing+household": {
        "concept": "домашний текстиль и уют: халаты, тапочки, пижамы, фартуки, прихватки, пледы",
        "products": "халаты махровые, тапочки домашние, пижамы, фартуки, прихватки, полотенца банные, пледы, носки тёплые",
        "brands": "H&M Home, Zara Home, IKEA, Arya, Togas, Mona Liza, Cleanelly, Love Republic, Befree, Ostin, Gloria Jeans",
        "examples": "Халат махровый, Тапочки домашние войлочные, Пижама хлопковая, Фартук кухонный, Плед флисовый",
    },
    "clothing+toys": {
        "concept": "детская одежда и игра: карнавальные костюмы, одежда с героями, одежда для кукол, косплей",
        "products": "карнавальные костюмы, платья принцесс, костюмы супергероев, футболки с героями, пижамы-кигуруми, одежда для кукол, косплей-костюмы",
        "brands": "LEGO Wear, Disney, Marvel, Paw Patrol, Barbie, H&M Kids, Zara Kids, Gloria Jeans Kids, Acoola, Gulliver, Crockid",
        "examples": "Костюм Человека-паука, Платье Эльзы, Пижама-кигуруми, Футболка с принтом, Костюм пирата",
    },
    "cosmetics+electronics": {
        "concept": "бьюти-гаджеты: LED-маски, массажёры для лица, дарсонваль, зеркала с подсветкой, фены, стайлеры",
        "products": "LED-маски, массажёры для лица, дарсонваль, зеркала с подсветкой, фены, стайлеры, эпиляторы, щётки электрические, триммеры",
        "brands": "Dyson, Philips, Braun, Rowenta, Xiaomi, Gezatone, US Medica, Babyliss, Remington, Scarlett, Polaris, L'Oreal, Nivea",
        "examples": "Фен Dyson Supersonic, Стайлер Dyson Airwrap, Массажёр для лица, Зеркало с подсветкой, Дарсонваль",
    },
    "cosmetics+food": {
        "concept": "натуральные масла и съедобная косметика: кокосовое масло, масло ши, какао-масло, мёд, алоэ — одновременно еда и уход",
        "products": "кокосовое масло, масло ши, какао-масло, оливковое масло, мёд, алоэ гель, овсянка для ванн, травяные сборы, шоколад для массажа",
        "brands": "Parachute, Baraka, Alter Ego, СпивакЪ, Levrana, Mi&Ko, Natura Siberica, Organic Shop, Домашний мёд, Алтайский мёд",
        "examples": "Масло кокосовое нерафинированное, Масло ши для тела, Мёд липовый натуральный, Гель алоэ вера, Какао-масло",
    },
    "cosmetics+household": {
        "concept": "дом, аромат и чистота с уходом: крем-мыло, аромасвечи, диффузоры, гели для душа, освежители",
        "products": "крем-мыло, мыло ручной работы, аромасвечи, диффузоры ароматические, саше, гели для душа, пены для ванн, освежители воздуха",
        "brands": "Dove, Nivea, Palmolive, Le Petit Marseillais, СпивакЪ, Natura Siberica, Organic Shop, Air Wick, Glade, Ambi Pur, Chirton",
        "examples": "Мыло Dove крем, Диффузор ароматический, Свеча ароматическая, Гель для душа, Пена для ванн",
    },
    "cosmetics+toys": {
        "concept": "детская и игровая косметика: наборы юного визажиста, детские лаки, блески, слаймы-уход",
        "products": "наборы детской косметики, детские лаки для ногтей, блески для губ детские, шампуни детские, слаймы, бомбочки для ванн",
        "brands": "Маленькая Фея, Принцесса, Barbie, Disney, Johnson's Baby, Bubchen, Ушастый нянь, Bondibon, LORI",
        "examples": "Набор детской косметики, Лак для ногтей детский, Бомбочка для ванны, Шампунь детский, Набор юного визажиста",
    },
    "electronics+food": {
        "concept": "умная кухня: смарт-весы, Bluetooth-термометры, кофемашины с приложением, умные чайники",
        "products": "весы кухонные умные, термометры Bluetooth, кофемашины с приложением, чайники умные, мультиварки умные, диспенсеры",
        "brands": "Xiaomi, Redmond, Polaris, Kitfort, Scarlett, De'Longhi, Krups, Tefal, Moulinex, Bork, Philips",
        "examples": "Весы кухонные Xiaomi, Кофемашина De'Longhi Magnifica, Чайник Redmond SkyKettle, Термометр кухонный, Мультиварка Redmond",
    },
    "electronics+household": {
        "concept": "умный дом: роботы-пылесосы, смарт-розетки, увлажнители, климатические станции",
        "products": "роботы-пылесосы, пылесосы вертикальные, смарт-розетки, увлажнители воздуха, очистители воздуха, обогреватели умные, вентиляторы",
        "brands": "Xiaomi, Roborock, iRobot, Dyson, Karcher, Thomas, Philips, Tefal, Redmond, Polaris, Bosch, Samsung, LG",
        "examples": "Xiaomi Mi Robot Vacuum, Dyson V15 Detect, Karcher WD 3, Увлажнитель Xiaomi, Очиститель воздуха",
    },
    "electronics+toys": {
        "concept": "интерактивные игрушки и гейминг: роботы, дроны, консоли, конструкторы с приложением",
        "products": "роботы игрушечные, дроны, консоли игровые, геймпады, конструкторы программируемые, машинки на радиоуправлении, VR-очки",
        "brands": "LEGO, DJI, Sony, Nintendo, Xbox, Hasbro, Mattel, Spin Master, VTech, HIPER, SYMA, Bondibon",
        "examples": "LEGO Technic набор, Дрон DJI Tello, Sony PlayStation 5, Робот-щенок, Машинка на радиоуправлении",
    },
    "food+household": {
        "concept": "кухня и хранение: посуда, контейнеры, сковороды, специи, крупы, диспенсеры",
        "products": "сковороды, кастрюли, контейнеры для хранения, крупы, специи, чаи, кофе, диспенсеры для круп, формы для выпечки",
        "brands": "Tefal, Rondell, Gipfel, IKEA, Luminarc, Мистраль, Увелка, Greenfield, Tess, Ahmad Tea, Jacobs, Nescafe, Kotanyi",
        "examples": "Сковорода Tefal Titanium, Кастрюля Rondell, Контейнер для хранения, Крупа гречневая, Чай Greenfield",
    },
    "food+toys": {
        "concept": "игра и вкус: игровые кухни, сладости-игрушки, наборы маленького повара",
        "products": "кухни детские игровые, наборы маленького повара, сладости с игрушкой, шоколадные яйца, наборы для лепки еды, кассы игрушечные",
        "brands": "Kinder, Chupa Chups, Полесье, Smoby, Step2, Bondibon, Play-Doh, Красный Октябрь, Бабаевский, Сладкая сказка",
        "examples": "Kinder Сюрприз, Кухня детская игровая, Набор маленького повара, Шоколад Алёнка, Леденец на палочке",
    },
    "household+toys": {
        "concept": "детский дом: палатки, манежи, бизиборды, игрушечная техника, детская мебель",
        "products": "палатки детские, манежи, бизиборды, пылесосы игрушечные, кухни игрушечные, столы детские, стульчики, мольберты",
        "brands": "Smoby, Step2, Полесье, IKEA Kids, Bondibon, Fisher-Price, Chicco, Happy Baby, VTech, Alatoys",
        "examples": "Палатка детская игровая, Бизиборд деревянный, Манеж детский, Столик детский, Пылесос игрушечный",
    },
}

SYSTEM_PROMPT = """Ты эксперт по реалистичным названиям гибридных товаров для российского маркетплейса.

СТРОГИЕ правила:
- Только короткие названия: 2-4 слова, как в реальных карточках товаров
- Товар ОДНОВРЕМЕННО относится к обеим категориям (его искали бы и там, и там)
- Формат: Бренд + Модель/Тип + (опционально) Характеристика/Объём
- Примеры хороших названий: "Масло кокосовое Parachute 200мл", "Xiaomi Smart Band 9", "Фен Dyson Supersonic"
- ЗАПРЕЩЕНО: маркетинговые описания ("Ultra Thin", "Slimmest", "Selfie Star", "Premium Edition" в конце)
- ЗАПРЕЩЕНО: скобки, кавычки, эмодзи, markdown, пояснения
- ЗАПРЕЩЕНО: нереальные сочетания брендов и продуктов
- ЗАПРЕЩЕНО: писать слова "товар", "продукт", "модель", "device", "edition"
- Каждое название на отдельной строке, БЕЗ нумерации
- ТОЛЬКО сами названия, без преамбул и заключений
- Не повторяй названия внутри списка
- Бренды пиши как они пишутся (Samsung, Xiaomi, Nivea, L'Oreal)
- Разнообразь: разные модели, цвета, объёмы, линейки, размеры"""

USER_PROMPT_TEMPLATE = """Сгенерируй РОВНО {count} уникальных реалистичных названий товаров,
которые ОДНОВРЕМЕННО относятся к категориям "{cat_a}" и "{cat_b}".

Концепция стыка: {concept}

Используй бренды: {brands}

Типы товаров на стыке: {products}

Примеры хороших названий: {examples}

ВАЖНО:
- РОВНО {count} названий
- Каждое название на ОТДЕЛЬНОЙ строке
- БЕЗ нумерации, БЕЗ маркеров, БЕЗ пояснений
- Только сами названия, без преамбул и заключений
- Все названия должны быть РАЗНЫМИ
- Равномерно используй РАЗНЫЕ бренды из списка — не зацикливайся на 2-3 брендах
- Варьируй типы товаров из списка, а не только самый очевидный
- Каждое название должно подходить ОБЕИМ категориям сразу
- Названия должны выглядеть как настоящие карточки маркетплейса"""


def _url(path: str) -> str:
    return f"{LMSTUDIO_BASE.rstrip('/')}{path}"


def list_models() -> list[dict]:
    req = urllib.request.Request(_url("/models"))
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("data", [])


def resolve_model(wanted: str) -> str:
    """Pick the model id to use: exact match, else prefix match (quant suffix)."""
    models = list_models()
    ids = [m.get("id", "") for m in models]
    if wanted in ids:
        return wanted
    base = wanted.split(":")[0]
    for mid in ids:
        if mid == base or mid.startswith(base):
            print(f"Model {wanted!r} not found, using loaded {mid!r}.")
            return mid
    loaded = [m.get("id", "") for m in models if m.get("loaded")]
    if len(loaded) == 1:
        print(f"Model {wanted!r} not found, using the only loaded model {loaded[0]!r}.")
        return loaded[0]
    raise SystemExit(
        f"Model {wanted!r} not found. Available: {ids}. "
        f"Load it in LM Studio (Local Server, port 8888) or pass --model."
    )


def check_lmstudio(model: str) -> str:
    try:
        resolved = resolve_model(model)
    except Exception as e:
        print(f"LM Studio недоступен: {e}")
        print("Убедись, что LM Studio запущен и Local Server включён (порт 8888).")
        raise SystemExit(1)
    print(f"LM Studio OK. Модель: {resolved}")
    return resolved


def call_lmstudio(prompt: str, system: str, model: str, max_tokens: int = 12000) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.9,
        "top_p": 0.95,
        "max_tokens": max_tokens,
        "stream": False,
    }
    req = urllib.request.Request(
        _url("/chat/completions"),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1200) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result["choices"][0]["message"]["content"]


def clean_line(line: str) -> str:
    line = line.strip()
    line = re.sub(r"^\*\*+|\*\*+$", "", line)
    line = re.sub(r"^\*+|\*+$", "", line)
    line = re.sub(r"^`+|`+$", "", line)
    line = re.sub(r"^#+", "", line)
    line = re.sub(r"^\d+[\.\)]\s*", "", line)
    line = re.sub(r"^[-•·—–]\s*", "", line)
    return line.strip().strip('"').strip("'").strip()


def is_valid_name(name: str) -> bool:
    if not name or len(name) < 5 or len(name) > 80:
        return False
    if "�" in name:  # битая кодировка — сразу в брак
        return False
    words = name.split()
    if len(words) < 2 or len(words) > 6:
        return False
    bad_markers = ["```", "##", "==", "function", "def ", "import ", "class ",
                   "{", "}", "<", ">", "TODO", "FIXME", "[", "]"]
    if any(b in name for b in bad_markers):
        return False
    bad_adjectives = [
        "ultra thin", "slimmest", "selfie star", "health focus",
        "compact edition", "premium edition", "professional studio",
        "mobile workstation", "edition", "топовый", "лучший выбор",
        "premium quality", "original", "новинка", "хит продаж",
    ]
    name_lower = name.lower()
    if any(p in name_lower for p in bad_adjectives):
        return False
    placeholders = ["название товара", "пример", "sample", "test123", "lorem",
                    "ipsum", "товар ", "product ", "item "]
    if any(p in name_lower for p in placeholders):
        return False
    if not re.search(r"[a-zA-Zа-яА-ЯёЁ]", name):
        return False
    if re.search(r"\b[a-z]{5,}\s+[a-z]{5,}\s+[a-z]{5,}", name_lower):
        return False
    return True


def parse_names(text: str) -> list[str]:
    names: list[str] = []
    for line in text.split("\n"):
        cleaned = clean_line(line)
        if is_valid_name(cleaned):
            names.append(cleaned)
    return names


def dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def generate_batch(
    pair_key: str, batch_idx: int, target_count: int, model: str, force: bool = False,
) -> list[str]:
    raw_path = BATCHES_DIR / f"{pair_key}_{batch_idx}_raw.txt"
    clean_path = BATCHES_DIR / f"{pair_key}_{batch_idx}_clean.txt"

    if clean_path.exists() and not force:
        cached = [n for n in clean_path.read_text(encoding="utf-8").splitlines() if n]
        print(f"    [cached] {len(cached)} names")
        return cached

    if raw_path.exists() and not force:
        print(f"    [reparsing from raw] {raw_path.name}")
        text = raw_path.read_text(encoding="utf-8")
        names = dedupe_preserve_order([n for n in parse_names(text) if is_valid_name(n)])
        print(f"    Reparsed: {len(names)} valid unique names")
        clean_path.write_text("\n".join(names), encoding="utf-8")
        return names

    cat_a, cat_b = pair_key.split("+")
    spec = PAIRS[pair_key]
    prompt = USER_PROMPT_TEMPLATE.format(
        count=target_count,
        cat_a=BASE_CATEGORIES[cat_a],
        cat_b=BASE_CATEGORIES[cat_b],
        concept=spec["concept"],
        brands=spec["brands"],
        products=spec["products"],
        examples=spec["examples"],
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"    Calling LM Studio (attempt {attempt}/{MAX_RETRIES})...")
            t0 = time.time()
            text = call_lmstudio(prompt, SYSTEM_PROMPT, model)
            elapsed = time.time() - t0
            raw_path.write_text(text, encoding="utf-8")

            names = dedupe_preserve_order([n for n in parse_names(text) if is_valid_name(n)])
            print(f"    Response in {elapsed:.1f}s, {len(names)} valid unique names")

            if len(names) < int(target_count * 0.6):
                print("    Too few names, retrying...")
                time.sleep(RETRY_DELAY_SEC)
                continue

            clean_path.write_text("\n".join(names), encoding="utf-8")
            return names
        except Exception as e:
            print(f"    Error: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SEC)
    return []


def generate_pair(
    pair_key: str, per_pair: int, batch_size: int, model: str, force: bool = False,
) -> list[str]:
    cat_a, cat_b = pair_key.split("+")
    print(f"\n[{pair_key}] {BASE_CATEGORIES[cat_a]} + {BASE_CATEGORIES[cat_b]}")
    pair_names: list[str] = []
    batch_idx = 0
    max_batches = (per_pair + batch_size - 1) // batch_size + 1
    while len(pair_names) < per_pair and batch_idx < max_batches:
        remaining = per_pair - len(pair_names)
        target = min(batch_size, remaining + 30)
        new_names = generate_batch(pair_key, batch_idx, target, model, force)
        pair_names.extend(new_names)
        pair_names = dedupe_preserve_order(pair_names)
        print(f"    Прогресс: {len(pair_names)}/{per_pair}")
        batch_idx += 1
    return pair_names[:per_pair]


def save_corpus(rows: list[dict[str, str]], out: Path) -> None:
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["category_a", "category_b", "name"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    global LMSTUDIO_BASE, OUTPUT_FILE
    parser = argparse.ArgumentParser(description="Генератор парного корпуса через LM Studio")
    parser.add_argument("--pairs", default="all",
                        help="Пары через запятую (напр. cosmetics+food) или 'all'")
    parser.add_argument("--per-pair", type=int, default=PER_PAIR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--base-url", default=LMSTUDIO_BASE)
    parser.add_argument("--out", default=str(OUTPUT_FILE))
    parser.add_argument("--force", action="store_true",
                        help="Игнорировать кэш и запросить всё заново")
    args = parser.parse_args()

    LMSTUDIO_BASE = args.base_url
    OUTPUT_FILE = Path(args.out)

    if args.pairs == "all":
        pair_keys = sorted(PAIRS.keys())
    else:
        pair_keys = [p.strip() for p in args.pairs.split(",") if p.strip()]
        unknown = [p for p in pair_keys if p not in PAIRS]
        if unknown:
            raise SystemExit(f"Unknown pairs: {unknown}. Available: {sorted(PAIRS)}")

    print("=" * 60)
    print("LM Studio Pair Generator")
    print(f"Model: {args.model}")
    print(f"Target: {len(pair_keys)} pairs x {args.per_pair} = "
          f"{len(pair_keys) * args.per_pair} names")
    print(f"Batch size: {args.batch_size}")
    print("=" * 60)

    model = check_lmstudio(args.model)

    all_rows: list[dict[str, str]] = []
    for pair_key in pair_keys:
        cat_a, cat_b = pair_key.split("+")
        names = generate_pair(pair_key, args.per_pair, args.batch_size, model, args.force)
        for n in names:
            all_rows.append({"category_a": cat_a, "category_b": cat_b, "name": n})

    seen: set[str] = set()
    final: list[dict[str, str]] = []
    for r in all_rows:
        key = r["name"].lower()
        if key not in seen:
            seen.add(key)
            final.append(r)

    save_corpus(final, OUTPUT_FILE)

    from collections import Counter
    counter = Counter(f"{r['category_a']}+{r['category_b']}" for r in final)
    print("\n" + "=" * 60)
    print(f"Saved: {OUTPUT_FILE}")
    print(f"Total: {len(final)}")
    print("By pair:")
    for pair, n in counter.most_common():
        print(f"  {pair:<24} {n}")
    lens = [len(r["name"].split()) for r in final]
    print(f"Length (words): min={min(lens)} avg={sum(lens)/len(lens):.2f} max={max(lens)}")
    print("=" * 60)

    print("\nSamples (3 per pair):")
    by_pair: dict[str, list[str]] = {}
    for r in final:
        by_pair.setdefault(f"{r['category_a']}+{r['category_b']}", []).append(r["name"])
    for pair, names in sorted(by_pair.items()):
        print(f"\n[{pair}]")
        for n in names[:3]:
            print(f"  - {n}")


if __name__ == "__main__":
    main()
