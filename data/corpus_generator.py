"""
Генератор синтетического корпуса названий товаров.

Архитектура:
- Каждая категория содержит несколько ШАБЛОНОВ (templates)
- Каждый шаблон имеет СОБСТВЕННЫЕ словари для каждого слота
- BRAND_LINE_COMPAT гарантирует, что линейка продукта соответствует бренду
- Post-фильтр удаляет плохие сочетания и абсурдные объёмы
"""
from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent
CATEGORIES_FILE = DATA_DIR / "categories.json"
CORPUS_FILE = DATA_DIR / "corpus.csv"

SEED = 42
TOTAL_NAMES = 3000
N_CATEGORIES = 6
PER_CATEGORY = TOTAL_NAMES // N_CATEGORIES  # 500

BRAND_PRODUCT_COMPAT: dict[str, dict[str, list[str]]] = {
    "electronics": {
        "beats": ["наушники", "колонка", "гарнитура", "акустика", "микрофон", "вкладыши"],
        "bose": ["наушники", "колонка", "гарнитура", "акустика", "микрофон", "сабвуфер", "вкладыши"],
        "jbl": ["наушники", "колонка", "гарнитура", "акустика", "микрофон", "сабвуфер", "вкладыши"],
        "marshall": ["наушники", "колонка", "акустика", "сабвуфер", "вкладыши"],
        "steelseries": ["мышь", "клавиатура", "гарнитура", "наушники", "геймпад", "микрофон", "вкладыши"],
        "razer": ["мышь", "клавиатура", "гарнитура", "наушники", "геймпад", "микрофон", "вкладыши"],
        "logitech": ["мышь", "клавиатура", "гарнитура", "наушники", "геймпад", "микрофон", "веб-камера", "колонка", "вкладыши"],
        "hyperx": ["мышь", "клавиатура", "гарнитура", "наушники", "геймпад", "микрофон", "вкладыши"],
        "defender": ["мышь", "клавиатура", "гарнитура", "наушники", "геймпад", "микрофон", "колонка", "веб-камера", "вкладыши"],
        "sven": ["мышь", "клавиатура", "гарнитура", "наушники", "геймпад", "микрофон", "веб-камера", "колонка", "вкладыши"],
        "lenovo": ["ноутбук", "планшет", "монитор", "мышь", "клавиатура", "гарнитура"],
        "hp": ["ноутбук", "планшет", "монитор", "мышь", "клавиатура", "гарнитура", "принтер", "сканер"],
        "dell": ["ноутбук", "планшет", "монитор", "мышь", "клавиатура", "гарнитура", "принтер", "сканер"],
        "asus": ["ноутбук", "планшет", "монитор", "мышь", "клавиатура", "гарнитура"],
        "msi": ["ноутбук", "монитор", "мышь", "клавиатура", "гарнитура"],
        "acer": ["ноутбук", "планшет", "монитор", "мышь", "клавиатура", "гарнитура"],
        "toshiba": ["телевизор", "ноутбук", "планшет", "HDD", "SSD", "флешка"],
        "canon": ["камера", "фотоаппарат", "принтер", "сканер", "объектив"],
        "nikon": ["камера", "фотоаппарат", "объектив"],
        "gopro": ["камера", "фотоаппарат", "крепление", "аккумулятор"],
        "pioneer": ["колонка", "наушники", "акустика", "микрофон", "усилитель", "вкладыши"],
        "roborock": ["робот-пылесос", "пылесос"],
        "irobot": ["робот-пылесос"],
    },
    "household": {
        "dyson": ["пылесос", "робот-пылесос", "фен", "вентилятор", "увлажнитель", "очиститель", "обогреватель", "сушилка"],
        "karcher": ["пылесос", "робот-пылесос", "очиститель", "пароочиститель", "мойка", "насос"],
        "thomas": ["пылесос", "робот-пылесос", "очиститель", "мойка"],
        "irobot": ["робот-пылесос"],
        "roborock": ["робот-пылесос", "пылесос"],
        "whirlpool": ["стиральная машина", "посудомоечная машина", "холодильник", "морозильник", "микроволновка", "духовка", "варочная панель", "вытяжка", "сушильная машина"],
        "indesit": ["стиральная машина", "посудомоечная машина", "холодильник", "морозильник", "микроволновка", "духовка", "варочная панель", "вытяжка"],
        "ariston": ["стиральная машина", "посудомоечная машина", "холодильник", "морозильник", "микроволновка", "духовка", "варочная панель", "вытяжка", "водонагреватель"],
        "zanussi": ["стиральная машина", "посудомоечная машина", "холодильник", "морозильник", "микроволновка", "духовка", "варочная панель", "вытяжка", "пылесос"],
        "beko": ["стиральная машина", "посудомоечная машина", "холодильник", "морозильник", "микроволновка", "духовка", "варочная панель", "вытяжка", "сушильная машина"],
        "gorenje": ["стиральная машина", "посудомоечная машина", "холодильник", "морозильник", "микроволновка", "духовка", "варочная панель", "вытяжка"],
        "haier": ["стиральная машина", "посудомоечная машина", "холодильник", "морозильник", "микроволновка", "духовка", "варочная панель", "вытяжка", "кондиционер", "телевизор"],
        "miele": ["пылесос", "стиральная машина", "посудомоечная машина", "холодильник", "морозильник", "микроволновка", "духовка", "варочная панель", "вытяжка", "кофемашина", "фен", "утюг", "парогенератор", "робот-пылесос"],
        "de'longhi": ["кофемашина", "кофеварка", "чайник", "блендер", "миксер", "тостер", "гриль", "аэрогриль", "мультиварка", "хлебопечка", "пароварка", "утюг", "фен", "обогреватель", "вентилятор", "увлажнитель", "весы", "кондиционер", "сковорода", "кастрюля", "турка"],
        "krups": ["кофемашина", "кофеварка", "чайник", "блендер", "миксер", "мясорубка", "соковыжималка", "тостер"],
        "ikea": ["сковорода", "кастрюля", "сотейник", "противень", "форма для выпечки", "тарелка", "миска", "кружка", "чашка", "салатник", "хлебница", "разделочная доска", "подставка", "контейнер", "чайник заварочный", "турка", "френч-пресс", "кастрюля-скороварка", "термос", "весы", "утюг", "гладильная доска", "пылесос", "обогреватель"],
    },
}

PRODUCT_NOUNS: dict[str, set[str]] = {
    "electronics": {
        "смартфон", "телефон", "ноутбук", "ультрабук", "планшет", "наушники", "гарнитура",
        "колонка", "сабвуфер", "телевизор", "монитор", "камера", "фотоаппарат",
        "мышь", "клавиатура", "геймпад", "роутер", "модем", "SSD", "HDD", "флешка",
        "аккумулятор", "зарядка", "адаптер", "чехол", "кабель", "штатив", "трипод",
        "микрофон", "веб-камера", "плеер", "часы", "браслет", "принтер", "сканер",
        "вкладыши", "крепление", "объектив", "усилитель", "акустика",
    },
    "household": {
        "пылесос", "робот-пылесос", "стиральная машина", "посудомоечная машина",
        "холодильник", "морозильник", "микроволновка", "духовка", "варочная панель",
        "вытяжка", "кофемашина", "кофеварка", "блендер", "миксер", "мясорубка",
        "соковыжималка", "тостер", "гриль", "аэрогриль", "мультиварка", "хлебопечка",
        "пароварка", "утюг", "парогенератор", "фен", "машинка для стрижки", "эпилятор",
        "бритва", "триммер", "обогреватель", "вентилятор", "увлажнитель", "чайник",
        "термос", "сковорода", "кастрюля", "сотейник", "казан", "противень",
        "форма для выпечки", "тарелка", "миска", "кружка", "чашка", "салатник",
        "хлебница", "разделочная доска", "подставка", "контейнер", "чайник заварочный",
        "турка", "френч-пресс", "кастрюля-скороварка", "гладильная доска", "весы",
        "очиститель", "пароочиститель", "мойка", "насос", "кондиционер", "водонагреватель",
        "сушильная машина", "электрическая зубная щетка", "сушилка",
    },
}


def is_brand_product_incompatible(name: str, category: str) -> bool:
    name_lower = name.lower()
    compat = BRAND_PRODUCT_COMPAT.get(category, {})
    nouns = PRODUCT_NOUNS.get(category, set())
    name_nouns = [n for n in name_lower.split() if n in nouns]
    if not name_nouns:
        return False
    for brand, valid_products in compat.items():
        if brand in name_lower:
            if not any(noun in valid_products for noun in name_nouns):
                return True
    return False


BRAND_LINE_COMPAT: dict[str, dict[str, list[str]]] = {
    "electronics": {
        "iphone": ["apple"],
        "ipad": ["apple"],
        "macbook": ["apple"],
        "airpods": ["apple"],
        "watch": ["apple"],
        "imac": ["apple"],
        "mac mini": ["apple"],
        "homepod": ["apple"],
        "galaxy": ["samsung"],
        "bespoke": ["samsung"],
        "the frame": ["samsung"],
        "odyssey": ["samsung"],
        "redmi": ["xiaomi"],
        "poco": ["xiaomi"],
        "black shark": ["xiaomi"],
        "xperia": ["sony"],
        "bravia": ["sony"],
        "alpha": ["sony"],
        "playstation": ["sony"],
        "oled": ["lg"],
        "nanocell": ["lg"],
        "qned": ["lg"],
        "gram": ["lg"],
        "ultragear": ["lg"],
        "ultrafine": ["lg"],
        "zenbook": ["asus"],
        "vivobook": ["asus"],
        "rog": ["asus"],
        "tuf": ["asus"],
        "expertbook": ["asus"],
        "proart": ["asus"],
        "thinkpad": ["lenovo"],
        "ideapad": ["lenovo"],
        "legion": ["lenovo"],
        "yoga": ["lenovo"],
        "thinkbook": ["lenovo"],
        "pavilion": ["hp"],
        "envy": ["hp"],
        "spectre": ["hp"],
        "omen": ["hp"],
        "probook": ["hp"],
        "elitebook": ["hp"],
        "xps": ["dell"],
        "inspiron": ["dell"],
        "latitude": ["dell"],
        "vostro": ["dell"],
        "alienware": ["dell"],
        "mate": ["huawei"],
        "pura": ["huawei"],
        "nova": ["huawei"],
        "freebuds": ["huawei"],
        "magic": ["honor"],
        "raider": ["msi"],
        "stealth": ["msi"],
        "titan": ["msi"],
        "katana": ["msi"],
        "prestige": ["msi"],
        "modern": ["msi"],
        "blade": ["razer"],
        "huntsman": ["razer"],
        "deathadder": ["razer"],
        "basilisk": ["razer"],
        "aspire": ["acer"],
        "swift": ["acer"],
        "predator": ["acer"],
        "nitro": ["acer"],
        "pixel": ["google"],
        "nord": ["oneplus"],
        "find": ["oppo"],
    },
    "toys": {
        "star wars": ["lego", "hasbro", "mattel", "trefl", "ravensburger", "spin master", "hot wheels"],
        "marvel": ["lego", "hasbro", "mattel", "trefl", "ravensburger", "hot wheels"],
        "disney": ["lego", "hasbro", "mattel", "trefl", "ravensburger", "hot wheels"],
        "frozen": ["lego", "hasbro", "mattel"],
        "toy story": ["lego", "hasbro", "mattel"],
        "cars": ["mattel", "hot wheels", "lego"],
        "city": ["lego"],
        "friends": ["lego"],
        "technic": ["lego"],
        "creator": ["lego"],
        "duplo": ["lego"],
        "harry potter": ["lego"],
        "batman": ["lego", "mattel"],
        "minecraft": ["lego"],
        "mario": ["lego"],
        "transformers": ["hasbro"],
        "my little pony": ["hasbro"],
        "nerf": ["hasbro"],
        "monopoly": ["hasbro"],
        "barbie": ["mattel"],
        "hot wheels": ["mattel"],
        "monster high": ["mattel"],
        "paw patrol": ["spin master", "trefl", "ravensburger"],
        "winx": ["hasbro"],
    },
}

BAD_SUBSTRINGS: list[tuple[str, str]] = [
    ("ferrero", "молоко"),
    ("ferrero", "кефир"),
    ("ferrero", "творог"),
    ("ferrero", "сметана"),
    ("ferrero", "йогурт"),
    ("ferrero", "масло"),
    ("ferrero", "сливки"),
    ("ferrero", "ряженка"),
    ("nestle", "молоко"),
    ("nestle", "кефир"),
    ("nestle", "творог"),
    ("alpen gold", "молоко"),
    ("alpen gold", "кефир"),
    ("alpen gold", "творог"),
    ("milka", "молоко"),
    ("milka", "творог"),
    ("milka", "кефир"),
    ("snickers", "молоко"),
    ("snickers", "творог"),
    ("snickers", "кефир"),
    ("mars", "молоко"),
    ("twix", "молоко"),
    ("bounty", "молоко"),
    ("kitkat", "молоко"),
    ("ferrero", "суп"),
    ("ferrero", "каша"),
    ("ferrero", "хлеб"),
    ("ferrero", "мясо"),
    ("ikea", "смартфон"),
    ("ikea", "наушники"),
    ("ikea", "телевизор"),
    ("ikea", "планшет"),
    ("ikea", "ноутбук"),
    ("ikea", "фотоаппарат"),
    ("nike", "смартфон"),
    ("nike", "телевизор"),
    ("nike", "наушники"),
    ("nike", "холодильник"),
    ("nike", "чайник"),
    ("nike", "планшет"),
    ("nike", "микроволновка"),
    ("nike", "помада"),
    ("nike", "шампунь"),
    ("nike", "крем"),
    ("nike", "сыр"),
    ("nike", "молоко"),
    ("nike", "масло"),
    ("nike", "торт"),
    ("nike", "конфеты"),
    ("nike", "печенье"),
    ("nike", "хлеб"),
    ("adidas", "смартфон"),
    ("adidas", "телевизор"),
    ("adidas", "наушники"),
    ("adidas", "холодильник"),
    ("adidas", "планшет"),
    ("adidas", "чайник"),
    ("adidas", "помада"),
    ("adidas", "шампунь"),
    ("adidas", "крем"),
    ("adidas", "сыр"),
    ("adidas", "молоко"),
    ("adidas", "масло"),
    ("adidas", "торт"),
    ("adidas", "конфеты"),
    ("adidas", "печенье"),
    ("adidas", "хлеб"),
    ("lego", "молоко"),
    ("lego", "хлеб"),
    ("lego", "шампунь"),
    ("lego", "помада"),
    ("lego", "смартфон"),
    ("lego", "телевизор"),
    ("lego", "чайник"),
    ("lego", "холодильник"),
    ("lego", "крем"),
    ("lego", "сыр"),
    ("hasbro", "молоко"),
    ("hasbro", "хлеб"),
    ("hasbro", "шампунь"),
    ("hasbro", "помада"),
    ("hasbro", "смартфон"),
    ("hasbro", "холодильник"),
    ("hasbro", "чайник"),
    ("hasbro", "крем"),
    ("hasbro", "сыр"),
    ("mattel", "молоко"),
    ("mattel", "хлеб"),
    ("mattel", "шампунь"),
    ("mattel", "помада"),
    ("mattel", "смартфон"),
    ("mattel", "холодильник"),
    ("mattel", "чайник"),
    ("mattel", "крем"),
    ("mattel", "сыр"),
    ("bosch", "помада"),
    ("bosch", "шампунь"),
    ("bosch", "крем"),
    ("bosch", "молоко"),
    ("bosch", "сыр"),
    ("bosch", "хлеб"),
    ("bosch", "футболка"),
    ("bosch", "джинсы"),
    ("bosch", "кроссовки"),
    ("bosch", "платье"),
    ("philips", "помада"),
    ("philips", "шампунь"),
    ("philips", "крем"),
    ("philips", "молоко"),
    ("philips", "сыр"),
    ("philips", "хлеб"),
    ("philips", "футболка"),
    ("philips", "джинсы"),
    ("philips", "платье"),
    ("tefal", "помада"),
    ("tefal", "шампунь"),
    ("tefal", "крем"),
    ("tefal", "молоко"),
    ("tefal", "сыр"),
    ("tefal", "хлеб"),
    ("tefal", "футболка"),
    ("tefal", "джинсы"),
    ("tefal", "платье"),
    ("nivea", "смартфон"),
    ("nivea", "телевизор"),
    ("nivea", "наушники"),
    ("nivea", "холодильник"),
    ("nivea", "футболка"),
    ("nivea", "джинсы"),
    ("nivea", "кроссовки"),
    ("nivea", "платье"),
    ("nivea", "молоко"),
    ("nivea", "хлеб"),
    ("nivea", "сыр"),
    ("l'oreal", "смартфон"),
    ("l'oreal", "телевизор"),
    ("l'oreal", "наушники"),
    ("l'oreal", "холодильник"),
    ("l'oreal", "футболка"),
    ("l'oreal", "джинсы"),
    ("l'oreal", "молоко"),
    ("l'oreal", "хлеб"),
    ("l'oreal", "сыр"),
    ("l'oreal", "масло"),
    ("maybelline", "смартфон"),
    ("maybelline", "телевизор"),
    ("maybelline", "наушники"),
    ("maybelline", "холодильник"),
    ("maybelline", "футболка"),
    ("maybelline", "джинсы"),
    ("maybelline", "молоко"),
    ("maybelline", "хлеб"),
    ("maybelline", "сыр"),
    ("dove", "смартфон"),
    ("dove", "телевизор"),
    ("dove", "наушники"),
    ("dove", "холодильник"),
    ("dove", "футболка"),
    ("dove", "джинсы"),
    ("dove", "молоко"),
    ("dove", "хлеб"),
    ("dove", "сыр"),
    ("простоквашино", "смартфон"),
    ("простоквашино", "наушники"),
    ("простоквашино", "футболка"),
    ("простоквашино", "кроссовки"),
    ("простоквашино", "шампунь"),
    ("простоквашино", "помада"),
    ("простоквашино", "холодильник"),
    ("heinz", "смартфон"),
    ("heinz", "наушники"),
    ("heinz", "футболка"),
    ("heinz", "кроссовки"),
    ("heinz", "шампунь"),
    ("heinz", "помада"),
    ("heinz", "холодильник"),
    ("heinz", "молоко"),
    ("heinz", "хлеб"),
    ("heinz", "сыр"),
    ("coca-cola", "смартфон"),
    ("coca-cola", "футболка"),
    ("coca-cola", "шампунь"),
    ("coca-cola", "холодильник"),
    ("coca-cola", "молоко"),
    ("coca-cola", "хлеб"),
    ("coca-cola", "сыр"),
    ("pepsi", "смартфон"),
    ("pepsi", "футболка"),
    ("pepsi", "шампунь"),
    ("pepsi", "молоко"),
    ("pepsi", "хлеб"),
    ("valio", "смартфон"),
    ("valio", "наушники"),
    ("valio", "футболка"),
    ("valio", "шампунь"),
    ("valio", "помада"),
    ("valio", "холодильник"),
    ("calvin klein", "молоко"),
    ("calvin klein", "хлеб"),
    ("calvin klein", "сыр"),
    ("calvin klein", "шампунь"),
    ("calvin klein", "смартфон"),
    ("tommy hilfiger", "молоко"),
    ("tommy hilfiger", "хлеб"),
    ("tommy hilfiger", "сыр"),
    ("tommy hilfiger", "шампунь"),
    ("tommy hilfiger", "смартфон"),
    ("lacoste", "молоко"),
    ("lacoste", "хлеб"),
    ("lacoste", "сыр"),
    ("lacoste", "шампунь"),
    ("lacoste", "смартфон"),
    ("columbia", "молоко"),
    ("columbia", "хлеб"),
    ("columbia", "сыр"),
    ("columbia", "шампунь"),
    ("columbia", "смартфон"),
    ("columbia", "помада"),
    ("the north face", "молоко"),
    ("the north face", "хлеб"),
    ("the north face", "сыр"),
    ("the north face", "шампунь"),
    ("the north face", "смартфон"),
    ("the north face", "помада"),
    ("patagonia", "молоко"),
    ("patagonia", "хлеб"),
    ("patagonia", "сыр"),
    ("patagonia", "шампунь"),
    ("patagonia", "смартфон"),
    ("patagonia", "помада"),
    ("красный октябрь", "смартфон"),
    ("красный октябрь", "наушники"),
    ("красный октябрь", "холодильник"),
    ("красный октябрь", "футболка"),
    ("красный октябрь", "шампунь"),
    ("красный октябрь", "помада"),
    ("красный октябрь", "хлеб"),
    ("красный октябрь", "молоко"),
    ("красный октябрь", "сыр"),
    ("бабаевский", "смартфон"),
    ("бабаевский", "наушники"),
    ("бабаевский", "холодильник"),
    ("бабаевский", "футболка"),
    ("бабаевский", "шампунь"),
    ("бабаевский", "помада"),
    ("бабаевский", "молоко"),
    ("бабаевский", "хлеб"),
    ("бабаевский", "сыр"),
    ("samsung", "футболка"),
    ("samsung", "джинсы"),
    ("samsung", "платье"),
    ("samsung", "кроссовки"),
    ("samsung", "молоко"),
    ("samsung", "хлеб"),
    ("samsung", "сыр"),
    ("samsung", "помада"),
    ("samsung", "шампунь"),
    ("samsung", "крем"),
    ("lg", "футболка"),
    ("lg", "джинсы"),
    ("lg", "платье"),
    ("lg", "кроссовки"),
    ("lg", "молоко"),
    ("lg", "хлеб"),
    ("lg", "сыр"),
    ("lg", "помада"),
    ("lg", "шампунь"),
    ("lg", "крем"),
    ("apple", "футболка"),
    ("apple", "джинсы"),
    ("apple", "платье"),
    ("apple", "кроссовки"),
    ("apple", "молоко"),
    ("apple", "хлеб"),
    ("apple", "сыр"),
    ("apple", "помада"),
    ("apple", "шампунь"),
    ("apple", "крем"),
    ("xiaomi", "футболка"),
    ("xiaomi", "джинсы"),
    ("xiaomi", "платье"),
    ("xiaomi", "кроссовки"),
    ("xiaomi", "молоко"),
    ("xiaomi", "хлеб"),
    ("xiaomi", "сыр"),
    ("xiaomi", "помада"),
    ("xiaomi", "шампунь"),
    ("xiaomi", "крем"),
    ("playmobil", "молоко"),
    ("playmobil", "хлеб"),
    ("playmobil", "сыр"),
    ("playmobil", "шампунь"),
    ("playmobil", "помада"),
    ("playmobil", "смартфон"),
    ("playmobil", "холодильник"),
    ("playmobil", "чайник"),
    ("hot wheels", "молоко"),
    ("hot wheels", "хлеб"),
    ("hot wheels", "сыр"),
    ("hot wheels", "шампунь"),
    ("hot wheels", "помада"),
    ("hot wheels", "смартфон"),
    ("hot wheels", "холодильник"),
    ("hot wheels", "чайник"),
    ("fisher-price", "молоко"),
    ("fisher-price", "хлеб"),
    ("fisher-price", "сыр"),
    ("fisher-price", "шампунь"),
    ("fisher-price", "помада"),
    ("fisher-price", "смартфон"),
    ("fisher-price", "холодильник"),
    ("fisher-price", "чайник"),
    ("librederm", "молоко"),
    ("librederm", "хлеб"),
    ("librederm", "сыр"),
    ("librederm", "смартфон"),
    ("librederm", "наушники"),
    ("librederm", "холодильник"),
    ("librederm", "чайник"),
    ("librederm", "футболка"),
    ("librederm", "джинсы"),
    ("librederm", "кроссовки"),
    ("чёрный жемчуг", "молоко"),
    ("чёрный жемчуг", "хлеб"),
    ("чёрный жемчуг", "сыр"),
    ("чёрный жемчуг", "смартфон"),
    ("чёрный жемчуг", "наушники"),
    ("чёрный жемчуг", "холодильник"),
    ("чёрный жемчуг", "футболка"),
    ("чёрный жемчуг", "джинсы"),
    ("чёрный жемчуг", "кроссовки"),
    ("чистая линия", "молоко"),
    ("чистая линия", "хлеб"),
    ("чистая линия", "сыр"),
    ("чистая линия", "смартфон"),
    ("чистая линия", "наушники"),
    ("чистая линия", "холодильник"),
    ("чистая линия", "футболка"),
    ("чистая линия", "джинсы"),
    ("чистая линия", "кроссовки"),
    ("natura siberica", "молоко"),
    ("natura siberica", "хлеб"),
    ("natura siberica", "сыр"),
    ("natura siberica", "смартфон"),
    ("natura siberica", "наушники"),
    ("natura siberica", "холодильник"),
    ("natura siberica", "футболка"),
    ("natura siberica", "джинсы"),
    ("natura siberica", "кроссовки"),
    ("levrana", "молоко"),
    ("levrana", "хлеб"),
    ("levrana", "сыр"),
    ("levrana", "смартфон"),
    ("levrana", "наушники"),
    ("levrana", "холодильник"),
    ("levrana", "футболка"),
    ("levrana", "джинсы"),
    ("levrana", "кроссовки"),
    ("organic shop", "молоко"),
    ("organic shop", "хлеб"),
    ("organic shop", "сыр"),
    ("organic shop", "смартфон"),
    ("organic shop", "наушники"),
    ("organic shop", "холодильник"),
    ("organic shop", "футболка"),
    ("organic shop", "джинсы"),
    ("organic shop", "кроссовки"),
    ("planeta organica", "молоко"),
    ("planeta organica", "хлеб"),
    ("planeta organica", "сыр"),
    ("planeta organica", "смартфон"),
    ("planeta organica", "наушники"),
    ("planeta organica", "холодильник"),
    ("planeta organica", "футболка"),
    ("planeta organica", "джинсы"),
    ("planeta organica", "кроссовки"),
    ("mulsan cosmetic", "молоко"),
    ("mulsan cosmetic", "хлеб"),
    ("mulsan cosmetic", "сыр"),
    ("mulsan cosmetic", "смартфон"),
    ("mulsan cosmetic", "наушники"),
    ("mulsan cosmetic", "холодильник"),
    ("mulsan cosmetic", "футболка"),
    ("mulsan cosmetic", "джинсы"),
    ("mulsan cosmetic", "кроссовки"),
    ("mi&ko", "молоко"),
    ("mi&ko", "хлеб"),
    ("mi&ko", "сыр"),
    ("mi&ko", "смартфон"),
    ("mi&ko", "наушники"),
    ("mi&ko", "холодильник"),
    ("mi&ko", "футболка"),
    ("mi&ko", "джинсы"),
    ("mi&ko", "кроссовки"),
    ("estee lauder", "молоко"),
    ("estee lauder", "хлеб"),
    ("estee lauder", "сыр"),
    ("estee lauder", "смартфон"),
    ("estee lauder", "наушники"),
    ("estee lauder", "холодильник"),
    ("estee lauder", "футболка"),
    ("estee lauder", "джинсы"),
    ("estee lauder", "кроссовки"),
    ("clinique", "молоко"),
    ("clinique", "хлеб"),
    ("clinique", "сыр"),
    ("clinique", "смартфон"),
    ("clinique", "наушники"),
    ("clinique", "холодильник"),
    ("clinique", "футболка"),
    ("clinique", "джинсы"),
    ("clinique", "кроссовки"),
    ("chanel", "молоко"),
    ("chanel", "хлеб"),
    ("chanel", "сыр"),
    ("chanel", "смартфон"),
    ("chanel", "наушники"),
    ("chanel", "холодильник"),
    ("chanel", "футболка"),
    ("chanel", "джинсы"),
    ("chanel", "кроссовки"),
    ("dior", "молоко"),
    ("dior", "хлеб"),
    ("dior", "сыр"),
    ("dior", "смартфон"),
    ("dior", "наушники"),
    ("dior", "холодильник"),
    ("dior", "футболка"),
    ("dior", "джинсы"),
    ("dior", "кроссовки"),
    ("lancome", "молоко"),
    ("lancome", "хлеб"),
    ("lancome", "сыр"),
    ("lancome", "смартфон"),
    ("lancome", "наушники"),
    ("lancome", "холодильник"),
    ("lancome", "футболка"),
    ("lancome", "джинсы"),
    ("lancome", "кроссовки"),
    ("clarins", "молоко"),
    ("clarins", "хлеб"),
    ("clarins", "сыр"),
    ("clarins", "смартфон"),
    ("clarins", "наушники"),
    ("clarins", "холодильник"),
    ("clarins", "футболка"),
    ("clarins", "джинсы"),
    ("clarins", "кроссовки"),
    ("bioderma", "молоко"),
    ("bioderma", "хлеб"),
    ("bioderma", "сыр"),
    ("bioderma", "смартфон"),
    ("bioderma", "наушники"),
    ("bioderma", "холодильник"),
    ("bioderma", "футболка"),
    ("bioderma", "джинсы"),
    ("bioderma", "кроссовки"),
    ("vichy", "молоко"),
    ("vichy", "хлеб"),
    ("vichy", "сыр"),
    ("vichy", "смартфон"),
    ("vichy", "наушники"),
    ("vichy", "холодильник"),
    ("vichy", "футболка"),
    ("vichy", "джинсы"),
    ("vichy", "кроссовки"),
    ("la roche-posay", "молоко"),
    ("la roche-posay", "хлеб"),
    ("la roche-posay", "сыр"),
    ("la roche-posay", "смартфон"),
    ("la roche-posay", "наушники"),
    ("la roche-posay", "холодильник"),
    ("la roche-posay", "футболка"),
    ("la roche-posay", "джинсы"),
    ("la roche-posay", "кроссовки"),
    ("cerave", "молоко"),
    ("cerave", "хлеб"),
    ("cerave", "сыр"),
    ("cerave", "смартфон"),
    ("cerave", "наушники"),
    ("cerave", "холодильник"),
    ("cerave", "футболка"),
    ("cerave", "джинсы"),
    ("cerave", "кроссовки"),
    ("данон", "футболка"),
    ("данон", "джинсы"),
    ("данон", "платье"),
    ("данон", "кроссовки"),
    ("данон", "смартфон"),
    ("данон", "наушники"),
    ("данон", "холодильник"),
    ("данон", "чайник"),
    ("данон", "шампунь"),
    ("данон", "помада"),
    ("данон", "крем"),
    ("danone", "футболка"),
    ("danone", "джинсы"),
    ("danone", "платье"),
    ("danone", "кроссовки"),
    ("danone", "смартфон"),
    ("danone", "наушники"),
    ("danone", "холодильник"),
    ("danone", "чайник"),
    ("danone", "шампунь"),
    ("danone", "помада"),
    ("danone", "крем"),
    ("dyson", "футболка"),
    ("dyson", "джинсы"),
    ("dyson", "платье"),
    ("dyson", "кроссовки"),
    ("dyson", "смартфон"),
    ("dyson", "наушники"),
    ("dyson", "молоко"),
    ("dyson", "хлеб"),
    ("dyson", "сыр"),
    ("dyson", "помада"),
    ("dyson", "шампунь"),
    ("dyson", "крем"),
    ("pantene", "молоко"),
    ("pantene", "хлеб"),
    ("pantene", "сыр"),
    ("pantene", "смартфон"),
    ("pantene", "наушники"),
    ("pantene", "холодильник"),
    ("pantene", "футболка"),
    ("pantene", "джинсы"),
    ("pantene", "кроссовки"),
    ("head&shoulders", "молоко"),
    ("head&shoulders", "хлеб"),
    ("head&shoulders", "сыр"),
    ("head&shoulders", "смартфон"),
    ("head&shoulders", "наушники"),
    ("head&shoulders", "холодильник"),
    ("head&shoulders", "футболка"),
    ("head&shoulders", "джинсы"),
    ("head&shoulders", "кроссовки"),
    ("schwarzkopf", "молоко"),
    ("schwarzkopf", "хлеб"),
    ("schwarzkopf", "сыр"),
    ("schwarzkopf", "смартфон"),
    ("schwarzkopf", "наушники"),
    ("schwarzkopf", "холодильник"),
    ("schwarzkopf", "футболка"),
    ("schwarzkopf", "джинсы"),
    ("schwarzkopf", "кроссовки"),
    ("syoss", "молоко"),
    ("syoss", "хлеб"),
    ("syoss", "сыр"),
    ("syoss", "смартфон"),
    ("syoss", "наушники"),
    ("syoss", "холодильник"),
    ("syoss", "футболка"),
    ("syoss", "джинсы"),
    ("syoss", "кроссовки"),
    ("garnier", "молоко"),
    ("garnier", "хлеб"),
    ("garnier", "сыр"),
    ("garnier", "смартфон"),
    ("garnier", "наушники"),
    ("garnier", "холодильник"),
    ("garnier", "футболка"),
    ("garnier", "джинсы"),
    ("garnier", "кроссовки"),
]


def load_categories() -> dict[str, Any]:
    with open(CATEGORIES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_bad_combination(name: str) -> bool:
    name_lower = name.lower()
    for bad_a, bad_b in BAD_SUBSTRINGS:
        if bad_a in name_lower and bad_b in name_lower:
            return True
    if re.search(r"\d+л", name_lower) and any(p in name_lower for p in [
        "наушники", "мышь", "клавиатура", "геймпад", "флешка", "адаптер",
        "зарядка", "аккумулятор", "кабель", "штатив",
    ]):
        return True
    if re.search(r"\b\d+кг\b", name_lower) and any(p in name_lower for p in [
        "наушники", "мышь", "клавиатура", "геймпад", "телефон", "смартфон",
        "флешка", "адаптер", "зарядка", "помада", "блеск", "тушь",
    ]):
        return True
    if re.search(r"wh-\s*$", name_lower) or re.search(r"wh-\s+[a-z]+", name_lower):
        if not re.search(r"wh-\d", name_lower):
            return True
    return False


def is_brand_line_incompatible(name: str, category: str) -> bool:
    name_lower = name.lower()
    compat = BRAND_LINE_COMPAT.get(category, {})
    for line, valid_brands in compat.items():
        if line in name_lower:
            if not any(brand in name_lower for brand in valid_brands):
                return True
    return False


def has_repeating_token(name: str) -> bool:
    tokens = name.lower().split()
    return len(tokens) != len(set(tokens))


def is_too_long(name: str) -> bool:
    return len(name.split()) > 6


def weighted_template_choice(templates: list[dict], rng: random.Random) -> dict:
    weights = [t.get("weight", 1) for t in templates]
    return rng.choices(templates, weights=weights, k=1)[0]


def generate_name_from_template(template: dict, rng: random.Random) -> str:
    slots = template["slots"]
    values = {slot: rng.choice(vocab) for slot, vocab in slots.items()}
    return template["pattern"].format(**values)


def build_corpus(
    data: dict[str, Any],
    per_category: int = PER_CATEGORY,
) -> list[dict[str, str]]:
    categories: list[str] = data["categories"]
    rng = random.Random(SEED)

    rows: list[dict[str, str]] = []
    seen_global: set[tuple[str, str]] = set()
    rejection_stats: dict[str, Counter] = defaultdict(Counter)

    for cat in categories:
        cat_data = data[cat]
        templates: list[dict] = cat_data["templates"]

        cat_seen: set[str] = set()
        attempts = 0
        max_attempts = per_category * 50

        while (
            sum(1 for r in rows if r["category"] == cat) < per_category
            and attempts < max_attempts
        ):
            attempts += 1
            template = weighted_template_choice(templates, rng)
            raw = generate_name_from_template(template, rng)
            name = normalize_text(raw)

            if not name:
                rejection_stats[cat]["empty"] += 1
                continue
            if has_repeating_token(name):
                rejection_stats[cat]["duplicate_token"] += 1
                continue
            if is_too_long(name):
                rejection_stats[cat]["too_long"] += 1
                continue
            if is_bad_combination(name):
                rejection_stats[cat]["bad_combination"] += 1
                continue
            if is_brand_line_incompatible(name, cat):
                rejection_stats[cat]["brand_line_mismatch"] += 1
                continue
            if is_brand_product_incompatible(name, cat):
                rejection_stats[cat]["brand_product_mismatch"] += 1
                continue

            key = (cat, name.lower())
            if key in seen_global:
                rejection_stats[cat]["duplicate_global"] += 1
                continue
            if name.lower() in cat_seen:
                rejection_stats[cat]["duplicate_in_cat"] += 1
                continue

            seen_global.add(key)
            cat_seen.add(name.lower())
            rows.append({"category": cat, "name": name})

    rng.shuffle(rows)
    return rows, rejection_stats


def save_corpus(rows: list[dict[str, str]]) -> None:
    with open(CORPUS_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "name"])
        writer.writeheader()
        writer.writerows(rows)


def summarize(
    rows: list[dict[str, str]],
    rejection_stats: dict[str, Counter],
) -> None:
    counter = Counter(r["category"] for r in rows)
    unique_names = {r["name"].lower() for r in rows}

    print(f"Корпус сохранён: {CORPUS_FILE}")
    print(f"Всего названий: {len(rows)}")
    print(f"Категорий: {len(counter)}")
    print(f"Уникальных названий: {len(unique_names)}")

    print("\nРаспределение по категориям:")
    for cat, n in counter.most_common():
        print(f"  {cat:<12} {n}")

    print("\nСтатистика отклонений (по причинам):")
    for cat, stats in rejection_stats.items():
        total = sum(stats.values())
        if total == 0:
            continue
        print(f"  {cat}:")
        for reason, n in stats.most_common():
            pct = 100.0 * n / total
            print(f"    {reason:<22} {n:>6} ({pct:5.1f}%)")

    by_cat: dict[str, list[str]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r["name"])

    print("\nПримеры (по 10 на категорию):")
    for cat, names in by_cat.items():
        print(f"\n[{cat}]")
        for n in names[:10]:
            print(f"  - {n}")


def main() -> None:
    data = load_categories()
    rows, rejection_stats = build_corpus(data, PER_CATEGORY)
    save_corpus(rows)
    summarize(rows, rejection_stats)


if __name__ == "__main__":
    main()
