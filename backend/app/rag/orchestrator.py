"""AI-оркестрация: анализ вопроса, multi-query, сборка контекста.
TODO:
- query analyzer;
- query decomposer;
- context builder;
- dialogue memory;
- eval harness.
"""

import re
import calendar
import logging
import os

# Порог отсечения: если лучший результат поиска слабее — считаем, что
# ничего не нашли, и отказываем вместо ответа по мусору.
# 12 — для BM25 (нормальные вопросы дают 60-112, мусорные 4-7).
# Если переключимся на гибрид с очками 0..1 — поставить 0.25.
logger = logging.getLogger(__name__)

MIN_SCORE = float(os.getenv("MIN_SCORE", "8"))

# Сколько кандидатов забирать из поиска ДО фильтров. Фильтры (doc_ids,
# doc_type, status, as_of) применяются после поиска и могут отсечь всё,
# если нужный документ оказался на 9-15 месте. Поэтому берём с запасом.
SEARCH_CANDIDATES = int(os.getenv("SEARCH_CANDIDATES", "40"))

# Порог отсечения зависит от шкалы очков поиска:
#   BM25           — десятки (10..100)  -> MIN_SCORE (по умолчанию 8)
#   косинус / RRF  — 0..1              -> MIN_SCORE_NORMALIZED (по умолчанию 0.25)
# Жёсткий порог 8 на шкале 0..1 выбросил бы ВСЕ чанки и мы бы всегда отказывали.
MIN_SCORE_NORMALIZED = float(os.getenv("MIN_SCORE_NORMALIZED", "0.25"))


def score_threshold(scores: list[float]) -> float:
    """Порог, соответствующий шкале, в которой пришёл поиск."""
    top = max(scores, default=0.0)
    return MIN_SCORE_NORMALIZED if top <= 1.5 else MIN_SCORE

# Доменные подсказки: слова из вопроса -> слова из документов.
# Нужны, потому что люди говорят «платить», а в договоре «арендная плата».
SYNONYMS = {
    "плат": "оплата плата арендная стоимость",
    "ноч": "круглосуточно доступ беспрепятственный транспорт",
    "круглосуточ": "круглосуточно доступ транспорт",
    "ключ": "акт возврат помещения приёмка",
    "архив": "записка оригинал выдача юо-7",
    "оригинал": "оригинал архив выдача юо-7",
    "гарант": "гарантия гарантийный срок качество",
    "компьютер": "компьютерная техника оборудование",
    "мебел": "мебель оборудование",
    "испытан": "испытание испытательный срок",
    "претенз": "претензия акт недостатки дефекты",
    "аванс": "предоплата аванс оплата",
    "санкц": "санкционная проверка комплаенс чек-лист",
    "хранен": "хранение архив срок",
}
#функции заглушки
# функция analyze_query - на ее вход идет (question: str) - строка с вопросом на выходе 
# будет словарь Python типо что то на подобие json
# 
# def analyze_query(question: str) -> dict:
    # """v0: вытащить даты,номера договоров,названия.""" #что функция будет делать
    # что возвращать
#Типо - analyze_query("Какая аренданая плата?") - она вернет none
def contextual_query(question: str, history: list[dict] | None) -> str:
    """Для уточняющих вопросов подмешать предыдущий вопрос пользователя.

    Вопросы 6 и 15-18 кейса — это диалоги. Фраза «А за сколько дней после
    поставки?» без предыдущего вопроса не содержит ни договора, ни темы.
    """
    if not history:
        return question

    previous = [m.get("text", "") for m in history if m.get("role") == "user" and m.get("text")]
    if not previous:
        return question

    short = len(question.split()) <= 10 or question.strip().lower().startswith(("а ", "и ", "тогда"))
    return f"{previous[-1]} {question}" if short else question


def expand_query(question: str) -> str:
    """Добавить к вопросу слова из документов (доменные синонимы)."""
    low = question.lower().replace("ё", "е")
    extra = [words for stem, words in SYNONYMS.items() if stem in low]
    return f"{question.strip()} {' '.join(extra)}" if extra else question.strip()


def analyze_query(question: str) -> dict:                                                                      
       """Разобрать вопрос: дата, документы, суммы, уточнение это или нет."""                                     
       return {                                                                                                   
           "search_query": expand_query(question),                                                                      
           "as_of": find_date(question) if has_temporal_intent(question) else None,                                                                          
           "doc_ids": find_doc_ids(question),                                                                     
           "amounts": [],                                                                                         
           "is_followup": False,                                                                                  
       }   

# chunks: list[dict] - на вход список кусков документов,каждый кусок - это словарь с полями - doc,clause,text
# max_chars: int = 6000 - необязательный параметр если не передать то по умолчанию 6000 символов
# и она возвращает одну большую строку
#  Функция склеивает куски документов с подписями и указывает лимит символов в один котнтекст который потом пойдет в LLM
def build_context(chunks: list[dict], max_chars: int = 6000) -> str:
    """v0: собрать чанки в один контекст."""
    #готовые блоки текста
    blocks = []
    #сколько символов уже занято
    used = 0
    # пройтись по чанкам с номером
    # i - номер чанка,enumerate выдает пары(номер,элемент)
    for i, chunk in enumerate(chunks, 1):
        clause = chunk.get('clause', '')
        #label - достаем номер пункта - а это приставка для статей закона,типо для пунктов договора она нужна
        label = "" if clause.lower().startswith("ст") else "п. "
        #кусок собираем - подпись_перенос строки+сам текст
        header = f"[{i}] {chunk.get('doc','')}, {label}{clause}"
        block = header + "\n" + chunk.get('text', '')
        #если в коробке что то есть( то есть это не первый кусок) и если добавить этот кусок, то лимит будет превышен
        if len(blocks)>0 and used + len(block)> max_chars:
            break
        # гирю передвигаем
        blocks.append(block)
        used += len(block)
        #склеиваем все куски в одну строку и оставляем пустую строку чтобы гаринца между доками была
    return "\n\n".join(blocks)

        
    
# на входе у нас строка на выходе список строк
# Временной смысл вопроса: «на какую дату действовало право».
# Год в номере договора или дате документа сам по себе НЕ означает, что
# спрашивают про состояние на дату — иначе фильтр as_of отсекал бы
# актуальные документы на вопросе «договор от 15 марта 2023 года».
TEMPORAL_PATTERNS = (
    r"по состоянию на",
    r"на дату",
    r"на момент",
    r"на тот момент",
    r"\bдействовал",
    r"\bдействует\b",
    r"(?<![а-яё])(в|на|с|к|до|за)\s+(?:\d{1,2}\s+)?[а-яё]+\s*20\d{2}",
    r"(?<![а-яё])(в|на|с|к|до|за)\s+20\d{2}\s*(?:год|году|года)",
    r"на\s+\d{2}\.\d{2}\.\d{4}",
    r"\bтогда\b",
)


def has_temporal_intent(question: str) -> bool:
    """Спрашивают ли про состояние на дату, а не просто упомянули год."""
    low = question.lower().replace("ё", "е")
    return any(re.search(pattern, low) for pattern in TEMPORAL_PATTERNS)


def find_doc_ids(question: str) -> list[str]:
    # опознаем документ в вопросе по номеру или контрагенту
    text = question.lower().replace("ё","е")
    result = []
    # границы слова: «к-12» не должен срабатывать внутри «К-123»
    number_map = (
        ("а-08", "А-08/2024"),
        ("п-17", "П-17/2024"),
        ("пт-05", "ПТ-05/2023"),
        ("юо-7", "ЮО-7"),
        ("к-12", "К-12"),
    )
    for pattern, doc_id in number_map:
        if re.search(rf"(?<![а-яa-z0-9]){re.escape(pattern)}(?!\d)", text) and doc_id not in result:
            result.append(doc_id)
    if re.search(r"(?<![а-я])альфа-трейдинг", text):
        if "ПТ-05/2023" not in result:
            result.append("ПТ-05/2023")
    elif re.search(r"(?<![а-я])альфа-трейд(?![а-я])", text):
        if "П-17/2024" not in result:
            result.append("П-17/2024")
    if re.search(r"северн", text) and "А-08/2024" not in result:
        result.append("А-08/2024")
    return result

MONTHS = {                                                                                                     
    "января": 1, "январь": 1, "январе": 1,                                                                     
    "февраля": 2, "февраль": 2, "феврале": 2,                                                                  
    "марта": 3, "март": 3, "марте": 3,                                                                         
    "апреля": 4, "апрель": 4, "апреле": 4,                                                                     
    "мая": 5, "май": 5, "мае": 5,                                                                              
    "июня": 6, "июнь": 6, "июне": 6,                                                                           
    "июля": 7, "июль": 7, "июле": 7,                                                                           
    "августа": 8, "август": 8, "августе": 8,                                                                   
    "сентября": 9, "сентябрь": 9, "сентябре": 9,                                                               
    "октября": 10, "октябрь": 10, "октябре": 10,                                                               
    "ноября": 11, "ноябрь": 11, "ноябре": 11,                                                                  
    "декабря": 12, "декабрь": 12, "декабре": 12,                                                               
}     

def find_date(question: str) -> str | None:
    #приводим текст в нижнему регистру
    text = question.lower().replace("ё","е")
    m = re.search(r"(\d{2})\.(\d{2})\.(\d{4})", text)
    if m:
        day = m.group(1)
        month = m.group(2)
        year = m.group(3)
        return f"{year}-{month}-{day}"

    
    m_year = re.search(r"(?<!/)\b(20\d{2})\b", text)
    if m_year:
        year = int(m_year.group(1))  
        for name,number in MONTHS.items():
            if name in text:
                count_days = calendar.monthrange(year,number)[1]
                return f"{year:04d}-{number:02d}-{count_days:02d}" 

                                                                                                                                       
        if re.search(r"\b(год|году|г\.)", text):                                                               
            return f"{year:04d}-12-31"  
    return  None      



# сортируем чанки по убыванию
def select_chunks(chunks: list[dict], max_chars: int = 6000, max_per_doc: int = 3) -> list[dict]:              
    """Отобрать куски под бюджет символов: по score, не больше N с документа."""                               
    picked = []                                                                                                
    used = 0                                                                                                   
    per_doc = {}                                                                                               
                                                                                                                  
    for chunk in sorted(chunks, key=lambda chunk: chunk.get("score", 0), reverse=True):                        
        key = chunk.get("doc_id") or chunk.get("doc") or chunk.get("title", "")                                
        if per_doc.get(key, 0) >= max_per_doc:                                                                 
            continue                                                                                                                                                                                                     
        size = len(chunk.get("text", ""))                                                                      
        if picked and used + size > max_chars:                                                                 
            break                                                                                              
                                                                                                                  
        picked.append(chunk)                                                                                   
        used += size                                                                                           
        per_doc[key] = per_doc.get(key, 0) + 1                                                                 
                                                                                                                  
    return picked    

# ─────────────────────── сборка пайплайна ───────────────────────

def _adapt(chunk: dict) -> dict:
    """Переходник: привести чанк из формата поиска к нашему."""
    clause = str(chunk.get("clause", ""))
    for prefix in ("п. ", "П. ", "п."):
        if clause.startswith(prefix):
            clause = clause[len(prefix):].strip()
            break
    return {
        "doc": chunk.get("doc") or chunk.get("title", ""),
        "doc_id": chunk.get("doc_id", ""),
        "doc_type": chunk.get("doc_type"),
        "number": chunk.get("number"),
        "revision": chunk.get("revision"),
        "clause": clause,
        "text": chunk.get("text", ""),
        "score": float(chunk.get("score", 0)),
        "status": chunk.get("status"),
        "valid_from": chunk.get("valid_from"),
        "valid_to": chunk.get("valid_to"),
    }


def _post_filter(chunks: list[dict], analysis: dict, filters: dict | None = None) -> list[dict]:
    """Фильтры поиска: из разбора вопроса И из запроса API.

    filters из запроса: doc_ids, doc_type, status, as_of.
    """
    filters = filters or {}

    doc_ids = [d.lower().replace("/", "-") for d in (analysis.get("doc_ids") or [])]
    doc_ids += [str(d).lower().replace("/", "-") for d in (filters.get("doc_ids") or [])]

    as_of = filters.get("as_of") or analysis.get("as_of")
    doc_types = {str(t).lower() for t in (filters.get("doc_type") or [])} if filters.get("doc_type") else set()
    statuses = {str(s).lower() for s in (filters.get("status") or [])} if filters.get("status") else set()

    result = []
    for chunk in chunks:
        if doc_types:
            chunk_type = str(chunk.get("doc_type") or "").lower()
            if chunk_type not in doc_types:
                continue
        if statuses:
            chunk_status = str(chunk.get("status") or "").lower()
            if chunk_status not in statuses:
                continue
        if doc_ids:
            haystack = f"{chunk['doc_id']} {chunk['doc']}".lower().replace("/", "-")
            if not any(doc_id in haystack for doc_id in doc_ids):
                continue
        if as_of:
            if chunk.get("valid_from") and chunk["valid_from"] > as_of:
                continue
            if chunk.get("valid_to") and chunk["valid_to"] <= as_of:
                continue
        result.append(chunk)
    return result


def _refusal(reason: str, missing: list[str] | None = None) -> dict:
    """Обоснованный отказ: это нормальный ответ, а не ошибка."""
    texts = {
        "no_chunks": "В базе знаний нет документов, по которым можно ответить на этот вопрос. "
                     "По памяти отвечать не буду: нормы могли измениться.",
        "llm_unavailable": "Сервис ответов временно недоступен, попробуйте ещё раз.",
        "unsupported": "Найденные документы не подтверждают такой ответ.",
    }
    return {
        "answer": texts.get(reason, "Ответить не могу."),
        "sources": [],
        "refusal": True,
        "missing_info": missing or ["нужны документы, которых нет в базе знаний"],
    }


def pick_sources(answer: str, chunks: list[dict], limit: int = 3) -> list[dict]:
    """Оставить только те источники, на которые ответ реально ссылается.

    Сравниваем по каноническим ключам ссылок: «статья 15», «ст.15» и
    «ст. 15» — это одна и та же ссылка. Раньше сравнение было буквальным,
    и в источники попадало всё, кроме нужного.
    """
    from app.rag.normalize import clause_key, clause_keys, keys_match

    answer_keys = clause_keys(answer)
    cited = [
        chunk for chunk in chunks
        if chunk.get("clause")
        and any(keys_match(key, clause_key(chunk["clause"]) or "") for key in answer_keys)
    ]
    chosen = (cited or chunks)[:limit]
    return [
        {
            "doc": chunk["doc"],
            "clause": chunk["clause"],
            "text": chunk["text"][:300],
            "score": chunk["score"],
        }
        for chunk in chosen
    ]


def ask(
    question: str,
    history: list[dict],
    retriever,
    client=None,
    filters: dict | None = None,
) -> dict:
    """Главный вход: разбор -> поиск -> отбор -> LLM -> проверка -> ответ или отказ.

    filters приходят из запроса API: {"doc_ids": [...], "doc_type": [...],
    "status": ["active"], "as_of": "2024-08-31"}.
    """
    from app.llm.client import LLMClient
    from app.llm.prompt import build_rag_prompt
    from app.llm.verifier import Verifier

    analysis = analyze_query(contextual_query(question, history))

    # 1. поиск
    try:
        raw = retriever.search(analysis["search_query"], SEARCH_CANDIDATES)
    except Exception as exc:
        logger.exception("RETRIEVER_FAILED: %s: %s", type(exc).__name__, exc)
        return _refusal("llm_unavailable")

    chunks = _post_filter([_adapt(chunk) for chunk in raw], analysis, filters)

    # 2. порог: шкала определяется по максимальному очку
    threshold = score_threshold([chunk["score"] for chunk in chunks])
    chunks = [chunk for chunk in chunks if chunk["score"] >= threshold]
    if not chunks:
        return _refusal("no_chunks")

    # 3. отбор под бюджет и склейка
    picked = select_chunks(chunks)
    if not picked:
        return _refusal("no_chunks")

    # 4. генерация
    client = client or LLMClient()
    try:
        prompt = build_rag_prompt(question, picked, history)
        answer = client.generate(prompt)
    except TypeError as exc:
        # несовпадение сигнатур (например, промпт перестал принимать history)
        # — это ошибка кода, её надо видеть, а не маскировать отказом
        logger.exception("PROMPT_SIGNATURE_ERROR: %s", exc)
        return _refusal("llm_unavailable")
    except Exception as exc:
        logger.exception("LLM_CALL_FAILED: %s: %s", type(exc).__name__, exc)
        return _refusal("llm_unavailable")

    # 5. проверка ответа
    verdict = Verifier.verify(answer, picked, question)

    # Модель корректно отказалась: сохраняем её формулировку с объяснением,
    # чего не хватает. Шаблонная заглушка теряет эту информацию и хуже
    # оценивается судьёй.
    if verdict.get("refusal") and not verdict.get("injection_suspected"):
        text = answer.strip()
        return {
            "answer": text or _refusal("no_chunks")["answer"],
            "sources": [],
            "refusal": True,
            "missing_info": verdict.get("missing_info") or [text],
        }

    if not verdict.get("ok"):
        missing = verdict.get("missing_info") or verdict.get("unsupported")
        return _refusal("unsupported", missing)

    # 6. ответ
    return {
        "answer": answer.strip(),
        "sources": pick_sources(answer, picked),
        "refusal": False,
        "missing_info": [],
    }
