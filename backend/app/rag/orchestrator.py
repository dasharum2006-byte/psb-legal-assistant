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
import os

# Порог отсечения: если лучший результат поиска слабее — считаем, что
# ничего не нашли, и отказываем вместо ответа по мусору.
# 12 — для BM25 (нормальные вопросы дают 60-112, мусорные 4-7).
# Если переключимся на гибрид с очками 0..1 — поставить 0.25.
MIN_SCORE = float(os.getenv("MIN_SCORE", "8"))

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
           "as_of": find_date(question),                                                                          
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
def find_doc_ids(question: str) -> list[str]:
    # опознаем документ в вопросе по номеру или контрагенту
    text = question.lower().replace("ё","е")
    result = []
    if "а-08" in text and "А-08/2024" not in result:
        result.append("А-08/2024")
    if "п-17" in text and "П-17/2024" not in result:
        result.append("П-17/2024")
    if "пт-05" in text and "ПТ-05/2023" not in result:
        result.append("ПТ-05/2023")
    if "юо-7" in text and "ЮО-7" not in result:
        result.append("ЮО-7")
    if "к-12" in text and "К-12" not in result:
        result.append("К-12")
    if "альфа-трейдинг" in text:
        if "ПТ-05/2023" not in result:
            result.append("ПТ-05/2023")
    elif "альфа-трейд" in text:
        if "П-17/2024" not in result:
            result.append("П-17/2024")
    if "северн" in text and "А-08/2024" not in result:
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

        for name, number in MONTHS.items():                                                                    
               if name in text:                                                                                   
                   count_days = calendar.monthrange(year, number)[1]                                              
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
        "clause": clause,
        "text": chunk.get("text", ""),
        "score": float(chunk.get("score", 0)),
        "status": chunk.get("status"),
        "valid_from": chunk.get("valid_from"),
        "valid_to": chunk.get("valid_to"),
    }


def _post_filter(chunks: list[dict], analysis: dict) -> list[dict]:
    """Фильтры на нашей стороне: пока поиск их не поддерживает."""
    doc_ids = [d.lower().replace("/", "-") for d in analysis.get("doc_ids") or []]
    as_of = analysis.get("as_of")

    result = []
    for chunk in chunks:
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
    """Оставить только те источники, на которые ответ реально ссылается."""
    cited = [chunk for chunk in chunks if chunk["clause"] and chunk["clause"] in answer]
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


def ask(question: str, history: list[dict], retriever, client=None) -> dict:
    """Главный вход: разбор -> поиск -> отбор -> LLM -> проверка -> ответ или отказ."""
    from app.llm.client import LLMClient
    from app.llm.prompt import build_rag_prompt
    from app.llm.verifier import Verifier

    analysis = analyze_query(contextual_query(question, history))

    # 1. поиск
    try:
        raw = retriever.search(analysis["search_query"], 8)
    except Exception:
        return _refusal("llm_unavailable")

    chunks = _post_filter([_adapt(chunk) for chunk in raw], analysis)

    # 2. порог: всё слабее MIN_SCORE — считаем, что не нашли
    chunks = [chunk for chunk in chunks if chunk["score"] >= MIN_SCORE]
    if not chunks:
        return _refusal("no_chunks")

    # 3. отбор под бюджет и склейка
    picked = select_chunks(chunks)
    if not picked:
        return _refusal("no_chunks")

    # 4. генерация
    client = client or LLMClient()
    try:
        answer = client.generate(build_rag_prompt(question, picked))
    except Exception:
        return _refusal("llm_unavailable")

    # 5. проверка ответа
    verdict = Verifier.verify(answer, picked, question)
    if not verdict.get("ok") or verdict.get("refusal"):
        missing = verdict.get("missing_info") or verdict.get("unsupported")
        return _refusal("unsupported", missing)

    # 6. ответ
    return {
        "answer": answer.strip(),
        "sources": pick_sources(answer, picked),
        "refusal": False,
        "missing_info": [],
    }
