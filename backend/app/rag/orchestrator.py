"""AI-оркестрация: анализ вопроса, multi-query, сборка контекста.

TODO:
- query analyzer;
- query decomposer;
- context builder;
- dialogue memory;
- eval harness.
"""
#функции заглушки
# функция analyze_query - на ее вход идет (question: str) - строка с вопросом на выходе 
# будет словарь Python типо что то на подобие json
# 
def analyze_query(question: str) -> dict:
    """v0: вытащить даты,номера договоров,названия.""" #что функция будет делать
    ... # что возвращать
#Типо - analyze_query("Какая аренданая плата?") - она вернет none


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

