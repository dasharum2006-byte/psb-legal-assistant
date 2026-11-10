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
#  Функция склеивает куски документов в один котнтекст который потом пойдет в LLM
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
        label = "" if clause.lower().startwith("ст") else "п. "
        header = f"[{i}] {chunk.get('doc','')}, п. {chunk.get('clause', '')}"
        block = header + "\n" + chunk.get('text', '')
        if len(blocks)>0 and used + len(block)> max_chars:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)
        
    


