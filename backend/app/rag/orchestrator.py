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
#функции заглушки
# функция analyze_query - на ее вход идет (question: str) - строка с вопросом на выходе 
# будет словарь Python типо что то на подобие json
# 
# def analyze_query(question: str) -> dict:
    # """v0: вытащить даты,номера договоров,названия.""" #что функция будет делать
    # что возвращать
#Типо - analyze_query("Какая аренданая плата?") - она вернет none
def analyze_query(question: str) -> dict:                                                                      
       """Разобрать вопрос: дата, документы, суммы, уточнение это или нет."""                                     
       return {                                                                                                   
           "search_query": question.strip(),                                                                      
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