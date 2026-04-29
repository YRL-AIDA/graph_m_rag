import html
import json
import logging
import re
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import pandas as pd
import requests
from openai import OpenAI

# --- Настройки логирования ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Константы и Промпты ---
TOKENIZER_URL = "http://192.168.19.127:9886/tokenize"
LLM_URL = 'http://192.168.19.127:9886/v1'

TUPLE_DELIMITER = "<|>"
RECORD_DELIMITER = "##"
COMPLETION_DELIMITER = "<|COMPLETE|>"
SUMMARIZE_PROMPT = """
You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or more entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we have the full context.
Limit the final description length to {max_length} words.

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""


"""A file containing prompts definition."""

GRAPH_EXTRACTION_PROMPT = """
-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.
 
-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"<|><entity_name><|><entity_type><|><entity_description>)
 
2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
 Format each relationship as ("relationship"<|><source_entity><|><target_entity><|><relationship_description><|><relationship_strength>)
 
3. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **##** as the list delimiter.
 
4. When finished, output <|COMPLETE|>
 
######################
-Examples-
######################
Example 1:
Entity_types: ORGANIZATION,PERSON
Text:
The Verdantis's Central Institution is scheduled to meet on Monday and Thursday, with the institution planning to release its latest policy decision on Thursday at 1:30 p.m. PDT, followed by a press conference where Central Institution Chair Martin Smith will take questions. Investors expect the Market Strategy Committee to hold its benchmark interest rate steady in a range of 3.5%-3.75%.
######################
Output:
("entity"<|>CENTRAL INSTITUTION<|>ORGANIZATION<|>The Central Institution is the Federal Reserve of Verdantis, which is setting interest rates on Monday and Thursday)
##
("entity"<|>MARTIN SMITH<|>PERSON<|>Martin Smith is the chair of the Central Institution)
##
("entity"<|>MARKET STRATEGY COMMITTEE<|>ORGANIZATION<|>The Central Institution committee makes key decisions about interest rates and the growth of Verdantis's money supply)
##
("relationship"<|>MARTIN SMITH<|>CENTRAL INSTITUTION<|>Martin Smith is the Chair of the Central Institution and will answer questions at a press conference<|>9)
<|COMPLETE|>

######################
Example 2:
Entity_types: ORGANIZATION
Text:
TechGlobal's (TG) stock skyrocketed in its opening day on the Global Exchange Thursday. But IPO experts warn that the semiconductor corporation's debut on the public markets isn't indicative of how other newly listed companies may perform.

TechGlobal, a formerly public company, was taken private by Vision Holdings in 2014. The well-established chip designer says it powers 85% of premium smartphones.
######################
Output:
("entity"<|>TECHGLOBAL<|>ORGANIZATION<|>TechGlobal is a stock now listed on the Global Exchange which powers 85% of premium smartphones)
##
("entity"<|>VISION HOLDINGS<|>ORGANIZATION<|>Vision Holdings is a firm that previously owned TechGlobal)
##
("relationship"<|>TECHGLOBAL<|>VISION HOLDINGS<|>Vision Holdings formerly owned TechGlobal from 2014 until present<|>5)
<|COMPLETE|>

######################
Example 3:
Entity_types: ORGANIZATION,GEO,PERSON
Text:
Five Aurelians jailed for 8 years in Firuzabad and widely regarded as hostages are on their way home to Aurelia.

The swap orchestrated by Quintara was finalized when $8bn of Firuzi funds were transferred to financial institutions in Krohaara, the capital of Quintara.

The exchange initiated in Firuzabad's capital, Tiruzia, led to the four men and one woman, who are also Firuzi nationals, boarding a chartered flight to Krohaara.

They were welcomed by senior Aurelian officials and are now on their way to Aurelia's capital, Cashion.

The Aurelians include 39-year-old businessman Samuel Namara, who has been held in Tiruzia's Alhamia Prison, as well as journalist Durke Bataglani, 59, and environmentalist Meggie Tazbah, 53, who also holds Bratinas nationality.
######################
Output:
("entity"<|>FIRUZABAD<|>GEO<|>Firuzabad held Aurelians as hostages)
##
("entity"<|>AURELIA<|>GEO<|>Country seeking to release hostages)
##
("entity"<|>QUINTARA<|>GEO<|>Country that negotiated a swap of money in exchange for hostages)
##
##
("entity"<|>TIRUZIA<|>GEO<|>Capital of Firuzabad where the Aurelians were being held)
##
("entity"<|>KROHAARA<|>GEO<|>Capital city in Quintara)
##
("entity"<|>CASHION<|>GEO<|>Capital city in Aurelia)
##
("entity"<|>SAMUEL NAMARA<|>PERSON<|>Aurelian who spent time in Tiruzia's Alhamia Prison)
##
("entity"<|>ALHAMIA PRISON<|>GEO<|>Prison in Tiruzia)
##
("entity"<|>DURKE BATAGLANI<|>PERSON<|>Aurelian journalist who was held hostage)
##
("entity"<|>MEGGIE TAZBAH<|>PERSON<|>Bratinas national and environmentalist who was held hostage)
##
("relationship"<|>FIRUZABAD<|>AURELIA<|>Firuzabad negotiated a hostage exchange with Aurelia<|>2)
##
("relationship"<|>QUINTARA<|>AURELIA<|>Quintara brokered the hostage exchange between Firuzabad and Aurelia<|>2)
##
("relationship"<|>QUINTARA<|>FIRUZABAD<|>Quintara brokered the hostage exchange between Firuzabad and Aurelia<|>2)
##
("relationship"<|>SAMUEL NAMARA<|>ALHAMIA PRISON<|>Samuel Namara was a prisoner at Alhamia prison<|>8)
##
("relationship"<|>SAMUEL NAMARA<|>MEGGIE TAZBAH<|>Samuel Namara and Meggie Tazbah were exchanged in the same hostage release<|>2)
##
("relationship"<|>SAMUEL NAMARA<|>DURKE BATAGLANI<|>Samuel Namara and Durke Bataglani were exchanged in the same hostage release<|>2)
##
("relationship"<|>MEGGIE TAZBAH<|>DURKE BATAGLANI<|>Meggie Tazbah and Durke Bataglani were exchanged in the same hostage release<|>2)
##
("relationship"<|>SAMUEL NAMARA<|>FIRUZABAD<|>Samuel Namara was a hostage in Firuzabad<|>2)
##
("relationship"<|>MEGGIE TAZBAH<|>FIRUZABAD<|>Meggie Tazbah was a hostage in Firuzabad<|>2)
##
("relationship"<|>DURKE BATAGLANI<|>FIRUZABAD<|>Durke Bataglani was a hostage in Firuzabad<|>2)
<|COMPLETE|>

######################
-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:"""

CONTINUE_PROMPT = "MANY entities and relationships were missed in the last extraction. Remember to ONLY emit entities that match any of the previously extracted types. Add them below using the same format:\n"
LOOP_PROMPT = "It appears some entities and relationships may have still been missed. Answer Y if there are still entities or relationships that need to be added, or N if there are none. Please answer with a single letter Y or N.\n"


# --- Утилиты ---

def clean_str(text: Any) -> str:
    """Удаляет HTML-экранирования и управляющие символы из строки."""
    if not isinstance(text, str):
        return text
    result = html.unescape(text.strip())
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)

def remove_think_tags(text: str) -> str:
    """Удаляет тег <think>...</think> из ответов моделей reasoning."""
    try:
        match = re.search(r"<\/think>", text)
        if match:
            return text[match.end():].strip()
    except Exception as e:
        logger.error(f"Error removing <think> tags: {e}")
    return text.strip()

# --- Клиент LLM и Токенизатор ---

class LLMClient:
    """Обертка над OpenAI API и VLLM токенизатором для удобной работы."""
    def __init__(self, base_url: str = LLM_URL, tokenizer_url: str = TOKENIZER_URL, api_key: str = 'EMPTY'):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.tokenizer_url = tokenizer_url

    def generate(self, messages: List[Dict[str, str]], model: str, **kwargs) -> Optional[str]:
        """Отправляет запрос к LLM и возвращает очищенный текстовый ответ."""
        try:
            logger.info(f"Generating content with model: {model}")
            response = self.client.chat.completions.create(
                messages=messages,
                model=model,
                **kwargs
            )
            raw_content = response.choices[0].message.content
            return remove_think_tags(raw_content)
        except Exception as e:
            logger.error(f"Failed to call LLM: {e}")
            return None

    def count_tokens(self, text: str, model: str) -> int:
        """Подсчитывает токены через удаленный endpoint."""
        try:
            response = requests.post(
                self.tokenizer_url,
                json={"model": model, "prompt": text},
                timeout=10
            )
            return response.json().get('count', len(text) // 4)
        except Exception as e:
            logger.warning(f"Token counting failed, using fallback. Error: {e}")
            return len(text) // 4 + 1

# --- Извлечение Графа (Extraction) ---

class GraphExtractor:
    """Класс для извлечения сущностей и связей из сырого текста."""
    def __init__(self, llm_client: LLMClient, model: str, max_gleanings: int):
        self.llm = llm_client
        self.model = model
        self.max_gleanings = max_gleanings

    def extract(self, text: str, entity_types: List[str], source_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info(f"Extracting graph for document ID: {source_id}")
        
        # 1. Основной запрос
        prompt = GRAPH_EXTRACTION_PROMPT.format(
            input_text=text,
            entity_types=",".join(entity_types)
        )
        messages = [{"role": "user", "content": prompt}]
        
        response_text = self.llm.generate(messages, self.model)
        if not response_text:
            return self._empty_dfs()

        full_result = response_text
        messages.append({"role": "assistant", "content": response_text})

        # 2. Gleaning (дополнительный сбор упущенных данных)
        for _ in range(self.max_gleanings):
            messages.append({"role": "user", "content": CONTINUE_PROMPT})
            continuation = self.llm.generate(messages, self.model)
            if not continuation:
                break
            
            full_result += continuation
            messages.append({"role": "assistant", "content": continuation})
            
            # Спрашиваем, нужно ли продолжать
            messages.append({"role": "user", "content": LOOP_PROMPT})
            loop_decision = self.llm.generate(messages, self.model)
            if not loop_decision or loop_decision.strip().upper() != "Y":
                break
        print(full_result)
        return self._parse_result(full_result, source_id)

    def _parse_result(self, result: str, source_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        entities, relationships = [], []
        records = [r.strip() for r in result.split(RECORD_DELIMITER)]

        for raw_record in records:
            record = re.sub(r"^\(|\)$", "", raw_record.strip())
            if not record or record == COMPLETION_DELIMITER:
                continue

            record_attributes = record.split(TUPLE_DELIMITER)
            record_type = record_attributes[0]

            if record_type == '"entity"' and len(record_attributes) >= 4:
                entity_name = clean_str(record_attributes[1].upper())
                entity_type = clean_str(record_attributes[2].upper())
                entity_description = clean_str(record_attributes[3])
                entities.append({
                    "title": entity_name,
                    "type": entity_type,
                    "description": entity_description,
                    "source_id": source_id,
                })

            if record_type == '"relationship"' and len(record_attributes) >= 5:
                source = clean_str(record_attributes[1].upper())
                target = clean_str(record_attributes[2].upper())
                edge_description = clean_str(record_attributes[3])
                try:
                    weight = float(record_attributes[-1])
                except ValueError:
                    weight = 1.0

                relationships.append({
                    "source": source,
                    "target": target,
                    "description": edge_description,
                    "source_id": source_id,
                    "weight": weight,
                })

        entities_df = pd.DataFrame(entities) if entities else self._empty_dfs()[0]
        relationships_df = pd.DataFrame(relationships) if relationships else self._empty_dfs()[1]
        
        # Формируем составные ключи для связей
        if not entities_df.empty and not relationships_df.empty:
            entity_map = dict(zip(entities_df['title'], entities_df['type']))
            mask = relationships_df["source"].isin(entity_map) & relationships_df["target"].isin(entity_map)
            relationships_df = relationships_df[mask].reset_index(drop=True)
            
            relationships_df['source'] = relationships_df['source'].apply(lambda x: f"{x}|{entity_map[x]}")
            relationships_df['target'] = relationships_df['target'].apply(lambda x: f"{x}|{entity_map[x]}")
        print(entities_df,relationships_df)
        return entities_df, relationships_df
    
    def _empty_dfs(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Возвращает пустые датафреймы нужной структуры."""
        return (
            pd.DataFrame(columns=["title", "type", "description", "source_id"]),
            pd.DataFrame(columns=["source", "target", "weight", "description", "source_id"])
        )


# --- Суммаризация Описаний (Summarization) ---

class GraphSummarizer:
    """Класс для объединения и суммаризации множественных описаний одной сущности/связи."""
    def __init__(self, llm_client: LLMClient, model: str, max_summary_length: int, max_input_tokens: int):
        self.llm = llm_client
        self.model = model
        self.max_summary_length = max_summary_length
        self.max_input_tokens = max_input_tokens

    def summarize_all(self, entities_df: pd.DataFrame, relationships_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info("Starting summarization process...")
        
        node_descriptions = [
            {"title": row.title, "description": self._summarize_item(row.title, list(set(row.description)))}
            for row in entities_df.itertuples(index=False)
        ]
        
        edge_descriptions = [
            {"source": row.source, "target": row.target, "description": self._summarize_item((row.source, row.target), list(set(row.description)))}
            for row in relationships_df.itertuples(index=False)
        ]

        return pd.DataFrame(node_descriptions), pd.DataFrame(edge_descriptions)

    def _summarize_item(self, item_id: Union[str, Tuple[str, str]], descriptions: List[str]) -> str:
        if not descriptions:
            return ""
        if len(descriptions) == 1:
            return descriptions[0]

        descriptions = sorted(descriptions)
        prompt_cost = self.llm.count_tokens(SUMMARIZE_PROMPT, self.model)
        usable_tokens = self.max_input_tokens - prompt_cost
        
        buffer = []
        result = ""

        for i, desc in enumerate(descriptions):
            usable_tokens -= self.llm.count_tokens(desc, self.model)
            buffer.append(desc)

            if (usable_tokens < 0 and len(buffer) > 1) or i == len(descriptions) - 1:
                result = self._call_llm_summarize(item_id, buffer)
                if i != len(descriptions) - 1:
                    buffer = [result]
                    usable_tokens = self.max_input_tokens - prompt_cost - self.llm.count_tokens(result, self.model)

        return result

    def _call_llm_summarize(self, item_id: Union[str, Tuple[str, str]], descriptions: List[str]) -> str:
        prompt = SUMMARIZE_PROMPT.format(
            entity_name=json.dumps(item_id, ensure_ascii=False),
            description_list=json.dumps(descriptions, ensure_ascii=False),
            max_length=self.max_summary_length,
        )
        response = self.llm.generate([{"role": "user", "content": prompt}], self.model)
        return response or ""

# --- Основной Пайплайн и Функции обработки данных ---

def merge_entities(entity_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    if not entity_dfs:
        return pd.DataFrame()
    return (
        pd.concat(entity_dfs, ignore_index=True)
        .groupby(["title", "type"], sort=False)
        .agg(description=("description", list), text_unit_ids=("source_id", list), frequency=("source_id", "count"))
        .reset_index()
    )

def merge_relationships(relationship_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    if not relationship_dfs:
        return pd.DataFrame()
    return (
        pd.concat(relationship_dfs, ignore_index=True)
        .groupby(["source", "target"], sort=False)
        .agg(description=("description", list), text_unit_ids=("source_id", list), weight=("weight", "sum"))
        .reset_index()
    )

def filter_orphan_relationships(relationships: pd.DataFrame, entities: pd.DataFrame) -> pd.DataFrame:
    """Удаляет связи, ссылающиеся на несуществующие узлы."""
    if relationships.empty or entities.empty:
        return relationships.iloc[0:0]

    entity_keys = set(entities["title"] + '|' + entities["type"])
    mask = relationships["source"].isin(entity_keys) & relationships["target"].isin(entity_keys)
    filtered = relationships[mask].reset_index(drop=True)
    
    if (dropped := len(relationships) - len(filtered)) > 0:
        logger.warning(f"Dropped {dropped} relationship(s) referencing non-existent entities.")
    return filtered

def run_extraction_pipeline(
    text_units: pd.DataFrame,
    extraction_model: str,
    summarization_model: str,
    entity_types: List[str],
    max_gleanings: int = 1,
    max_summary_length: int = 500,
    max_input_tokens: int = 4000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Главная функция для запуска всего процесса извлечения и суммаризации графа."""
    llm_client = LLMClient()
    extractor = GraphExtractor(llm_client, extraction_model, max_gleanings)
    summarizer = GraphSummarizer(llm_client, summarization_model, max_summary_length, max_input_tokens)

    entity_dfs, relationship_dfs = [], []

    # 1. Извлечение графа для каждого документа
    for _, row in text_units.iterrows():
        entities, relationships = extractor.extract(row['text'], entity_types, row['id'])
        entity_dfs.append(entities)
        relationship_dfs.append(relationships)

    # 2. Слияние результатов
    merged_entities = merge_entities(entity_dfs)
    merged_relationships = merge_relationships(relationship_dfs)
    valid_relationships = filter_orphan_relationships(merged_relationships, merged_entities)

    if merged_entities.empty or valid_relationships.empty:
        raise ValueError("Graph Extraction failed: No valid entities or relationships detected.")

    # 3. Суммаризация описаний
    entity_summaries, relationship_summaries = summarizer.summarize_all(merged_entities, valid_relationships)

    # 4. Обновление финальных датафреймов
    final_entities = merged_entities.drop(columns=["description"]).merge(entity_summaries, on="title", how="left")
    final_relationships = valid_relationships.drop(columns=["description"]).merge(relationship_summaries, on=["source", "target"], how="left")

    return final_entities, final_relationships

# --- Экспорт в Neo4j (Утилита из оригинального кода) ---
def save_nodes_to_neo4j_api(entity: pd.DataFrame, relations: pd.DataFrame, api_base_url: str, timeout: int = 30) -> Optional[Dict]:
    """Отправляет узлы и связи в API графовой базы (например, Neo4j)."""
    payload = {
        'entities': entity.to_dict(orient='records'),
        'relationships': relations.to_dict(orient='records')
    }
    try:
        response = requests.post(f"{api_base_url.rstrip('/')}/entities", json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to save to Neo4j API: {e}")
        return None



