import asyncio
import html
import json
import logging
import re
from typing import Any, Coroutine, Dict, List, Optional, Tuple, Type, TypeVar, Union
import requests
import aiohttp
import pandas as pd
from openai import AsyncOpenAI
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

# --- Настройки логирования ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# --- Константы и Промпты ---
#TOKENIZER_URL = "http://192.168.19.127:9886/tokenize"
#LLM_URL = 'http://192.168.19.127:9886/v1'
TOKENIZER_URL = "http://localhost:9886/tokenize"
LLM_URL = 'http://localhost:9886/v1'
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


class AsyncLLMClient:
    """Асинхронная обертка над AsyncOpenAI и aiohttp для токенизатора."""

    def __init__(self, base_url: str = LLM_URL, tokenizer_url: str = TOKENIZER_URL, api_key: str = 'EMPTY'):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.tokenizer_url = tokenizer_url
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()

    @property
    def session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            raise RuntimeError("AsyncLLMClient session not started. Use 'async with'.")
        return self._session

    async def generate(
            self,
            messages: List[Dict[str, str]],
            model: str,
            **kwargs
    ) -> Optional[str]:
        """Асинхронный вызов LLM. Принимает любые доп. параметры (как в твоем примере)."""
        try:
            logger.info(f"Generating content with model: {model}")
            response = await self.client.chat.completions.create(
                messages=messages,
                model=model,
                **kwargs
            )
            raw_content = response.choices[0].message.content
            return remove_think_tags(raw_content)
        except Exception as e:
            logger.error(f"Failed to call LLM: {e}")
            return None

    async def count_tokens(self, text: str, model: str) -> int:
        """Асинхронный подсчет токенов."""
        try:
            async with self.session.post(
                    self.tokenizer_url,
                    json={"model": model, "prompt": text},
                    timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                response.raise_for_status()
                data = await response.json()
                return data.get('count', len(text) // 4)
        except Exception as e:
            logger.warning(f"Token counting failed, using fallback. Error: {e}")
            return len(text) // 4 + 1

    async def generate_structured(
            self,
            messages: List[Dict[str, str]],
            model: str,
            response_model: Type[T],
            **kwargs,
    ) -> Optional[T]:
        """Асинхронный вызов LLM с парсингом JSON-ответа в Pydantic-модель."""
        try:
            logger.info(f"Generating structured content with model: {model}")
            response = await self.client.chat.completions.create(
                messages=messages,
                model=model,
                response_format={"type": "json_object"},
                **kwargs,
            )
            raw_content = response.choices[0].message.content
            if not raw_content:
                return None
            cleaned = remove_think_tags(raw_content)
            json_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, re.DOTALL)
            json_text = json_match.group(1) if json_match else cleaned
            return response_model.model_validate(json.loads(json_text))
        except Exception as e:
            logger.error(f"Failed to call LLM with structured output: {e}")
            return None
# --- Извлечение Графа (Extraction) ---
class AsyncGraphExtractor:
    """Асинхронный класс для извлечения сущностей и связей из сырого текста."""

    def __init__(self, llm_client: AsyncLLMClient, model: str, max_gleanings: int):
        self.llm = llm_client
        self.model = model
        self.max_gleanings = max_gleanings

    async def extract(self, text: str, entity_types: List[str], source_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info(f"Extracting graph for document ID: {source_id}")

        prompt = GRAPH_EXTRACTION_PROMPT.format(
            input_text=text,
            entity_types=",".join(entity_types)
        )
        messages = [{"role": "user", "content": prompt}]

        response_text = await self.llm.generate(messages, self.model)
        logger.info(f"Response from LLM: len={len(response_text) if response_text else 0}, "
                f"type={type(response_text).__name__}, "
                f"first_100_chars={response_text if response_text else 'EMPTY'!r}")
        if not response_text:
            return self._empty_dfs()

        full_result = response_text
        messages.append({"role": "assistant", "content": response_text})

        for _ in range(self.max_gleanings):
            messages.append({"role": "user", "content": CONTINUE_PROMPT})
            continuation = await self.llm.generate(messages, self.model)
            if not continuation or COMPLETION_DELIMITER in continuation:
                full_result += continuation
                break

            full_result += continuation
            messages.append({"role": "assistant", "content": continuation})

            messages.append({"role": "user", "content": LOOP_PROMPT})
            loop_decision = await self.llm.generate(messages, self.model, max_tokens=5)
            if not loop_decision or loop_decision.strip().upper() != "Y":
                break
        out = self._parse_result(full_result, source_id)
        logger.info(f"parsing out: len={out[0] if response_text else 0}, ")
        return out

    # Методы _parse_result и _empty_dfs не выполняют I/O и остаются синхронными
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
        return (
            pd.DataFrame(columns=["title", "type", "description", "source_id"]),
            pd.DataFrame(columns=["source", "target", "weight", "description", "source_id"])
        )


# --- АСИНХРОННАЯ Суммаризация Описаний ---

class AsyncGraphSummarizer:
    """Асинхронный класс для объединения и суммаризации описаний."""

    def __init__(self, llm_client: AsyncLLMClient, model: str, max_summary_length: int, max_input_tokens: int):
        self.llm = llm_client
        self.model = model
        self.max_summary_length = max_summary_length
        self.max_input_tokens = max_input_tokens

    async def summarize_all(self, entities_df: pd.DataFrame, relationships_df: pd.DataFrame) -> Tuple[
        pd.DataFrame, pd.DataFrame]:
        logger.info("Starting async summarization process...")

        # Создаем задачи для параллельной суммаризации
        node_tasks = [
            self._summarize_item(row.title, list(set(row.description)))
            for row in entities_df.itertuples()
        ]
        edge_tasks = [
            self._summarize_item((row.source, row.target), list(set(row.description)))
            for row in relationships_df.itertuples()
        ]

        # Выполняем задачи параллельно
        node_summary_list = await asyncio.gather(*node_tasks)
        edge_summary_list = await asyncio.gather(*edge_tasks)

        # Собираем результаты
        summarized_entities = entities_df[['title']].copy()
        summarized_entities['description'] = node_summary_list

        summarized_relationships = relationships_df[['source', 'target']].copy()
        summarized_relationships['description'] = edge_summary_list

        return summarized_entities, summarized_relationships

    async def _summarize_item(self, item_id: Union[str, Tuple[str, str]], descriptions: List[str]) -> str:
        if not descriptions:
            return ""
        if len(descriptions) == 1:
            return descriptions[0]

        descriptions = sorted(descriptions)
        prompt_cost = await self.llm.count_tokens(SUMMARIZE_PROMPT, self.model)
        usable_tokens = self.max_input_tokens - prompt_cost

        buffer = []
        result = ""

        for i, desc in enumerate(descriptions):
            usable_tokens -= await self.llm.count_tokens(desc, self.model)
            buffer.append(desc)

            # Если токены закончились или это последняя итерация
            if (usable_tokens < 0 and len(buffer) > 1) or i == len(descriptions) - 1:
                # Если в буффере только один элемент после предыдущей суммаризации
                current_text = buffer[0] if len(buffer) == 1 else await self._call_llm_summarize(item_id, buffer)

                # Если это не конец, готовимся к следующему циклу
                if i < len(descriptions) - 1:
                    buffer = [current_text]
                    token_cost = await self.llm.count_tokens(current_text, self.model)
                    usable_tokens = self.max_input_tokens - prompt_cost - token_cost
                else:  # Если это конец, то это и есть финальный результат
                    result = current_text

        return result

    async def _call_llm_summarize(self, item_id: Union[str, Tuple[str, str]], descriptions: List[str]) -> str:
        prompt = SUMMARIZE_PROMPT.format(
            entity_name=json.dumps(item_id, ensure_ascii=False),
            description_list=json.dumps(descriptions, ensure_ascii=False),
            max_length=self.max_summary_length,
        )
        response = await self.llm.generate([{"role": "user", "content": prompt}], self.model)
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

def finalize_entities(
    entities_table: pd.DataFrame,
    degree_map: dict[str, int],
) -> pd.DataFrame:
    """
    Дополняет датафрейм сущностей столбцами 'degree'.
    Дедуплирует по title (сохраняет первую попавшуюся запись),
    присваивает degree по карте degree_map, и human_readable_id по порядку.

    Args:
        entities_table (pd.DataFrame): Таблица сущностей.
        degree_map (dict[str, int]): Карта степеней для сущностей.

    Returns:
        pd.DataFrame: Дедуплицированный и дополненный датафрейм.
    """
    df = entities_table.copy()
    # Сброс индекса для надёжности
    df = df.reset_index(drop=True)
    # Удаляем дубликаты по title (оставляем первую запись)
    df = df.drop_duplicates(subset=["title", "type"], keep="first").reset_index(drop=True)
    # degree
    df["degree"] = (df["title"] + "|" + df["type"]).map(degree_map).fillna(0).astype(int)
    return df

def finalize_relationships(
    relationships_table: pd.DataFrame,
    degree_map: dict[str, int],
) -> pd.DataFrame:
    """
    Дополняет датафрейм связей столбцом 'combined_degree'.
    Дедуплирует по паре (source, target), присваивает combined_degree как сумму степеней.
    """
    df = relationships_table.copy()
    # Сброс индекса для надёжности
    df = df.reset_index(drop=True)
    # Удаляем дубликаты по (source, target) (оставляем первую запись)
    df = df.drop_duplicates(subset=["source", "target"], keep="first").reset_index(drop=True)
    # combined_degree
    df["combined_degree"] = (
        df["source"].map(degree_map).fillna(0).astype(int) +
        df["target"].map(degree_map).fillna(0).astype(int)
    )
    # human_readable_id (можно добавить если необходимо, как по примеру с entities)
    return df


def _build_degree_map(
    relationships_table: pd.DataFrame,
) -> dict[str, int]:    
    """
    Строит карту степеней для связей.
    Args:
        relationships_table (pd.DataFrame): Таблица связей.

    Returns:
        dict[str, int]: Карта степеней для связей.
    """

    seen: set[tuple[str, str]] = set()
    degree: dict[str, int] = {}
    for row in relationships_table.itertuples():
        lo, hi = sorted((row.source, row.target))
        if (lo, hi) not in seen:
            seen.add((lo, hi))
            degree[lo] = degree.get(lo, 0) + 1
            degree[hi] = degree.get(hi, 0) + 1
    return degree

async def run_extraction_pipeline_async(
        text_units: pd.DataFrame,
        extraction_model: str,
        summarization_model: str,
        entity_types: List[str],
        max_gleanings: int = 1,
        max_summary_length: int = 500,
        max_input_tokens: int = 4000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Асинхронная версия основного пайплайна.

    Последовательность действий сохранена:
    1. Извлечение графа (параллельно по чанкам)
    2. Слияние результатов
    3. Суммаризация (параллельно по сущностям/связям)
    4. Формирование финальных датафреймов
    """

    async with AsyncLLMClient() as llm_client:
        extractor = AsyncGraphExtractor(llm_client, extraction_model, max_gleanings)
        summarizer = AsyncGraphSummarizer(
            llm_client, summarization_model, max_summary_length, max_input_tokens
        )

        # ═══ Этап 1: Извлечение графа для каждого документа (параллельно) ═══
        logger.info(f"Stage 1: Extracting graph from {len(text_units)} text units...")
        extraction_tasks = [
            extractor.extract(row['text'], entity_types, row['id'])
            for _, row in text_units.iterrows()
        ]
        extraction_results = await asyncio.gather(*extraction_tasks)
        logger.info("Stage 1 complete: All extractions finished.")
        logger.info(f"extraction results {extraction_results}")
        entity_dfs = [res[0] for res in extraction_results]
        relationship_dfs = [res[1] for res in extraction_results]
        
        # ═══ Этап 2: Слияние результатов ═══
        logger.info("Stage 2: Merging extraction results...")
        merged_entities = merge_entities(entity_dfs)
        merged_relationships = merge_relationships(relationship_dfs)
        valid_relationships = filter_orphan_relationships(merged_relationships, merged_entities)

        if merged_entities.empty:
            raise ValueError("Graph Extraction failed: No valid entities detected.")

        logger.info(
            f"Stage 2 complete: {len(merged_entities)} entities, "
            f"{len(valid_relationships)} relationships"
        )

        # ═══ Этап 3: Суммаризация описаний (параллельно) ═══
        logger.info("Stage 3: Summarizing descriptions...")
        entity_summaries, relationship_summaries = await summarizer.summarize_all(
            merged_entities, valid_relationships
        )
        logger.info("Stage 3 complete: All descriptions summarized.")

        # ═══ Этап 4: Формирование финальных датафреймов ═══
        logger.info("Stage 4: Building final DataFrames...")
        final_entities = merged_entities.drop(columns=["description"]).merge(
            entity_summaries, on="title", how="left"
        )

        if not valid_relationships.empty:
            final_relationships = valid_relationships.drop(columns=["description"]).merge(
                relationship_summaries, on=["source", "target"], how="left"
            )
        else:
            final_relationships = pd.DataFrame(
                columns=["source", "target", "weight", "description", "source_id", "text_unit_ids"]
            )
        degree_map =  _build_degree_map(final_relationships)
        final_entities = finalize_entities(final_entities, degree_map)
        final_relationships = finalize_relationships(final_relationships, degree_map)
        logger.info("Stage 4 complete: Pipeline finished successfully!")
        return final_entities, final_relationships




