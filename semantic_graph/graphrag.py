import asyncio
import hashlib
import html
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Coroutine, Dict, List, Optional, Tuple, Type, TypeVar, Union
import requests
import aiohttp
import pandas as pd
from openai import AsyncOpenAI
from pydantic import BaseModel

from config import (
    COMPLETION_DELIMITER,
    CONTINUE_PROMPT,
    DISAMBIGUATION_ENABLED,
    DISAMBIGUATION_SIMILARITY_THRESHOLD,
    GRAPH_EXTRACTION_PROMPT,
    LLM_API_KEY,
    LLM_URL,
    LOOP_PROMPT,
    RECORD_DELIMITER,
    SUMMARIZE_PROMPT,
    TOKENIZER_URL,
    TUPLE_DELIMITER,
)
from embeddings import disambiguate_entities

T = TypeVar("T", bound=BaseModel)

# --- Настройки логирования ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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


def _update_compound_title(compound: str, title_map: Dict[str, str]) -> str:
    """Update the title portion of a compound key ``"TITLE|TYPE"``.

    Used after disambiguation renames an entity title: relationships store
    sources/targets as ``"TITLE|TYPE"``, so the title part must be updated
    to match the new disambiguated title.
    """
    if "|" not in compound:
        return title_map.get(compound, compound)
    title, _, rest = compound.partition("|")
    if title in title_map:
        return f"{title_map[title]}|{rest}"
    return compound


class AsyncLLMClient:
    """Асинхронная обертка над AsyncOpenAI и aiohttp для токенизатора."""

    def __init__(self, base_url: str = LLM_URL, tokenizer_url: str = TOKENIZER_URL, api_key: str = LLM_API_KEY):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.tokenizer_url = tokenizer_url
        self._session: Optional[aiohttp.ClientSession] = None
        self._tokenizer_available: bool = True

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
        """Асинхронный подсчет токенов с fallback на word-count."""
        if not self._tokenizer_available:
            return int(len(text.split()) * 1.3)

        try:
            async with self.session.post(
                    self.tokenizer_url,
                    json={"model": model, "prompt": text},
                    timeout=aiohttp.ClientTimeout(total=10000)
            ) as response:
                response.raise_for_status()
                data = await response.json()
                return data.get('count', len(text) // 4)
        except Exception as e:
            self._tokenizer_available = False
            logger.warning(f"Tokenizer service unavailable, using word-count fallback. Error: {e}")
            return int(len(text.split()) * 1.3)

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
        self._cache: dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}
        self._cache_dir: Path = Path(os.getenv("EXTRACTION_CACHE_DIR", "./data/extraction_cache"))
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    async def extract(self, text: str, entity_types: List[str], source_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info(f"Extracting graph for document ID: {source_id}")

        # Check cache
        cache_key = self._compute_cache_key(text)
        if cache_key in self._cache:
            logger.info(f"Memory cache hit for {source_id} (key {cache_key[:12]}...)")
            return self._cache[cache_key]
        disk_result = self._load_from_disk_cache(cache_key)
        if disk_result is not None:
            logger.info(f"Disk cache hit for {source_id} (key {cache_key[:12]}...)")
            self._cache[cache_key] = disk_result
            return disk_result

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
        # Save to cache
        self._cache[cache_key] = out
        self._save_to_disk_cache(cache_key, out[0], out[1])
        return out

    async def extract_batch(
        self, chunks: List[Tuple[str, str]], entity_types: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Извлекает сущности и связи из нескольких чанков за один вызов LLM.

        Args:
            chunks: Список кортежей (text, source_id).
            entity_types: Типы сущностей для извлечения.

        Returns:
            Объединённые датафреймы сущностей и связей; source_id для каждой записи
            определяется по наилучшему совпадению имени сущности в тексте чанка.
        """
        if not chunks:
            return self._empty_dfs()

        if len(chunks) == 1:
            return await self.extract(chunks[0][0], entity_types, chunks[0][1])

        # Check cache for each chunk
        cached_entities: List[pd.DataFrame] = []
        cached_relationships: List[pd.DataFrame] = []
        uncached_items: List[Tuple[str, str, str]] = []

        for text, source_id in chunks:
            cache_key = self._compute_cache_key(text)
            if cache_key in self._cache:
                ent_df, rel_df = self._cache[cache_key]
                cached_entities.append(ent_df)
                cached_relationships.append(rel_df)
                continue
            disk_result = self._load_from_disk_cache(cache_key)
            if disk_result is not None:
                ent_df, rel_df = disk_result
                self._cache[cache_key] = (ent_df, rel_df)
                cached_entities.append(ent_df)
                cached_relationships.append(rel_df)
                continue
            uncached_items.append((text, source_id, cache_key))

        # Extract uncached items via LLM
        if uncached_items:
            batch_entities, batch_relationships = await self._extract_uncached(
                uncached_items, entity_types
            )
            cached_entities.append(batch_entities)
            cached_relationships.append(batch_relationships)
            # Cache individual chunk results by splitting by source_id
            for text, source_id, cache_key in uncached_items:
                chunk_ent = (
                    batch_entities[batch_entities["source_id"] == source_id]
                    if not batch_entities.empty
                    else batch_entities
                )
                chunk_rel = (
                    batch_relationships[batch_relationships["source_id"] == source_id]
                    if not batch_relationships.empty
                    else batch_relationships
                )
                self._cache[cache_key] = (chunk_ent, chunk_rel)
                self._save_to_disk_cache(cache_key, chunk_ent, chunk_rel)

        # Merge all results
        merged_entities = (
            pd.concat(cached_entities, ignore_index=True)
            if cached_entities
            else self._empty_dfs()[0]
        )
        merged_relationships = (
            pd.concat(cached_relationships, ignore_index=True)
            if cached_relationships
            else self._empty_dfs()[1]
        )
        return merged_entities, merged_relationships

    async def _extract_uncached(
        self,
        items: List[Tuple[str, str, str]],
        entity_types: List[str],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Extract entities/relationships for uncached chunks as a combined batch.

        Args:
            items: List of (text, source_id, cache_key) tuples.
            entity_types: Entity types to extract.

        Returns:
            Combined (entities_df, relationships_df) for all items.
        """
        chunk_map: Dict[str, str] = {}
        combined_parts: List[str] = []
        for text, source_id, _cache_key in items:
            chunk_map[source_id] = text
            combined_parts.append(f"--- CHUNK {source_id} ---\n{text}")

        combined_text = "\n\n".join(combined_parts)
        logger.info(
            f"Extracting graph for batch of {len(items)} uncached chunks: "
            f"{[sid for _, sid, _ in items]}"
        )

        prompt = GRAPH_EXTRACTION_PROMPT.format(
            input_text=combined_text,
            entity_types=",".join(entity_types)
        )
        messages = [{"role": "user", "content": prompt}]

        response_text = await self.llm.generate(messages, self.model)
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

        return self._parse_result_batch(full_result, chunk_map)

    def _parse_result_batch(
        self, result: str, chunk_map: Dict[str, str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Парсит результат батчевого извлечения и сопоставляет source_id для каждой записи."""
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
                source_id = self._find_source(entity_name, chunk_map)
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
                # Для связей используем source_id источника (первой сущности)
                source_id = self._find_source(source, chunk_map)
                relationships.append({
                    "source": source,
                    "target": target,
                    "description": edge_description,
                    "source_id": source_id,
                    "weight": weight,
                })

        entities_df = pd.DataFrame(entities) if entities else self._empty_dfs()[0]
        relationships_df = pd.DataFrame(relationships) if relationships else self._empty_dfs()[1]

        if not entities_df.empty and not relationships_df.empty:
            entity_map = dict(zip(entities_df['title'], entities_df['type']))
            mask = relationships_df["source"].isin(entity_map) & relationships_df["target"].isin(entity_map)
            relationships_df = relationships_df[mask].reset_index(drop=True)
            relationships_df['source'] = relationships_df['source'].apply(lambda x: f"{x}|{entity_map[x]}")
            relationships_df['target'] = relationships_df['target'].apply(lambda x: f"{x}|{entity_map[x]}")

        logger.debug(f"Batch extracted entities:\n{entities_df}\nRelationships:\n{relationships_df}")
        return entities_df, relationships_df

    @staticmethod
    def _find_source(entity_name: str, chunk_map: Dict[str, str]) -> str:
        """Сопоставляет имя сущности с source_id по содержимому чанков."""
        entity_lower = entity_name.lower()
        for source_id, text in chunk_map.items():
            if entity_lower in text.lower():
                return source_id
        # Fallback: возвращаем первый доступный source_id
        return next(iter(chunk_map.keys())) if chunk_map else "unknown"

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
                # Extract confidence if available (5th field, 1-10 scale)
                try:
                    entity_confidence = int(float(record_attributes[4])) if len(record_attributes) >= 5 else 5
                except (ValueError, IndexError):
                    entity_confidence = 5
                entities.append({
                    "title": entity_name,
                    "type": entity_type,
                    "description": entity_description,
                    "source_id": source_id,
                    "confidence": entity_confidence,
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
        logger.debug(f"Extracted entities:\n{entities_df}\nRelationships:\n{relationships_df}")
        return entities_df, relationships_df

    @staticmethod
    def _compute_cache_key(text: str) -> str:
        """Compute SHA256 hash of text for cache lookup."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _load_from_disk_cache(
        self, cache_key: str
    ) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Try to load cached extraction result from disk."""
        cache_file = self._cache_dir / f"{cache_key}.json"
        if not cache_file.exists():
            return None
        try:
            with open(cache_file) as f:
                data = json.load(f)
            entities_df = pd.DataFrame(data["entities"])
            relationships_df = pd.DataFrame(data["relationships"])
            logger.info(f"Cache hit for key {cache_key[:12]}...")
            return entities_df, relationships_df
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_key[:12]}...: {e}")
            return None

    def _save_to_disk_cache(
        self,
        cache_key: str,
        entities_df: pd.DataFrame,
        relationships_df: pd.DataFrame,
    ):
        """Save extraction result to disk cache."""
        cache_file = self._cache_dir / f"{cache_key}.json"
        try:
            data = {
                "entities": entities_df.to_dict(orient="records"),
                "relationships": relationships_df.to_dict(orient="records"),
            }
            with open(cache_file, "w") as f:
                json.dump(data, f)
            logger.debug(f"Cached extraction result for key {cache_key[:12]}...")
        except Exception as e:
            logger.warning(f"Failed to write cache {cache_key[:12]}...: {e}")

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
        max_chunks_per_batch: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Асинхронная версия основного пайплайна.

    Последовательность действий сохранена:
    1. Извлечение графа (параллельно по чанкам, с батчингом)
    2. Слияние результатов
    3. Суммаризация (параллельно по сущностям/связям)
    4. Формирование финальных датафреймов

    Args:
        max_chunks_per_batch: Количество чанков, объединяемых в один батч.
            При 1 поведение идентично оригинальному (один вызов LLM на чанк).
    """

    async with AsyncLLMClient() as llm_client:
        extractor = AsyncGraphExtractor(llm_client, extraction_model, max_gleanings)
        summarizer = AsyncGraphSummarizer(
            llm_client, summarization_model, max_summary_length, max_input_tokens
        )

        # ═══ Этап 1: Извлечение графа (с батчингом) ═══
        logger.info(f"Stage 1: Extracting graph from {len(text_units)} text units "
                     f"(batch size: {max_chunks_per_batch})...")

        chunks: List[Tuple[str, str]] = [
            (row['text'], row['id']) for _, row in text_units.iterrows()
        ]

        if max_chunks_per_batch <= 1:
            # Оригинальное поведение: один чанк = один вызов
            extraction_tasks = [
                extractor.extract(text, entity_types, source_id)
                for text, source_id in chunks
            ]
        else:
            # Группируем чанки в батчи
            batches = [
                chunks[i:i + max_chunks_per_batch]
                for i in range(0, len(chunks), max_chunks_per_batch)
            ]
            logger.info(f"Created {len(batches)} batch(es) for extraction")
            extraction_tasks = [
                extractor.extract_batch(batch, entity_types)
                for batch in batches
            ]

        extraction_results = await asyncio.gather(*extraction_tasks)
        logger.info("Stage 1 complete: All extractions finished.")
        entity_dfs = [res[0] for res in extraction_results]
        relationship_dfs = [res[1] for res in extraction_results]

        # ═══ Stage 1.5: Embedding-based entity disambiguation ═══
        if DISAMBIGUATION_ENABLED and entity_dfs:
            all_entities_df = pd.concat(entity_dfs, ignore_index=True)
            if not all_entities_df.empty:
                logger.info("Stage 1.5: Disambiguating entities via embedding similarity...")
                entity_records = all_entities_df.to_dict(orient="records")
                try:
                    disambiguated = await disambiguate_entities(
                        entity_records,
                        llm_client.session,
                        threshold=DISAMBIGUATION_SIMILARITY_THRESHOLD,
                    )
                    # Build old-title → new-title mapping for renamed entities
                    title_map: Dict[str, str] = {}
                    for old, new in zip(entity_records, disambiguated):
                        if old.get("title") != new.get("title"):
                            title_map[old["title"]] = new["title"]

                    if title_map:
                        # Update relationship DataFrames to reference new titles
                        for rel_df in relationship_dfs:
                            if rel_df.empty:
                                continue
                            for col in ("source", "target"):
                                # Compound keys are "TITLE|TYPE"; replace the title part
                                rel_df[col] = rel_df[col].apply(
                                    lambda x: _update_compound_title(x, title_map)
                                )

                    # Replace entity_dfs with the disambiguated result
                    entity_dfs = [pd.DataFrame(disambiguated)]
                    logger.info(
                        "Stage 1.5 complete: %d title(s) renamed.",
                        len(title_map),
                    )
                except Exception:
                    logger.warning(
                        "Entity disambiguation failed, proceeding with original entity titles.",
                        exc_info=True,
                    )

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




