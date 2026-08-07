"""Генерация отчётов по сообществам графа знаний."""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Hashable, Iterable
from dataclasses import dataclass
from hashlib import sha512
from typing import Any, cast

import pandas as pd
from pydantic import BaseModel, Field

import config
from config import (
    COMMUNITY_REPORT_PROMPT, INCREMENTAL_COMMUNITY_REPORTS,
    INPUT_TEXT_KEY, MAX_LENGTH_KEY,
)
from graphrag import AsyncLLMClient

logger = logging.getLogger(__name__)


# --- Типы и модели ответа LLM ---


class FindingModel(BaseModel):
    summary: str = Field(description="The summary of the finding.")
    explanation: str = Field(description="An explanation of the finding.")


class CommunityReportResponse(BaseModel):
    title: str = Field(description="The title of the report.")
    summary: str = Field(description="A summary of the report.")
    findings: list[FindingModel] = Field(description="A list of findings in the report.")
    rating: float = Field(description="The rating of the report.")
    rating_explanation: str = Field(description="An explanation of the rating.")


@dataclass
class CommunityReportsResult:
    output: str
    structured_output: CommunityReportResponse | None


# --- Утилиты для DataFrame ---


def gen_sha512_hash(item: dict[str, Any], hashcode: Iterable[str]) -> str:
    hashed = "".join([str(item[column]) for column in hashcode])
    return f"{sha512(hashed.encode('utf-8'), usedforsecurity=False).hexdigest()}"


def drop_columns(df: pd.DataFrame, *column: str) -> pd.DataFrame:
    return df.drop(list(column), axis=1)


def where_column_equals(df: pd.DataFrame, column: str, value: Any) -> pd.DataFrame:
    return cast("pd.DataFrame", df[df[column] == value])


def antijoin(df: pd.DataFrame, exclude: pd.DataFrame, column: str) -> pd.DataFrame:
    return df.loc[~df.loc[:, column].isin(exclude.loc[:, column])]


def join(
    left: pd.DataFrame, right: pd.DataFrame, key: str, strategy: str = "left"
) -> pd.DataFrame:
    return left.merge(right, on=key, how=strategy)


def union(*frames: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(list(frames))


def select(df: pd.DataFrame, *columns: str) -> pd.DataFrame:
    return cast("pd.DataFrame", df[list(columns)])


def get_levels(
    df: pd.DataFrame, level_column: str = config.COMMUNITY_LEVEL
) -> list[int]:
    levels = df[level_column].dropna().unique()
    levels = [int(lvl) for lvl in levels if lvl != -1]
    return sorted(levels, reverse=True)


# --- Подготовка данных ---


def explode_communities(
    communities: pd.DataFrame, entities: pd.DataFrame
) -> pd.DataFrame:
    community_join = communities.explode("entity_ids").loc[
        :, ["community", "level", "entity_ids"]
    ]
    nodes = entities.merge(
        community_join, left_on="id", right_on="entity_ids", how="left"
    )
    return nodes.loc[nodes.loc[:, config.COMMUNITY_ID] != -1]


def _prep_nodes(input: pd.DataFrame) -> pd.DataFrame:
    input.loc[:, config.DESCRIPTION] = input.loc[:, config.DESCRIPTION].fillna(
        "No Description"
    )
    input.loc[:, config.NODE_DETAILS] = input.loc[
        :,
        [
            config.ID,
            config.TITLE,
            config.DESCRIPTION,
            config.NODE_DEGREE,
        ],
    ].to_dict(orient="records")
    return input


def _prep_edges(input: pd.DataFrame) -> pd.DataFrame:
    input.fillna(value={config.DESCRIPTION: "No Description"}, inplace=True)
    input.loc[:, config.EDGE_DETAILS] = input.loc[
        :,
        [
            config.ID,
            config.EDGE_SOURCE,
            config.EDGE_TARGET,
            config.DESCRIPTION,
            config.EDGE_DEGREE,
        ],
    ].to_dict(orient="records")
    return input


# --- Построение контекста ---


def _edge_key(
    edge: dict,
    source_column: str = config.EDGE_SOURCE,
    target_column: str = config.EDGE_TARGET,
) -> tuple[str, str]:
    return (edge[source_column], edge[target_column])


async def sort_context(
    local_context: list[dict],
    llm: AsyncLLMClient,
    model: str,
    sub_community_reports: list[dict] | None = None,
    max_context_tokens: int | None = None,
    node_name_column: str = config.ID,
    node_details_column: str = config.NODE_DETAILS,
    edge_id_column: str = config.ID,
    edge_details_column: str = config.EDGE_DETAILS,
    edge_degree_column: str = config.EDGE_DEGREE,
    edge_source_column: str = config.EDGE_SOURCE,
    edge_target_column: str = config.EDGE_TARGET,
) -> str:
    def _get_context_string(
        entities: list[dict],
        edges: list[dict],
        reports: list[dict] | None = None,
    ) -> str:
        contexts = []
        if reports:
            report_df = pd.DataFrame(reports)
            if not report_df.empty:
                contexts.append(
                    f"----Reports-----\n{report_df.to_csv(index=False, sep=',')}"
                )
        for label, data in [("Entities", entities), ("Relationships", edges)]:
            if data:
                data_df = pd.DataFrame(data)
                if not data_df.empty:
                    contexts.append(
                        f"-----{label}-----\n{data_df.to_csv(index=False, sep=',')}"
                    )
        return "\n\n".join(contexts)

    edges = [
        e
        for record in local_context
        for e in record.get(edge_details_column, [])
        if isinstance(e, dict)
    ]
    node_details = {
        record[node_name_column]: record[node_details_column]
        for record in local_context
    }

    edges.sort(key=lambda x: (-x.get(edge_degree_column, 0), x.get(edge_id_column, "")))

    edge_keys: set[tuple[str, str]] = set()
    node_ids: set[str] = set()
    sorted_edges, sorted_nodes = [], []
    context_string = ""

    for edge in edges:
        source, target = edge[edge_source_column], edge[edge_target_column]
        for node in [node_details.get(source), node_details.get(target)]:
            if node:
                node_id = node.get(config.ID, node.get(node_name_column))
                if node_id not in node_ids:
                    node_ids.add(node_id)
                    sorted_nodes.append(node)

        edge_key = _edge_key(edge, edge_source_column, edge_target_column)
        if edge_key not in edge_keys:
            edge_keys.add(edge_key)
            sorted_edges.append(edge)

        new_context_string = _get_context_string(
            sorted_nodes, sorted_edges, sub_community_reports
        )
        if (
            max_context_tokens
            and await llm.count_tokens(new_context_string, model) > max_context_tokens
        ):
            break
        context_string = new_context_string

    return context_string or _get_context_string(
        sorted_nodes, sorted_edges, sub_community_reports
    )


async def parallel_sort_context_batch(
    community_df: pd.DataFrame,
    llm: AsyncLLMClient,
    model: str,
    max_context_tokens: int,
) -> pd.DataFrame:
    context_strings = []
    for context_list in community_df[config.ALL_CONTEXT]:
        context_strings.append(
            await sort_context(
                context_list, llm, model, max_context_tokens=max_context_tokens
            )
        )
    community_df = community_df.copy()
    community_df[config.CONTEXT_STRING] = context_strings

    sizes = []
    for context_string in community_df[config.CONTEXT_STRING]:
        sizes.append(await llm.count_tokens(context_string, model))
    community_df[config.CONTEXT_SIZE] = sizes
    community_df[config.CONTEXT_EXCEED_FLAG] = (
        community_df[config.CONTEXT_SIZE] > max_context_tokens
    )
    return community_df


async def build_mixed_context(
    context: list[dict],
    llm: AsyncLLMClient,
    model: str,
    max_context_tokens: int,
) -> str:
    sorted_context = sorted(
        context, key=lambda x: x[config.CONTEXT_SIZE], reverse=True
    )

    substitute_reports = []
    final_local_contexts = []
    exceeded_limit = True
    context_string = ""

    for idx, sub_community_context in enumerate(sorted_context):
        if exceeded_limit:
            if sub_community_context[config.FULL_CONTENT]:
                substitute_reports.append({
                    config.COMMUNITY_ID: sub_community_context[config.SUB_COMMUNITY],
                    config.FULL_CONTENT: sub_community_context[config.FULL_CONTENT],
                })
            else:
                final_local_contexts.extend(sub_community_context[config.ALL_CONTEXT])
                continue

            remaining_local_context = []
            for rid in range(idx + 1, len(sorted_context)):
                remaining_local_context.extend(sorted_context[rid][config.ALL_CONTEXT])
            new_context_string = await sort_context(
                local_context=remaining_local_context + final_local_contexts,
                llm=llm,
                model=model,
                sub_community_reports=substitute_reports,
            )
            if await llm.count_tokens(new_context_string, model) <= max_context_tokens:
                exceeded_limit = False
                context_string = new_context_string
                break

    if exceeded_limit:
        substitute_reports = []
        for sub_community_context in sorted_context:
            substitute_reports.append({
                config.COMMUNITY_ID: sub_community_context[config.SUB_COMMUNITY],
                config.FULL_CONTENT: sub_community_context[config.FULL_CONTENT],
            })
            new_context_string = pd.DataFrame(substitute_reports).to_csv(
                index=False, sep=","
            )
            if await llm.count_tokens(new_context_string, model) > max_context_tokens:
                break
            context_string = new_context_string
    return context_string


async def _prepare_reports_at_level(
    node_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    llm: AsyncLLMClient,
    model: str,
    level: int,
    max_context_tokens: int = 16000,
) -> pd.DataFrame:
    level_node_df = node_df[node_df[config.COMMUNITY_LEVEL] == level]
    logger.info("Number of nodes at level=%s => %s", level, len(level_node_df))
    nodes_set = set(level_node_df[config.ID])

    level_edge_df = edge_df[
        edge_df.loc[:, config.EDGE_SOURCE].isin(nodes_set)
        & edge_df.loc[:, config.EDGE_TARGET].isin(nodes_set)
    ]
    level_edge_df.loc[:, config.EDGE_DETAILS] = level_edge_df.loc[
        :,
        [
            config.ID,
            config.EDGE_SOURCE,
            config.EDGE_TARGET,
            config.DESCRIPTION,
            config.EDGE_DEGREE,
        ],
    ].to_dict(orient="records")

    source_edges = (
        level_edge_df
        .groupby(config.EDGE_SOURCE)
        .agg({config.EDGE_DETAILS: "first"})
        .reset_index()
        .rename(columns={config.EDGE_SOURCE: config.ID})
    )
    target_edges = (
        level_edge_df
        .groupby(config.EDGE_TARGET)
        .agg({config.EDGE_DETAILS: "first"})
        .reset_index()
        .rename(columns={config.EDGE_TARGET: config.ID})
    )

    merged_node_df = level_node_df.merge(
        source_edges, on=config.ID, how="left"
    ).merge(target_edges, on=config.ID, how="left")

    merged_node_df.loc[:, config.EDGE_DETAILS] = merged_node_df.loc[
        :, f"{config.EDGE_DETAILS}_x"
    ].combine_first(merged_node_df.loc[:, f"{config.EDGE_DETAILS}_y"])

    merged_node_df.drop(
        columns=[f"{config.EDGE_DETAILS}_x", f"{config.EDGE_DETAILS}_y"], inplace=True
    )

    merged_node_df = (
        merged_node_df
        .groupby([
            config.ID,
            config.COMMUNITY_ID,
            config.COMMUNITY_LEVEL,
            config.NODE_DEGREE,
        ])
        .agg({
            config.NODE_DETAILS: "first",
            config.EDGE_DETAILS: lambda x: list(x.dropna()),
        })
        .reset_index()
    )

    merged_node_df[config.ALL_CONTEXT] = merged_node_df.loc[
        :,
        [
            config.ID,
            config.NODE_DEGREE,
            config.NODE_DETAILS,
            config.EDGE_DETAILS,
        ],
    ].to_dict(orient="records")

    community_df = (
        merged_node_df
        .groupby(config.COMMUNITY_ID)
        .agg({config.ALL_CONTEXT: list})
        .reset_index()
    )

    return await parallel_sort_context_batch(
        community_df, llm, model, max_context_tokens=max_context_tokens
    )


async def build_local_context(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    llm: AsyncLLMClient,
    model: str,
    max_context_tokens: int = 16000,
) -> pd.DataFrame:
    levels = get_levels(nodes, config.COMMUNITY_LEVEL)
    dfs = []
    for level in levels:
        communities_at_level_df = await _prepare_reports_at_level(
            nodes, edges, llm, model, level, max_context_tokens
        )
        communities_at_level_df.loc[:, config.COMMUNITY_LEVEL] = level
        dfs.append(communities_at_level_df)
    return pd.concat(dfs)


def _drop_community_level(df: pd.DataFrame) -> pd.DataFrame:
    return drop_columns(df, config.COMMUNITY_LEVEL)


def _at_level(level: int, df: pd.DataFrame) -> pd.DataFrame:
    return where_column_equals(df, config.COMMUNITY_LEVEL, level)


def _antijoin_reports(df: pd.DataFrame, reports: pd.DataFrame) -> pd.DataFrame:
    return antijoin(df, reports, config.COMMUNITY_ID)


async def _sort_and_trim_context(
    df: pd.DataFrame,
    llm: AsyncLLMClient,
    model: str,
    max_context_tokens: int,
) -> pd.Series:
    results = []
    for context_list in df[config.ALL_CONTEXT]:
        results.append(
            await sort_context(
                context_list, llm, model, max_context_tokens=max_context_tokens
            )
        )
    return pd.Series(results, index=df.index)


async def _build_mixed_context_series(
    df: pd.DataFrame,
    llm: AsyncLLMClient,
    model: str,
    max_context_tokens: int,
) -> pd.Series:
    results = []
    for context_list in df[config.ALL_CONTEXT]:
        results.append(
            await build_mixed_context(
                context_list, llm, model, max_context_tokens=max_context_tokens
            )
        )
    return pd.Series(results, index=df.index)


def _get_subcontext_df(
    level: int, report_df: pd.DataFrame, local_context_df: pd.DataFrame
) -> pd.DataFrame:
    sub_report_df = _drop_community_level(_at_level(level, report_df))
    sub_context_df = _at_level(level, local_context_df)
    sub_context_df = join(sub_context_df, sub_report_df, config.COMMUNITY_ID)
    sub_context_df.rename(
        columns={config.COMMUNITY_ID: config.SUB_COMMUNITY}, inplace=True
    )
    return sub_context_df


async def _get_community_df(
    level: int,
    invalid_context_df: pd.DataFrame,
    sub_context_df: pd.DataFrame,
    community_hierarchy_df: pd.DataFrame,
    llm: AsyncLLMClient,
    model: str,
    max_context_tokens: int,
) -> pd.DataFrame:
    community_df = _drop_community_level(_at_level(level, community_hierarchy_df))
    invalid_community_ids = select(invalid_context_df, config.COMMUNITY_ID)
    subcontext_selection = select(
        sub_context_df,
        config.SUB_COMMUNITY,
        config.FULL_CONTENT,
        config.ALL_CONTEXT,
        config.CONTEXT_SIZE,
    )

    invalid_communities = join(
        community_df, invalid_community_ids, config.COMMUNITY_ID, "inner"
    )
    community_df = join(
        invalid_communities, subcontext_selection, config.SUB_COMMUNITY
    )
    community_df[config.ALL_CONTEXT] = community_df.apply(
        lambda x: {
            config.SUB_COMMUNITY: x[config.SUB_COMMUNITY],
            config.ALL_CONTEXT: x[config.ALL_CONTEXT],
            config.FULL_CONTENT: x[config.FULL_CONTENT],
            config.CONTEXT_SIZE: x[config.CONTEXT_SIZE],
        },
        axis=1,
    )
    community_df = (
        community_df
        .groupby(config.COMMUNITY_ID)
        .agg({config.ALL_CONTEXT: list})
        .reset_index()
    )
    community_df[config.CONTEXT_STRING] = await _build_mixed_context_series(
        community_df, llm, model, max_context_tokens
    )
    community_df[config.COMMUNITY_LEVEL] = level
    return community_df


async def build_level_context(
    report_df: pd.DataFrame | None,
    community_hierarchy_df: pd.DataFrame,
    local_context_df: pd.DataFrame,
    llm: AsyncLLMClient,
    model: str,
    level: int,
    max_context_tokens: int,
) -> pd.DataFrame:
    level_context_df = local_context_df.loc[
        local_context_df.loc[:, config.COMMUNITY_LEVEL] == level
    ]
    valid_context_df = level_context_df.loc[
        ~level_context_df.loc[:, config.CONTEXT_EXCEED_FLAG]
    ]
    invalid_context_df = level_context_df.loc[
        level_context_df.loc[:, config.CONTEXT_EXCEED_FLAG]
    ]

    if invalid_context_df.empty:
        return valid_context_df

    if report_df is None or report_df.empty:
        invalid_context_df = invalid_context_df.copy()
        invalid_context_df.loc[:, config.CONTEXT_STRING] = await _sort_and_trim_context(
            invalid_context_df, llm, model, max_context_tokens
        )
        sizes = []
        for context_string in invalid_context_df[config.CONTEXT_STRING]:
            sizes.append(await llm.count_tokens(context_string, model))
        invalid_context_df[config.CONTEXT_SIZE] = sizes
        invalid_context_df[config.CONTEXT_EXCEED_FLAG] = False
        return union(valid_context_df, invalid_context_df)

    level_context_df = _antijoin_reports(level_context_df, report_df)

    sub_context_df = _get_subcontext_df(level + 1, report_df, local_context_df)
    community_df = await _get_community_df(
        level,
        invalid_context_df,
        sub_context_df,
        community_hierarchy_df,
        llm,
        model,
        max_context_tokens,
    )

    remaining_df = _antijoin_reports(invalid_context_df, community_df)
    remaining_df = remaining_df.copy()
    remaining_df.loc[:, config.CONTEXT_STRING] = await _sort_and_trim_context(
        remaining_df, llm, model, max_context_tokens
    )

    result = union(valid_context_df, community_df, remaining_df)
    sizes = []
    for context_string in result[config.CONTEXT_STRING]:
        sizes.append(await llm.count_tokens(context_string, model))
    result[config.CONTEXT_SIZE] = sizes
    result[config.CONTEXT_EXCEED_FLAG] = False
    return result


# --- Извлечение отчётов через LLM ---


class AsyncCommunityReportExtractor:
    """Асинхронный генератор отчётов по сообществам."""

    def __init__(
        self,
        llm_client: AsyncLLMClient,
        model: str,
        extraction_prompt: str,
        max_report_length: int,
    ):
        self.llm = llm_client
        self.model = model
        self.extraction_prompt = extraction_prompt
        self.max_report_length = max_report_length

    async def extract(self, input_text: str) -> CommunityReportsResult:
        prompt = self.extraction_prompt.format(**{
            INPUT_TEXT_KEY: input_text,
            MAX_LENGTH_KEY: str(self.max_report_length),
        })
        output = await self.llm.generate_structured(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            response_model=CommunityReportResponse,
        )
        text_output = self._get_text_output(output) if output else ""
        return CommunityReportsResult(structured_output=output, output=text_output)

    def _get_text_output(self, report: CommunityReportResponse) -> str:
        report_sections = "\n\n".join(
            f"## {f.summary}\n\n{f.explanation}" for f in report.findings
        )
        return f"# {report.title}\n\n{report.summary}\n\n{report_sections}"


async def _generate_report(
    extractor: AsyncCommunityReportExtractor,
    community_id: int,
    community_level: int,
    community_context: str,
) -> dict[str, Any] | None:
    try:
        results = await extractor.extract(community_context)
        report = results.structured_output
        if report is None:
            logger.warning("No report found for community: %s", community_id)
            return None

        return {
            config.COMMUNITY_ID: community_id,
            config.FULL_CONTENT: results.output,
            config.COMMUNITY_LEVEL: community_level,
            config.RATING: report.rating,
            "title": report.title,
            config.EXPLANATION: report.rating_explanation,
            config.SUMMARY: report.summary,
            config.FINDINGS: [
                {"explanation": f.explanation, "summary": f.summary}
                for f in report.findings
            ],
            config.FULL_CONTENT_JSON: report.model_dump_json(indent=4),
        }
    except Exception:
        logger.exception("Error processing community: %s", community_id)
        return None


async def _process_rows_async(
    input_df: pd.DataFrame,
    transform: Callable[[pd.Series], Awaitable[dict[str, Any] | None]],
    max_concurrent: int,
    progress_msg: str = "",
) -> list[dict[str, Any] | None]:
    semaphore = asyncio.Semaphore(max_concurrent)
    total = len(input_df)
    completed = 0

    async def execute_row(row: tuple[Hashable, pd.Series]) -> dict[str, Any] | None:
        nonlocal completed
        async with semaphore:
            result = await transform(row[1])
        completed += 1
        if progress_msg:
            logger.info("%s%s/%s", progress_msg, completed, total)
        return result

    tasks = [
        asyncio.create_task(execute_row(row)) for row in input_df.iterrows()
    ]
    return await asyncio.gather(*tasks)


async def summarize_communities(
    nodes: pd.DataFrame,
    communities: pd.DataFrame,
    local_contexts: pd.DataFrame,
    level_context_builder: Callable,
    extractor: AsyncCommunityReportExtractor,
    llm: AsyncLLMClient,
    model: str,
    max_input_length: int,
    max_concurrent: int,
    incremental: bool | None = None,
    existing_report_hashes: dict[int, str] | None = None,
) -> pd.DataFrame:
    reports: list[dict[str, Any]] = []
    community_hierarchy = (
        communities
        .explode("children")
        .rename({"children": "sub_community"}, axis=1)
        .loc[:, ["community", "level", "sub_community"]]
    ).dropna()

    # Determine incremental mode
    use_incremental = (
        INCREMENTAL_COMMUNITY_REPORTS if incremental is None else incremental
    )
    if use_incremental and existing_report_hashes is None:
        existing_report_hashes = {}
    skipped_count = 0
    skip_hash_cols = [config.CONTEXT_STRING]

    levels = get_levels(nodes)
    level_contexts = []
    for level in levels:
        level_context = await level_context_builder(
            pd.DataFrame(reports),
            community_hierarchy_df=community_hierarchy,
            local_context_df=local_contexts,
            level=level,
            llm=llm,
            model=model,
            max_context_tokens=max_input_length,
        )
        level_contexts.append(level_context)

    for i, level_context in enumerate(level_contexts):

        async def run_generate(record: pd.Series) -> dict[str, Any] | None:
            cid = int(record[config.COMMUNITY_ID])
            cur_hash = _community_content_hash(record, skip_hash_cols)
            # Incremental check: skip if community content hash unchanged
            if use_incremental and existing_report_hashes:
                prev_hash = existing_report_hashes.get(cid)
                if prev_hash and cur_hash == prev_hash:
                    return None
            report = await _generate_report(
                extractor,
                community_id=record[config.COMMUNITY_ID],
                community_level=record[config.COMMUNITY_LEVEL],
                community_context=record[config.CONTEXT_STRING],
            )
            if report:
                report["content_hash"] = cur_hash
            return report

        local_reports = await _process_rows_async(
            level_context,
            run_generate,
            max_concurrent=max_concurrent,
            progress_msg=f"level {levels[i]} summarize communities progress: ",
        )
        valid = [lr for lr in local_reports if lr is not None]
        skipped = len(local_reports) - len(valid)
        reports.extend(valid)
        skipped_count += skipped
        if skipped:
            logger.info(
                "Level %s: generated %d, skipped %d (incremental)",
                levels[i], len(valid), skipped,
            )

    if skipped_count:
        logger.info(
            "Incremental: skipped %d unchanged communities, generated %d",
            skipped_count, len(reports),
        )
    return pd.DataFrame(reports)


def finalize_community_reports(
    reports: pd.DataFrame,
    communities: pd.DataFrame,
) -> pd.DataFrame:
    community_reports = reports.merge(
        communities.loc[:, ["id","community", "parent", "children", "size", "period"]],
        on="community",
        how="left",
        copy=False,
    )
    community_reports["community"] = community_reports["community"].astype(int)
    community_reports["human_readable_id"] = community_reports["community"]
    #community_reports["id"] = community_reports.apply(
    #    lambda row: gen_sha512_hash(row, ["full_content"]), axis=1
    #)
    return community_reports.loc[:, config.COMMUNITY_REPORTS_FINAL_COLUMNS]


# --- Основной пайплайн ---


    
def _community_content_hash(row: pd.Series, columns: list[str]) -> str:
    """Stable hash of a community's member list for incremental skip detection."""
    content = "|".join(str(row.get(c, "")) for c in columns)
    return sha512(content.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


async def run_community_reports_pipeline_async(
    relationships: pd.DataFrame,
    entities: pd.DataFrame,
    communities: pd.DataFrame,
    model: str,
    prompt: str = COMMUNITY_REPORT_PROMPT,
    max_input_length: int = 16000,
    max_report_length: int = 2000,
    max_concurrent: int = 4,
    incremental: bool | None = None,
    existing_report_hashes: dict[int, str] | None = None,
) -> pd.DataFrame:
    """
    Асинхронный пайплайн генерации отчётов по сообществам.

    Последовательность действий:
    1. Подготовка узлов и рёбер
    2. Построение локального контекста для каждого сообщества
    3. Генерация отчётов по уровням иерархии (параллельно внутри уровня)
    4. Формирование финального DataFrame
    """
    async with AsyncLLMClient() as llm_client:
        extractor = AsyncCommunityReportExtractor(
            llm_client, model, prompt, max_report_length
        )

        # ═══ Этап 1: Подготовка данных ═══
        logger.info("Stage 1: Preparing nodes and edges...")
        nodes = explode_communities(communities, entities)
        nodes = _prep_nodes(nodes)
        edges = _prep_edges(relationships)
        logger.info(
            "Stage 1 complete: %s nodes, %s edges",
            len(nodes),
            len(edges),
        )

        # ═══ Этап 2: Построение локального контекста ═══
        logger.info("Stage 2: Building local context for communities...")
        # Выводим все колонки по 4 строки (примеров) для nodes и edges: временно увеличиваем ширину вывода и число столбцов
        with pd.option_context('display.max_columns', None, 'display.width', 0):
            logger.debug("Nodes sample:\n%s", nodes.head(4))
            logger.debug("Edges sample:\n%s", edges.head(4))
 
        local_contexts = await build_local_context(
            nodes, edges, llm_client, model, max_input_length
        )
        logger.info(
            "Stage 2 complete: %s community contexts built",
            len(local_contexts),
        )

        # ═══ Этап 3: Генерация отчётов по уровням ═══
        logger.info("Stage 3: Generating community reports...")
        community_reports = await summarize_communities(
            nodes,
            communities,
            local_contexts,
            build_level_context,
            extractor,
            llm_client,
            model,
            max_input_length,
            max_concurrent,
            incremental=incremental,
            existing_report_hashes=existing_report_hashes,
        )
        logger.info(
            "Stage 3 complete: %s reports generated",
            len(community_reports),
        )

        # ═══ Этап 4: Финализация ═══
        logger.info("Stage 4: Building final DataFrame...")
        result = finalize_community_reports(community_reports, communities)
        logger.info("Stage 4 complete: Pipeline finished successfully!")
        return result
