# Preserving Page Number Information During Indexing

**Date:** 2026-08-11  
**Status:** Proposal

---

## 1. Current State Analysis

### 1.1 What Already Works

| Layer | Where `page_idx` exists | Status |
|-------|------------------------|--------|
| MinerU raw output | `content_list[].page_idx` | ✅ Present |
| Qdrant payload | `payload.page_idx` in `compute_embeddings_for_elements()` | ✅ Stored and indexed at [`app/src/qdrant_client_api.py:273`](app/src/qdrant_client_api.py:273) |
| Semantic graph DataFrame | `df["page"]` in [`build_chunks_dataframe()`](semantic_graph/Qdrant_extractor/dataframe_builder.py:42) | ✅ Extracted from Qdrant |

### 1.2 What's Missing

| Layer | Gap |
|-------|-----|
| **`Region` dataclass** | No `page_idx` field — [`documet_index/dtype/region.py:32-40`](documet_index/dtype/region.py:32-40) |
| **`Region.to_dict()`** | Doesn't serialize `page_idx` — [`documet_index/dtype/region.py:46-63`](documet_index/dtype/region.py:46-63) |
| **`create_graph_from_mineru_result()`** | Reads `page_idx = element.get("page_idx", 0)` at line 166 of [`document.py`](documet_index/dtype/document.py:166) but **never passes it** to any `Region(...)` constructor |
| **Neo4j Region nodes** | Cypher `CREATE` in [`Manager.add_document()`](documet_index/manager.py:188-210) has no `page_idx` property |
| **Neo4j Section nodes** | No `page_start`/`page_end` metadata |
| **Semantic graph entities** | No `page_idx` provenance linking back to source pages |

---

## 2. Proposed Changes

### 2.1 Add `page_idx` to `Region` class

**File:** [`documet_index/dtype/region.py`](documet_index/dtype/region.py)

```python
class Region:
    def __init__(self, text: str, image: str, bbox: BBox, style: Style,
                 order: int, label: str, element_data: str, page_idx: int = 0):
        self.text = text
        self.image = image
        self.bbox = bbox
        self.style = style
        self.order = order
        self.label = label
        self.element_data = element_data
        self.page_idx = page_idx          # NEW

    def to_dict(self):
        return {
            "label": self.label,
            "text": self.text,
            "image": self.image,
            "bbox": { ... } if self.bbox else {},
            "style": { ... } if self.style else {},
            "order": self.order,
            "element_data": self.element_data,
            "page_idx": self.page_idx,    # NEW
        }
```

### 2.2 Pass `page_idx` in all `Region(...)` constructor calls

**File:** [`documet_index/dtype/document.py`](documet_index/dtype/document.py), function `create_graph_from_mineru_result()`

Every `Region(...)` call needs `page_idx=page_idx` added. The `page_idx` variable is already extracted at line 166:

```python
page_idx = element.get("page_idx", 0)
```

All ~12 Region constructor calls (text, title, image, image_caption, image_footnote, table, table_caption, table_footnote, equation, generic) need the extra parameter. Example for title:

```python
# Before
regions.append(Region(
    text=f"Title: {text}",
    image="",
    bbox=BBox(*bbox) if len(bbox) == 4 else BBox(0, 0, 0, 0),
    style=Style(-1),
    order=element_index,
    label="title",
    element_data=text
))
# After
regions.append(Region(
    text=f"Title: {text}",
    image="",
    bbox=BBox(*bbox) if len(bbox) == 4 else BBox(0, 0, 0, 0),
    style=Style(-1),
    order=element_index,
    label="title",
    element_data=text,
    page_idx=page_idx,       # NEW
))
```

### 2.3 Store `page_idx` on Neo4j Region nodes

**File:** [`documet_index/manager.py`](documet_index/manager.py), method `add_document()`

Add `page_idx` to the Cypher `CREATE` statement. Currently the Region creation at lines 208-210 is:

```cypher
CREATE (reg{id}:Region:{label} {{
    region_id: '{region_id}',
    text: '{text_escaped}',
    image: '{image_escaped}',
    bbox: '{bbox_json}',
    style: '{style_json}',
    order: {order},
    element_data: '{element_data_escaped}'
}})
```

Add `page_idx`:

```cypher
CREATE (reg{id}:Region:{label} {{
    region_id: '{region_id}',
    text: '{text_escaped}',
    image: '{image_escaped}',
    bbox: '{bbox_json}',
    style: '{style_json}',
    order: {order},
    element_data: '{element_data_escaped}',
    page_idx: {reg['page_idx']}       -- NEW
}})
```

The `page_idx` value is already available in the `reg` dict (because `to_dict()` now includes it).

### 2.4 Add page metadata to Section nodes (optional enhancement)

**File:** [`documet_index/manager.py`](documet_index/manager.py), method `_build_sections()`

Add `start_page` and `end_page` to each section dict. This requires looking up the `page_idx` of the first and last region in the section:

```python
sections.append({
    "section_id": section_id,
    "title": clean_title,
    "regions": [r["id"] for r in section_regions],
    "start_order": section_regions[0]["order"],
    "end_order": section_regions[-1]["order"],
    "start_page": section_regions[0].get("page_idx", 0),   # NEW
    "end_page": section_regions[-1].get("page_idx", 0),     # NEW
})
```

And in the Cypher `CREATE` for Section nodes:

```cypher
CREATE (sec_{sec_id_norm}:Section {{
    section_id: '{sec_id_norm}',
    title: '{title_escaped}',
    start_order: {sec['start_order']},
    end_order: {sec['end_order']},
    start_page: {sec['start_page']},    -- NEW
    end_page: {sec['end_page']}         -- NEW
}})
```

### 2.5 Propagate `page_idx` to semantic graph entities

**File:** [`semantic_graph/manager.py`](semantic_graph/manager.py) — entity/relationship creation methods

When entities and relationships are extracted from chunks (which come from `build_chunks_dataframe` that already has `page`), add `page_idx` to the entity/relationship nodes in Neo4j.

Currently `build_chunks_dataframe()` returns a DataFrame with a `page` column (line 42). The `run_extraction_pipeline_async` processes these text units but doesn't thread `page` through to entities. Options:

**Option A (simple):** Add `page_idx` as a property on Entity nodes. Each entity gets `page_idx` from the chunk it was extracted from. If an entity appears in multiple chunks, store the list of pages.

**Option B (relational):** Create `MENTIONED_ON` relations: `(Entity)-[:MENTIONED_ON {page: N}]->(Chunk)`. This is more expressive but more complex.

**Recommendation:** Start with Option A — a simple integer/list property on Entity nodes, then add Option B later if needed.

---

## 3. Migration and Reindexing

### 3.1 Impact

- **Structural graph (Neo4j):** Existing Region nodes will lack `page_idx`. Newly indexed documents will have it. No data migration required unless you want to backfill.
- **Qdrant:** Already has `page_idx` in payload — no changes needed.
- **Semantic graph (Neo4j):** New entities will have `page_idx`; existing entities won't.

### 3.2 Rollout Plan

1. Apply changes to `Region` class, `to_dict()`, and `create_graph_from_mineru_result()` (Sections 2.1-2.2).
2. Apply changes to `Manager.add_document()` Cypher (Section 2.3).
3. (Optionally) Apply Section node changes (Section 2.4).
4. (Optionally) Apply semantic entity changes (Section 2.5).
5. Reindex documents. Old documents remain without `page_idx` — queries should use `COALESCE(n.page_idx, 0)` or treat missing `page_idx` as `null`.
6. Add a Neo4j constraint or index on `page_idx` if filtering by page becomes frequent:

```cypher
CREATE INDEX region_page_idx IF NOT EXISTS FOR (r:Region) ON (r.page_idx);
```

---

## 4. Query Usage Examples

Once implemented, page numbers become available for:

```cypher
// Find all regions on a specific page
MATCH (r:Region) WHERE r.page_idx = 5 RETURN r.region_id, r.text, r.label

// Get a page range for a section
MATCH (s:Section {section_id: 'hash|section_2'})
RETURN s.title, s.start_page, s.end_page

// Retrieval API: add page_idx to returned context
// In api.py search results, already partially done — see
// app/src/api.py:1620 "page_idx": original_element.get("page_idx", 0)
```

---

## 5. Summary of Files to Modify

| File | Changes |
|------|---------|
| [`documet_index/dtype/region.py`](documet_index/dtype/region.py) | Add `page_idx` param to `Region.__init__` and `to_dict()` |
| [`documet_index/dtype/document.py`](documet_index/dtype/document.py) | Pass `page_idx` to all `Region(...)` calls in `create_graph_from_mineru_result()` |
| [`documet_index/manager.py`](documet_index/manager.py) | Add `page_idx` to Region Cypher CREATE; optionally add `start_page`/`end_page` to Sections |
| [`semantic_graph/manager.py`](semantic_graph/manager.py) | (Optional) Add `page_idx` to entity/relationship nodes |
