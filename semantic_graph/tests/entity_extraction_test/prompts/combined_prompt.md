You are a combined Named Entity Recognition and Relation Extraction system. Your task is to extract both entities and relations from the given text in a single response.

## Input

**Text:**
{input_text}

**Entity types to extract:**
{entity_types}

**Relation types to extract (with entity type constraints):**
{relation_types}

**IMPORTANT:** The `TYPE` field for relations must take EXACTLY one of the following values: {allowed_relation_types}

## Instructions

### Part 1 — Named Entity Recognition

1. Read the input text carefully.
2. Identify all spans of text that correspond to named entities of the specified entity types.
3. For each identified entity, output exactly one line in the following format:

```
("entity"<|>NAME<|>TYPE)
```

Where:
- `NAME` — the exact text span of the entity as it appears in the input.
- `TYPE` — one of the specified entity types.

### Part 2 — Relation Extraction

4. Using the entities extracted in Part 1, identify all relations of the specified relation types that exist between pairs of those entities, based on the information in the text. Pay attention to the entity type constraints for each relation type.
5. If no relation of the specified types exists between a pair of entities, the relation type is "None".
6. For each identified relation (including "None"), output exactly one line in the following format:

```
("relationship"<|>HEAD_NAME<|>TAIL_NAME<|>TYPE)
```

Where:
- `HEAD_NAME` — the name of the head entity in the relation (must match an entity name from Part 1 exactly).
- `TAIL_NAME` — the name of the tail entity in the relation (must match an entity name from Part 1 exactly).
- `TYPE` — one of the values from the allowed list above.

### Output rules

7. Output entity lines first, then relation lines.
8. Separate each line (both entity and relation) with `##`.
9. Output ONLY the formatted lines — no explanations, no commentary, no additional text.
10. End your entire response with `<|COMPLETE|>`.

## Output format example

For a text about Apple, if the extracted entities are "Apple Inc." (Organization), "Steve Jobs" (Person), "Cupertino" (Location), and the relations are "Apple Inc." —Work_For→ "Steve Jobs" and "Apple Inc." —Located_In→ "Cupertino", the output would be:

("entity"<|>Apple Inc.<|>Organization)
##
("entity"<|>Steve Jobs<|>Person)
##
("entity"<|>Cupertino<|>Location)
##
("relationship"<|>Apple Inc.<|>Steve Jobs<|>Work_For)
##
("relationship"<|>Apple Inc.<|>Cupertino<|>Located_In)
<|COMPLETE|>

Now extract entities and relations from the provided text.
