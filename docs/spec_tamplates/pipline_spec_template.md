# Pipeline: {Название пайплайна}

## Purpose
- Что делает пайплайн, какой результат производит

## Stages
| Stage | Input | Processing | Output | Concurrency |
|-------|-------|------------|--------|-------------|
| 1. ... | ... | ... | ... | sequential/parallel |

## Data Flow Diagram
- Визуализация этапов (mermaid или текстом)

## LLM Interactions
- Ссылки на файлы промптов: `prompts/{prompt-name}.md`
- Требования к промптам (не точный текст)

## Performance Constraints
- Ограничения по токенам, таймауты, параллелизм

## Error Recovery
- Что происходит при сбое на каждом этапе