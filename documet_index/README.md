

Запуск neo4j

```bash
docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/neo4j123 neo4j
```


Создание .env
```bash
cp .env.example .env
```

Простой тест (создает 2 файла, удаляет первый, выводит информацию о кол-во узлов)
```bash
python main.py
```