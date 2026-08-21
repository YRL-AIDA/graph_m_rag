"""Unified configuration for semantic_graph modules."""

import logging
import os
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

# --- LLM / model ---
#MODEL_NAME = 'Qwen/Qwen3-4B-Instruct-2507'
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-VL-32B-Thinking")
N4G_URL = os.getenv("N4G_URL", "http://0.0.0.0:7474")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")

# --- Qdrant ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333/")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OUTPUT_DIR = "data/processed"
DOCUMENT_ID_FIELD = "file_hash"

# --- Graph processing ---
MAX_CLUSTER_SIZE = 10
USE_LCC = True
CLUSTERIZATION_SEED = 256
ENTITY_TYPES = [
    'ORGANIZATION',
    'INSTITUTION',
    'PERSON',
    'GEO',
    'EVENT',
    'PRODUCT',
    'CONCEPT',
    'LAW',
    'NUMBER',
    'DATE',
    'GPE',
    'NORP',
    'ANATOMY',
    'SYMPTOM',
    'DISEASE',
    'PROCEDURE',
    'STRUCTURE',
    'COLOR',
]
MIN_ENTITY_CONFIDENCE = int(os.getenv("MIN_ENTITY_CONFIDENCE", "5"))
ADAPTIVE_BUDGET_ALLOCATION = os.getenv("ADAPTIVE_BUDGET_ALLOCATION", "true").lower() in ("1", "true", "yes")
DISAMBIGUATION_THRESHOLD = float(os.getenv("DISAMBIGUATION_THRESHOLD", "0.7"))
# Weighted clustering: amplify edge-weight differences so Leiden distinguishes
# strong from weak connections (LLM weights tend to cluster near 1.0).
LEIDEN_WEIGHT_EXPONENT = float(os.getenv("LEIDEN_WEIGHT_EXPONENT", "1.5"))
LEIDEN_MIN_WEIGHT = float(os.getenv("LEIDEN_MIN_WEIGHT", "0.1"))
# B6: Auto-resolution for Leiden — compute resolution from graph statistics
# (avg_degree / 10, clamped to [0.5, 2.0]) instead of using a fixed 1.0.
LEIDEN_AUTO_RESOLUTION = os.getenv("LEIDEN_AUTO_RESOLUTION", "true").lower() in ("1", "true", "yes")
# Incremental community reports: skip re-generation for communities whose
# entity memberships and edges haven't changed since the last run.
INCREMENTAL_COMMUNITY_REPORTS = os.getenv("INCREMENTAL_COMMUNITY_REPORTS", "true").lower() in ("1", "true", "yes")

# C8: Entity-aware query expansion — enrich question entities with top-k similar
# entities from Qdrant embeddings before running graph traversal.
ENTITY_QUERY_EXPANSION_ENABLED = os.getenv("ENTITY_QUERY_EXPANSION_ENABLED", "true").lower() in ("1", "true", "yes")
ENTITY_QUERY_EXPANSION_TOP_K = int(os.getenv("ENTITY_QUERY_EXPANSION_TOP_K", "3"))
ENTITY_QUERY_EXPANSION_SIM_THRESHOLD = float(os.getenv("ENTITY_QUERY_EXPANSION_SIM_THRESHOLD", "0.7"))

# D2: Weighted bridge connections — entity confidence multiplier for bridge edge
# weights (0-1).  Confidence/10 maps 1..10 → 0.1..1.0.  Multiplier scales that.
BRIDGE_WEIGHT_CONFIDENCE_MULTIPLIER = float(os.getenv("BRIDGE_WEIGHT_CONFIDENCE_MULTIPLIER", "1.0"))
BRIDGE_WEIGHT_PROXIMITY_DECAY = float(os.getenv("BRIDGE_WEIGHT_PROXIMITY_DECAY", "0.15"))

# C3: Hybrid search — combine dense vector similarity with keyword-based BM25-alike
# scoring.  When enabled, text-block search results are re-ranked using both the
# Qdrant cosine score and a keyword overlap score.  Requires no re-indexing.
HYBRID_SEARCH_ENABLED = os.getenv("HYBRID_SEARCH_ENABLED", "true").lower() in ("1", "true", "yes")
# alpha=1.0 → dense-only, alpha=0.0 → keyword-only, 0.7 is a good default.
HYBRID_SEARCH_ALPHA = float(os.getenv("HYBRID_SEARCH_ALPHA", "0.7"))

# --- API server ---
API_HOST = "0.0.0.0"
API_PORT = 9595

# --- Community report pipeline ---
INPUT_TEXT_KEY = "input_text"
MAX_LENGTH_KEY = "max_report_length"

# --- Document content ---
CONTENT_LABELS = ['text', 'table', 'image']

# --- Embedding service ---
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "http://192.168.19.127:10115/embedding")
EMBEDDING_TIMEOUT = int(os.getenv("EMBEDDING_TIMEOUT", "30"))
EMBEDDING_MAX_CONCURRENCY = int(os.getenv("EMBEDDING_MAX_CONCURRENCY", "8"))
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "0"))  # 0 = auto-detect from first response
EMBEDDING_RETRY_COUNT = int(os.getenv("EMBEDDING_RETRY_COUNT", "3"))
EMBEDDING_RETRY_BASE_DELAY = float(os.getenv("EMBEDDING_RETRY_BASE_DELAY", "1.0"))
ENTITY_EMBEDDINGS_COLLECTION = os.getenv("ENTITY_EMBEDDINGS_COLLECTION", "entity_embeddings")
ENTITY_EMBEDDINGS_BATCH_SIZE = int(os.getenv("ENTITY_EMBEDDINGS_BATCH_SIZE", "100"))
ENTITY_EMBEDDINGS_NAMESPACE = uuid.UUID(os.getenv("ENTITY_EMBEDDINGS_NAMESPACE", "a7f1b2c3-4d5e-6f78-9abc-def012345678"))

COMMUNITY_EMBEDDINGS_COLLECTION = os.getenv("COMMUNITY_EMBEDDINGS_COLLECTION", "community_embeddings")
COMMUNITY_EMBEDDINGS_BATCH_SIZE = int(os.getenv("COMMUNITY_EMBEDDINGS_BATCH_SIZE", "100"))
COMMUNITY_EMBEDDINGS_NAMESPACE = uuid.UUID(os.getenv("COMMUNITY_EMBEDDINGS_NAMESPACE", "b8e2c3d4-5e6f-7a89-bcde-f01234567890"))

# --- Disambiguation ---
DISAMBIGUATION_ENABLED: bool = os.getenv("DISAMBIGUATION_ENABLED", "true").lower() == "true"
DISAMBIGUATION_SIMILARITY_THRESHOLD: float = float(os.getenv("DISAMBIGUATION_SIMILARITY_THRESHOLD", "0.85"))


# --- DataFrame field names and column schemas ---

ID = "id"
SHORT_ID = "human_readable_id"
TITLE = "title"
DESCRIPTION = "description"

TYPE = "type"

# POST-PREP NODE TABLE SCHEMA
NODE_DEGREE = "degree"
NODE_FREQUENCY = "frequency"
CONFIDENCE = "confidence"
NODE_DETAILS = "node_details"

# POST-PREP EDGE TABLE SCHEMA
EDGE_SOURCE = "source"
EDGE_TARGET = "target"
EDGE_DEGREE = "combined_degree"
EDGE_DETAILS = "edge_details"
EDGE_WEIGHT = "weight"

# COMMUNITY HIERARCHY TABLE SCHEMA
SUB_COMMUNITY = "sub_community"

# COMMUNITY CONTEXT TABLE SCHEMA
ALL_CONTEXT = "all_context"
CONTEXT_STRING = "context_string"
CONTEXT_SIZE = "context_size"
CONTEXT_EXCEED_FLAG = "context_exceed_limit"

# COMMUNITY REPORT TABLE SCHEMA
COMMUNITY_ID = "community"
COMMUNITY_LEVEL = "level"
COMMUNITY_PARENT = "parent"
COMMUNITY_CHILDREN = "children"
SUMMARY = "summary"
FINDINGS = "findings"
RATING = "rank"
EXPLANATION = "rating_explanation"
FULL_CONTENT = "full_content"
FULL_CONTENT_JSON = "full_content_json"
CONTENT_HASH = "content_hash"

ENTITY_IDS = "entity_ids"
RELATIONSHIP_IDS = "relationship_ids"
TEXT_UNIT_IDS = "text_unit_ids"
COVARIATE_IDS = "covariate_ids"
DOCUMENT_ID = "document_id"
DEGREE = "degree"

PERIOD = "period"
SIZE = "size"

# text units
ENTITY_DEGREE = "entity_degree"
ALL_DETAILS = "all_details"
TEXT = "text"
N_TOKENS = "n_tokens"

CREATION_DATE = "creation_date"
RAW_DATA = "raw_data"

# the following lists define the final content and ordering of columns in the data model parquet outputs
ENTITIES_FINAL_COLUMNS = [
    ID,
    SHORT_ID,
    TITLE,
    TYPE,
    DESCRIPTION,
    TEXT_UNIT_IDS,
    NODE_FREQUENCY,
    NODE_DEGREE,
]

RELATIONSHIPS_FINAL_COLUMNS = [
    ID,
    SHORT_ID,
    EDGE_SOURCE,
    EDGE_TARGET,
    DESCRIPTION,
    EDGE_WEIGHT,
    EDGE_DEGREE,
    TEXT_UNIT_IDS,
]

COMMUNITIES_FINAL_COLUMNS = [
    ID,
    SHORT_ID,
    COMMUNITY_ID,
    COMMUNITY_LEVEL,
    COMMUNITY_PARENT,
    COMMUNITY_CHILDREN,
    TITLE,
    ENTITY_IDS,
    RELATIONSHIP_IDS,
    TEXT_UNIT_IDS,
    PERIOD,
    SIZE,
]

COMMUNITY_REPORTS_FINAL_COLUMNS = [
    ID,
    SHORT_ID,
    COMMUNITY_ID,
    COMMUNITY_LEVEL,
    COMMUNITY_PARENT,
    COMMUNITY_CHILDREN,
    TITLE,
    SUMMARY,
    FULL_CONTENT,
    RATING,
    EXPLANATION,
    FINDINGS,
    FULL_CONTENT_JSON,
    CONTENT_HASH,
    PERIOD,
    SIZE,
]

TEXT_UNITS_FINAL_COLUMNS = [
    ID,
    SHORT_ID,
    TEXT,
    N_TOKENS,
    DOCUMENT_ID,
    ENTITY_IDS,
    RELATIONSHIP_IDS,
]

DOCUMENTS_FINAL_COLUMNS = [
    ID,
    SHORT_ID,
    TITLE,
    TEXT,
    TEXT_UNIT_IDS,
    CREATION_DATE,
    RAW_DATA,
]

# --- Graph extraction delimiters and prompts ---
#TOKENIZER_URL = "http://192.168.19.127:9886/tokenize"
#LLM_URL = 'http://192.168.19.127:9886/v1'
#TOKENIZER_URL = "http://localhost:9886/tokenize"
#LLM_URL = 'http://localhost:9886/v1'
TOKENIZER_URL = os.getenv("TOKENIZER_URL", "http://192.168.19.127:8888/tokenize")
LLM_URL = os.getenv("LLM_URL", "http://192.168.19.127:8888/v1")

# --- Query extraction model (separate from main model) ---
QUERY_EXTRACTION_MODEL_NAME = os.getenv("QUERY_EXTRACTION_MODEL_NAME", MODEL_NAME)
QUERY_EXTRACTION_LLM_URL = os.getenv("QUERY_EXTRACTION_LLM_URL", LLM_URL)
QUERY_EXTRACTION_API_KEY = os.getenv("QUERY_EXTRACTION_API_KEY", LLM_API_KEY)
QUERY_EXTRACTION_TOKENIZER_URL = os.getenv("QUERY_EXTRACTION_TOKENIZER_URL", TOKENIZER_URL)

# --- Qdrant collections ---
DOCUMENTS_COLLECTION = os.getenv("DOCUMENTS_COLLECTION", "documents")

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

INLINE_GRAPH_EXTRACTION_PROMPT = """
-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.
 
-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"<|><entity_name><|><entity_type><|><entity_description><|><entity_confidence>)
where entity_confidence is an integer 1-10 indicating how confident you are that this is a distinct and meaningful entity (10 = very clearly defined, 1 = uncertain/vague).
 
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
("entity"<|>CENTRAL INSTITUTION<|>ORGANIZATION<|>The Central Institution is the Federal Reserve of Verdantis, which is setting interest rates on Monday and Thursday<|>9)
##
("entity"<|>MARTIN SMITH<|>PERSON<|>Martin Smith is the chair of the Central Institution<|>8)
##
("entity"<|>MARKET STRATEGY COMMITTEE<|>ORGANIZATION<|>The Central Institution committee makes key decisions about interest rates and the growth of Verdantis's money supply<|>7)
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
("entity"<|>TECHGLOBAL<|>ORGANIZATION<|>TechGlobal is a stock now listed on the Global Exchange which powers 85% of premium smartphones<|>8)
##
("entity"<|>VISION HOLDINGS<|>ORGANIZATION<|>Vision Holdings is a firm that previously owned TechGlobal<|>7)
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
("entity"<|>FIRUZABAD<|>GEO<|>Firuzabad held Aurelians as hostages<|>8)
##
("entity"<|>AURELIA<|>GEO<|>Country seeking to release hostages<|>7)
##
("entity"<|>QUINTARA<|>GEO<|>Country that negotiated a swap of money in exchange for hostages<|>7)
##
##
("entity"<|>TIRUZIA<|>GEO<|>Capital of Firuzabad where the Aurelians were being held<|>6)
##
("entity"<|>KROHAARA<|>GEO<|>Capital city in Quintara<|>6)
##
("entity"<|>CASHION<|>GEO<|>Capital city in Aurelia<|>6)
##
("entity"<|>SAMUEL NAMARA<|>PERSON<|>Aurelian who spent time in Tiruzia's Alhamia Prison<|>8)
##
("entity"<|>ALHAMIA PRISON<|>GEO<|>Prison in Tiruzia<|>7)
##
("entity"<|>DURKE BATAGLANI<|>PERSON<|>Aurelian journalist who was held hostage<|>8)
##
("entity"<|>MEGGIE TAZBAH<|>PERSON<|>Bratinas national and environmentalist who was held hostage<|>8)
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
Example 4:
Entity_types: ORGANIZATION,INSTITUTION,PERSON,GEO,EVENT,PRODUCT,CONCEPT,LAW,NUMBER,DATE,GPE,NORP,ANATOMY,SYMPTOM,DISEASE,PROCEDURE,STRUCTURE,COLOR
Text:
Costco Wholesale Corporation reported net sales of $226.95 billion for fiscal year 2024, an increase of 5.0% from $216.1 billion in 2023. The company, subject to Sarbanes-Oxley Act Section 404 compliance, operates 871 warehouses globally. On January 12, 2024, the Board declared a quarterly cash dividend of $1.16 per share. The Kirkland Signature brand accounted for 28% of total revenue. The SEC filed a comment letter on March 3, 2024 regarding goodwill impairment testing methodology under ASC 350. The company's effective income tax rate was 24.5% for fiscal 2024, compared to 23.1% in the prior year, due to changes in OECD Pillar Two global minimum tax rules effective from January 1, 2024. The workforce includes Americans, Canadians, Japanese, Mexicans, and British employees. The Democratic and Republican lawmakers debated the OECD tax treaty ratification in Congress. The Kirkland Signature appliance series launched in midnight black and arctic white color variants. In a separate medical study, a barium swallow examination revealed abnormal esophageal motility with tertiary contractions in the distal esophagus. The patient presented with dysphagia and retrosternal chest pain, and was diagnosed with diffuse esophageal spasm. The recommended procedure was endoscopic balloon dilation of the lower esophageal sphincter.
######################
Output:
("entity"<|>COSTCO WHOLESALE CORPORATION<|>ORGANIZATION<|>Costco is a wholesale retailer reporting $226.95 billion in net sales for fiscal 2024<|>10)
##
("entity"<|>SEC<|>INSTITUTION<|>The Securities and Exchange Commission is a federal regulatory agency that filed a comment letter to Costco<|>9)
##
("entity"<|>BOARD OF DIRECTORS<|>INSTITUTION<|>Costco's Board declared a quarterly cash dividend of $1.16 per share on January 12, 2024<|>8)
##
("entity"<|>KIRKLAND SIGNATURE<|>PRODUCT<|>Kirkland Signature is Costco's private-label brand accounting for 28% of total revenue<|>9)
##
("entity"<|>$226.95 BILLION<|>NUMBER<|>Net sales for fiscal year 2024<|>10)
##
("entity"<|>$216.1 BILLION<|>NUMBER<|>Net sales for fiscal year 2023<|>9)
##
("entity"<|>$1.16<|>NUMBER<|>Quarterly cash dividend per share declared on January 12, 2024<|>9)
##
("entity"<|>24.5%<|>NUMBER<|>Effective income tax rate for fiscal 2024<|>10)
##
("entity"<|>23.1%<|>NUMBER<|>Effective income tax rate for the prior year<|>9)
##
("entity"<|>28%<|>NUMBER<|>Percentage of total revenue from Kirkland Signature brand<|>9)
##
("entity"<|>SOX SECTION 404<|>LAW<|>Sarbanes-Oxley Act Section 404 requires management assessment of internal controls<|>9)
##
("entity"<|>ASC 350<|>LAW<|>Accounting Standards Codification 350 governs goodwill impairment testing methodology<|>9)
##
("entity"<|>OECD PILLAR TWO<|>LAW<|>OECD global minimum tax rules effective January 1, 2024 affecting multinational tax rates<|>9)
##
("entity"<|>GOODWILL IMPAIRMENT TESTING<|>CONCEPT<|>Methodology for testing whether goodwill value on balance sheet has declined<|>8)
##
("entity"<|>FISCAL YEAR 2024<|>DATE<|>Costco's fiscal year 2024 reporting period<|>9)
##
("entity"<|>JANUARY 12, 2024<|>DATE<|>Date when Board declared quarterly dividend<|>10)
##
("entity"<|>MARCH 3, 2024<|>DATE<|>Date when SEC filed comment letter to Costco<|>10)
##
("entity"<|>JANUARY 1, 2024<|>DATE<|>Effective date of OECD Pillar Two global minimum tax rules<|>10)
##
("entity"<|>UNITED STATES<|>GPE<|>Country where Costco is headquartered and primary market<|>10)
##
("entity"<|>CANADA<|>GPE<|>Country with significant Costco warehouse operations<|>9)
##
("entity"<|>JAPAN<|>GPE<|>Country with Costco international warehouse operations<|>9)
##
("entity"<|>MEXICO<|>GPE<|>Country with Costco international warehouse operations<|>8)
##
("entity"<|>UNITED KINGDOM<|>GPE<|>Country with Costco international warehouse operations<|>7)
##
("entity"<|>AMERICANS<|>NORP<|>Nationality group of US-based Costco employees<|>9)
##
("entity"<|>CANADIANS<|>NORP<|>Nationality group of Canadian Costco employees<|>8)
##
("entity"<|>JAPANESE<|>NORP<|>Nationality group of Japanese Costco employees<|>8)
##
("entity"<|>DEMOCRATS<|>NORP<|>Democratic party lawmakers debating OECD tax treaty ratification in Congress<|>8)
##
("entity"<|>REPUBLICANS<|>NORP<|>Republican party lawmakers debating OECD tax treaty ratification in Congress<|>8)
##
("entity"<|>DISTAL ESOPHAGUS<|>ANATOMY<|>Lower portion of the esophagus where tertiary contractions were observed<|>9)
##
("entity"<|>LOWER ESOPHAGEAL SPHINCTER<|>ANATOMY<|>Muscular ring at the gastroesophageal junction targeted for dilation<|>9)
##
("entity"<|>DYSPHAGIA<|>SYMPTOM<|>Difficulty swallowing reported by the patient<|>10)
##
("entity"<|>RETROSTERNAL CHEST PAIN<|>SYMPTOM<|>Pain behind the sternum experienced by the patient<|>9)
##
("entity"<|>DIFFUSE ESOPHAGEAL SPASM<|>DISEASE<|>Motility disorder characterized by tertiary contractions in the esophagus<|>10)
##
("entity"<|>BARIUM SWALLOW EXAMINATION<|>PROCEDURE<|>Diagnostic imaging test used to evaluate esophageal motility<|>9)
##
("entity"<|>ENDOSCOPIC BALLOON DILATION<|>PROCEDURE<|>Therapeutic procedure to widen the lower esophageal sphincter<|>10)
##
("entity"<|>TERTIARY CONTRACTIONS<|>STRUCTURE<|>Abnormal simultaneous esophageal contractions observed on barium swallow<|>9)
##
("entity"<|>MIDNIGHT BLACK<|>COLOR<|>Color variant of Kirkland Signature appliance series<|>8)
##
("entity"<|>ARCTIC WHITE<|>COLOR<|>Color variant of Kirkland Signature appliance series<|>8)
##
("relationship"<|>COSTCO WHOLESALE CORPORATION<|>$226.95 BILLION<|>Costco reported net sales of $226.95 billion for fiscal 2024<|>10)
##
("relationship"<|>COSTCO WHOLESALE CORPORATION<|>$216.1 BILLION<|>Costco reported net sales of $216.1 billion in 2023<|>9)
##
("relationship"<|>COSTCO WHOLESALE CORPORATION<|>BOARD OF DIRECTORS<|>Costco's Board declared a quarterly cash dividend<|>8)
##
("relationship"<|>BOARD OF DIRECTORS<|>$1.16<|>Board declared a quarterly dividend of $1.16 per share<|>9)
##
("relationship"<|>COSTCO WHOLESALE CORPORATION<|>KIRKLAND SIGNATURE<|>Costco owns and sells Kirkland Signature brand products<|>10)
##
("relationship"<|>KIRKLAND SIGNATURE<|>MIDNIGHT BLACK<|>Kirkland Signature appliance series available in midnight black<|>8)
##
("relationship"<|>KIRKLAND SIGNATURE<|>ARCTIC WHITE<|>Kirkland Signature appliance series available in arctic white<|>8)
##
("relationship"<|>KIRKLAND SIGNATURE<|>28%<|>Kirkland Signature accounted for 28% of Costco's total revenue<|>9)
##
("relationship"<|>COSTCO WHOLESALE CORPORATION<|>SEC<|>SEC filed a comment letter to Costco<|>8)
##
("relationship"<|>SEC<|>GOODWILL IMPAIRMENT TESTING<|>SEC inquired about goodwill impairment testing methodology<|>8)
##
("relationship"<|>GOODWILL IMPAIRMENT TESTING<|>ASC 350<|>ASC 350 governs goodwill impairment testing methodology<|>9)
##
("relationship"<|>COSTCO WHOLESALE CORPORATION<|>SOX SECTION 404<|>Costco is subject to Sarbanes-Oxley Act Section 404 compliance requirements<|>9)
##
("relationship"<|>COSTCO WHOLESALE CORPORATION<|>24.5%<|>Costco's effective income tax rate was 24.5% for fiscal 2024<|>10)
##
("relationship"<|>COSTCO WHOLESALE CORPORATION<|>23.1%<|>Costco's prior year effective tax rate was 23.1%<|>9)
##
("relationship"<|>24.5%<|>OECD PILLAR TWO<|>Change in tax rate from 23.1% to 24.5% driven by OECD Pillar Two rules<|>8)
##
("relationship"<|>BOARD OF DIRECTORS<|>JANUARY 12, 2024<|>Board declared dividend on January 12, 2024<|>10)
##
("relationship"<|>SEC<|>MARCH 3, 2024<|>SEC filed comment letter on March 3, 2024<|>10)
##
("relationship"<|>OECD PILLAR TWO<|>JANUARY 1, 2024<|>OECD Pillar Two rules became effective on January 1, 2024<|>9)
##
("relationship"<|>BARIUM SWALLOW EXAMINATION<|>DISTAL ESOPHAGUS<|>Barium swallow revealed abnormal motility in the distal esophagus<|>9)
##
("relationship"<|>DIFFUSE ESOPHAGEAL SPASM<|>TERTIARY CONTRACTIONS<|>Diffuse esophageal spasm is characterized by tertiary contractions<|>10)
##
("relationship"<|>DYSPHAGIA<|>DIFFUSE ESOPHAGEAL SPASM<|>Dysphagia is a symptom of diffuse esophageal spasm<|>9)
##
("relationship"<|>RETROSTERNAL CHEST PAIN<|>DIFFUSE ESOPHAGEAL SPASM<|>Retrosternal chest pain is a symptom of diffuse esophageal spasm<|>9)
##
("relationship"<|>ENDOSCOPIC BALLOON DILATION<|>LOWER ESOPHAGEAL SPHINCTER<|>Endoscopic balloon dilation targets the lower esophageal sphincter<|>10)
##
("relationship"<|>ENDOSCOPIC BALLOON DILATION<|>DIFFUSE ESOPHAGEAL SPASM<|>Endoscopic balloon dilation is a treatment for diffuse esophageal spasm<|>9)
##
("relationship"<|>TERTIARY CONTRACTIONS<|>DISTAL ESOPHAGUS<|>Tertiary contractions were observed in the distal esophagus<|>9)
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


# --- Prompt loading helper (defined after INLINE_GRAPH_EXTRACTION_PROMPT
#     so the fallback reference is resolved at definition time) ---
def _load_prompt() -> str:
    """Load GRAPH_EXTRACTION_PROMPT from file, fallback to inline string."""
    try:
        prompt_path = Path(__file__).parent / "prompts" / "graph-extraction.md"
        if prompt_path.is_file():
            return prompt_path.read_text(encoding="utf-8")
    except Exception:
        pass
    return INLINE_GRAPH_EXTRACTION_PROMPT


GRAPH_EXTRACTION_PROMPT = _load_prompt()


# --- Community report prompt ---
COMMUNITY_REPORT_PROMPT = """
You are an AI assistant that helps a human analyst to perform general information discovery. Information discovery is the process of identifying and assessing relevant information associated with certain entities (e.g., organizations and individuals) within a network.

# Goal
Write a comprehensive report of a community, given a list of entities that belong to the community as well as their relationships and optional associated claims. The report will be used to inform decision-makers about information associated with the community and their potential impact. The content of this report includes an overview of the community's key entities, their legal compliance, technical capabilities, reputation, and noteworthy claims.

# Report Structure

The report should include the following sections:

- TITLE: community's name that represents its key entities - title should be short but specific. When possible, include representative named entities in the title.
- SUMMARY: An executive summary of the community's overall structure, how its entities are related to each other, and significant information associated with its entities.
- IMPACT SEVERITY RATING: a float score between 0-10 that represents the severity of IMPACT posed by entities within the community.  IMPACT is the scored importance of a community.
- RATING EXPLANATION: Give a single sentence explanation of the IMPACT severity rating.
- DETAILED FINDINGS: A list of 5-10 key insights about the community. Each insight should have a short summary followed by multiple paragraphs of explanatory text grounded according to the grounding rules below. Be comprehensive.

Return output as a well-formed JSON-formatted string with the following format:
    {{
        "title": <report_title>,
        "summary": <executive_summary>,
        "rating": <impact_severity_rating>,
        "rating_explanation": <rating_explanation>,
        "findings": [
            {{
                "summary":<insight_1_summary>,
                "explanation": <insight_1_explanation>
            }},
            {{
                "summary":<insight_2_summary>,
                "explanation": <insight_2_explanation>
            }}
        ]
    }}

# Grounding Rules

Points supported by data should list their data references as follows:

"This is an example sentence supported by multiple data references [Data: <dataset name> (record ids); <dataset name> (record ids)]."

Do not list more than 5 record ids in a single reference. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

For example:
"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Reports (1), Entities (5, 7); Relationships (23); Claims (7, 2, 34, 64, 46, +more)]."

where id values represent the id (not the index) of the relevant data record.

Do not include information where the supporting evidence for it is not provided.

Limit the total report length to {max_report_length} words.

# Example Input
-----------
Text:

Entities

id,title,description
VERDANT OASIS PLAZA|LOCATION,VERDANT OASIS PLAZA,Verdant Oasis Plaza is the location of the Unity March
HARMONY ASSEMBLY|ORGANIZATION,HARMONY ASSEMBLY,Harmony Assembly is an organization that is holding a march at Verdant Oasis Plaza

Relationships

id,source,target,description
rel-37,VERDANT OASIS PLAZA|LOCATION,UNITY MARCH|EVENT,Verdant Oasis Plaza is the location of the Unity March
rel-38,VERDANT OASIS PLAZA|LOCATION,HARMONY ASSEMBLY|ORGANIZATION,Harmony Assembly is holding a march at Verdant Oasis Plaza
rel-39,VERDANT OASIS PLAZA|LOCATION,UNITY MARCH|EVENT,The Unity March is taking place at Verdant Oasis Plaza
rel-40,VERDANT OASIS PLAZA|LOCATION,TRIBUNE SPOTLIGHT|ORGANIZATION,Tribune Spotlight is reporting on the Unity march taking place at Verdant Oasis Plaza
rel-41,VERDANT OASIS PLAZA|LOCATION,BAILEY ASADI|PERSON,Bailey Asadi is speaking at Verdant Oasis Plaza about the march
rel-43,HARMONY ASSEMBLY|ORGANIZATION,UNITY MARCH|EVENT,Harmony Assembly is organizing the Unity March
Output:
{{
    "title": "Verdant Oasis Plaza and Unity March",
    "summary": "The community revolves around the Verdant Oasis Plaza, which is the location of the Unity March. The plaza has relationships with the Harmony Assembly, Unity March, and Tribune Spotlight, all of which are associated with the march event.",
    "rating": 5.0,
    "rating_explanation": "The impact severity rating is moderate due to the potential for unrest or conflict during the Unity March.",
    "findings": [
        {{
            "summary": "Verdant Oasis Plaza as the central location",
            "explanation": "Verdant Oasis Plaza is the central entity in this community, serving as the location for the Unity March. This plaza is the common link between all other entities, suggesting its significance in the community. The plaza's association with the march could potentially lead to issues such as public disorder or conflict, depending on the nature of the march and the reactions it provokes. [Data: Entities (VERDANT OASIS PLAZA|LOCATION), Relationships (rel-37, rel-38, rel-39, rel-40, rel-41,+more)]"
        }},
        {{
            "summary": "Harmony Assembly's role in the community",
            "explanation": "Harmony Assembly is another key entity in this community, being the organizer of the march at Verdant Oasis Plaza. The nature of Harmony Assembly and its march could be a potential source of threat, depending on their objectives and the reactions they provoke. The relationship between Harmony Assembly and the plaza is crucial in understanding the dynamics of this community. [Data: Entities (HARMONY ASSEMBLY|ORGANIZATION), Relationships (rel-38, rel-43)]"
        }},
        {{
            "summary": "Unity March as a significant event",
            "explanation": "The Unity March is a significant event taking place at Verdant Oasis Plaza. This event is a key factor in the community's dynamics and could be a potential source of threat, depending on the nature of the march and the reactions it provokes. The relationship between the march and the plaza is crucial in understanding the dynamics of this community. [Data: Relationships (rel-39)]"
        }},
        {{
            "summary": "Role of Tribune Spotlight",
            "explanation": "Tribune Spotlight is reporting on the Unity March taking place in Verdant Oasis Plaza. This suggests that the event has attracted media attention, which could amplify its impact on the community. The role of Tribune Spotlight could be significant in shaping public perception of the event and the entities involved. [Data: Relationships (rel-40)]"
        }}
    ]
}}


# Real Data

Use the following text for your answer. Do not make anything up in your answer.

Text:
{input_text}

The report should include the following sections:

- TITLE: community's name that represents its key entities - title should be short but specific. When possible, include representative named entities in the title.
- SUMMARY: An executive summary of the community's overall structure, how its entities are related to each other, and significant information associated with its entities.
- IMPACT SEVERITY RATING: a float score between 0-10 that represents the severity of IMPACT posed by entities within the community.  IMPACT is the scored importance of a community.
- RATING EXPLANATION: Give a single sentence explanation of the IMPACT severity rating.
- DETAILED FINDINGS: A list of 5-10 key insights about the community. Each insight should have a short summary followed by multiple paragraphs of explanatory text grounded according to the grounding rules below. Be comprehensive.
Return output as a well-formed JSON-formatted string with the following format:
    {{
        "title": <report_title>,
        "summary": <executive_summary>,
        "rating": <impact_severity_rating>,
        "rating_explanation": <rating_explanation>,
        "findings": [
            {{
                "summary":<insight_1_summary>,
                "explanation": <insight_1_explanation>
            }},
            {{
                "summary":<insight_2_summary>,
                "explanation": <insight_2_explanation>
            }}
        ]
    }}

# Grounding Rules

Points supported by data should list their data references as follows:

"This is an example sentence supported by multiple data references [Data: <dataset name> (record ids); <dataset name> (record ids)]."

Do not list more than 5 record ids in a single reference. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

For example:
"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Reports (1), Entities (5, 7); Relationships (23); Claims (7, 2, 34, 64, 46, +more)]."

where id values represent the id (not the index) of the relevant data record.

Do not include information where the supporting evidence for it is not provided.

Limit the total report length to {max_report_length} words.

Output:"""


# ──────────────────────────────────────────────
#  Configuration validation and diagnostics
# ──────────────────────────────────────────────

_SENSITIVE_KEYS = {"LLM_API_KEY", "QDRANT_API_KEY"}


def validate_config() -> list[str]:
    """Check critical configuration values and return a list of warnings."""
    warnings: list[str] = []

    if not LLM_API_KEY:
        warnings.append("LLM_API_KEY is empty – LLM authentication may fail.")
    if not QDRANT_URL:
        warnings.append("QDRANT_URL is empty.")
    if QDRANT_API_KEY is None:
        warnings.append("QDRANT_API_KEY is not set (None).")

    return warnings


def print_config_summary() -> None:
    """Log non-sensitive configuration values at INFO level."""
    for name in sorted(vars()):
        if name.startswith("_") or not name.isupper():
            continue
        value = vars()[name]
        if callable(value):
            continue
        if name in _SENSITIVE_KEYS:
            continue
        # Skip long string prompts – only log short scalar values
        if isinstance(value, str) and len(value) > 200:
            continue
        logger.info("Config %s = %s", name, repr(value))


