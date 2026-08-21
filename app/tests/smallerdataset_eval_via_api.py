"""
SmallerDataset evaluation pipeline using the main application API.

Based on mmlongdoceval_via_api.py but adapted for the SmallerDataset.
This script uses the /ask-document endpoint from the main application
with use_llm=True — the endpoint handles context assembly, semantic graph
enrichment, and LLM generation server-side.
"""

from typing import List, Union, Optional, Dict, Any
from pydantic import BaseModel, ConfigDict
import requests
import json
import time
import os

from MMLongDocEval.extract_answer import extract_answer
from MMLongDocEval.eval_score import eval_score, eval_acc_and_f1, show_results


def read_json(filename):
    with open(filename) as json_data:
        data = json.load(json_data)
        json_data.close()
    return data


def read_jsonl(filename):
    with open(filename) as f:
        data = [json.loads(line) for line in f]
    f.close()
    return data


class Message(BaseModel):
    model_config = ConfigDict(extra='ignore')

    type: str
    text: Union[str, None] = None
    image: Union[str, None] = None
    image_url: Union[str, None] = None


class EmbedRequest(BaseModel):
    messages: List[Message]


class MessageEmbedding(BaseModel):
    message_id: int
    embedding: List[float]


class EmbedResponse(BaseModel):
    messages: List[MessageEmbedding]


class EmbedErrorResponse(BaseModel):
    detail: str


def extract_answer_qwen_api(question, output, prompt):
    """Extract answer using custom Qwen API."""
    import utils.qwen_qa_utils as custom_qwen

    tt = custom_qwen.ModelMessageDict()
    tt.add_text_content(prompt)
    answer = f"Question: {question}\nAnalysis:{output}"
    tt.add_text_content(answer)
    result = custom_qwen.send_messasge(messages=[tt], base_url='http://192.168.19.127:8888/v1')
    return result[1][0]


# ========== Client for Main Application API ==========
class MainAppClient:
    """
    Client for the main application API.
    Uses the /ask-document endpoint for question answering with RAG.
    """

    def __init__(self, base_url: str, timeout: int = 300):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout

    def ask_document(
        self,
        file_hash: str,
        question: str,
        limit: int = 30,
        collection_name: Optional[str] = None,
        use_llm: bool = False,
        use_reranker: bool = False,
        use_semantic_graph: bool = False,
        use_structured_graph: bool = False,
        use_iterative_search: bool = False
    ) -> Dict[str, Any]:
        request_payload = {
            "file_hash": file_hash,
            "question": question,
            "limit": limit,
            "use_llm": use_llm,
            "use_reranker": use_reranker,
            "use_semantic_graph": use_semantic_graph,
            "use_structured_graph": use_structured_graph,
            "use_iterative_search": use_iterative_search
        }

        if collection_name:
            request_payload["collection_name"] = collection_name

        url = f"{self.base_url}/ask-document"
        response = requests.post(url, json=request_payload, timeout=self.timeout)
        response.raise_for_status()

        return response.json()


# ========== Main pipeline ==========
if __name__ == "__main__":

    # Load file hash comparison dictionary (same as MMLongDocEval)
    HNKdict = read_json(f"{os.getcwd()}/file_hash_comparison.json")

    # Load SmallerDataset
    dataset_path = "/home/sunveil/Documents/projects/laba/graph-m-rag/data/SmallerDataset/samples.json"
    output_filename = f"{os.getcwd()}/SmallerDataset_answers.json"

    if os.path.exists(output_filename):
        dataset = read_json(output_filename)
    else:
        dataset = read_json(dataset_path)

    reserve_check = True

    # Load extraction prompt
    with open(f"{os.getcwd()}/MMLongDocEval/prompt_for_answer_extraction.md", 'r') as f:
        extract_prompt = f.read()

    # Initialize client for main application API
    client = MainAppClient(base_url="http://0.0.0.0:9191")

    limit = 30

    for k, caser in enumerate(dataset):
        start_time = time.time()
        model_answer_time = 0
        model_extract_time = 0

        # Skip already processed cases
        if k < -1:
            continue

        caser['status'] = 'unknown'

        if 'response' in caser.keys():
            print(f"\n>>>> Already processed! {k}/{len(dataset)}\n\n")
            continue

        print(f"\n>>> question {k}/{len(dataset)}\n\n")

        query = caser['question']

        # Check if document exists in hash dictionary
        if caser['doc_id'] not in HNKdict.keys():
            print(f"doc {caser['doc_id']} not in base")
            caser['response'] = 'failed to find the document'
            continue
        else:
            fileHash = HNKdict[caser['doc_id']]
            print(f"File hash: {fileHash}")

        # Initialize result variables
        model_answer = None
        retrieves = None
        score = 0
        extracted_res = None

        try:
            # Call the main application API with use_llm=True.
            # The endpoint handles context assembly, semantic graph enrichment,
            # and LLM generation server-side.
            print(f"\n\nQuery: {query}\n\n")

            result = client.ask_document(
                file_hash=fileHash,
                question=query,
                limit=limit,
                use_llm=True,
                use_reranker=True,
                use_semantic_graph=True,
                use_structured_graph=True,
                use_iterative_search=True
            )

            if result.get('status') == 'success':
                answers = result.get('answers', [])

                # Print retrieved chunks
                for elem in answers:
                    print(f"{elem.get('text', '')} <-> {elem.get('score', 0)}\n\n")

                print(f"Total retrieves: {len(answers)}")

                if len(answers) > 0:
                    # Prepare retrieves metadata
                    retrieves = []
                    for elem in answers:
                        retrieves.append({
                            'qdrant_id': elem.get('element_index', 0),
                            'type': elem.get('element_type', 'unknown'),
                            'file_hash': fileHash,
                            'content': elem.get('text', ''),
                            'page_idx': elem.get('page_idx', 0),
                            'score': elem.get('score', 0)
                        })

                    # LLM answer is generated server-side by the endpoint
                    model_answer = result.get('llm_answer', '')

                    if not model_answer:
                        print('No LLM answer returned from endpoint')
                        caser['status'] = 'no llm answer'
                        continue

                    model_answer_time = time.time() - start_time
                    start_time = time.time()

                    print(f">>> Full model answer:\n{model_answer}\n\n")

                    # Extract answer using extraction API
                    extracted_res = extract_answer_qwen_api(query, model_answer, extract_prompt)
                    model_extract_time = time.time() - start_time

                    print(f">>> Extracted answer:\n{extracted_res}\n\n>>> Correct answer: {caser['answer']}\n\n\n")

                    # Store results
                    caser['response'] = model_answer
                    caser['extracted_res'] = extracted_res
                    caser['score'] = score
                    caser['retrieves'] = retrieves
                    caser['status'] = 'completed'
                else:
                    print('No retrieves found')
                    caser['status'] = 'no retrieves'
            else:
                print(f"API returned error: {result.get('message', 'Unknown error')}")
                caser['status'] = f"api_error: {result.get('message', 'Unknown error')}"

        except Exception as e:
            print(f"Error: {e}")
            caser['status'] = f'error: {str(e)}'

        print(f"\n answering time: {model_answer_time}; answer extraction time: {model_extract_time} \n")

        # Log processing times
        with open('proceeding_time_smaller.txt', 'a') as timesF:
            timesF.write(f"{model_answer_time};{model_extract_time}\n")

        # Save intermediate results
        if reserve_check:
            with open(output_filename, 'w') as f:
                json.dump(dataset, f)

    # Save final results
    with open(output_filename, 'w') as f:
        json.dump(dataset, f)
