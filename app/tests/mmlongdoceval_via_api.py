"""
MMLongDocEval pipeline using the main application API.
This script uses the /ask-document endpoint from the main application
instead of directly accessing Qdrant, MinIO, and LLM services.
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
    """Extract answer using custom Qwen API (kept for compatibility)"""
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

    def __init__(self, base_url: str, timeout: int = 120):
        """
        Args:
            base_url: The base URL of the main application API (e.g. http://localhost:8000).
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout

    def ask_document(
        self,
        file_hash: str,
        question: str,
        limit: int = 30,
        collection_name: Optional[str] = None,
        use_llm: bool = False,
        use_reranker: bool = False
    ) -> Dict[str, Any]:
        """
        Ask a question about a document using the main application's RAG pipeline.

        Args:
            file_hash: Hash of the document to query.
            question: The question to ask.
            limit: Maximum number of retrieved chunks.
            collection_name: Optional collection name in Qdrant.
            use_llm: Whether to generate answer using LLM.
            use_reranker: Whether to use reranker for re-ranking results.

        Returns:
            Dictionary with status, answers, retrieves, and optional llm_answer.

        Raises:
            requests.RequestException: If the HTTP request fails.
        """
        # Build the request payload
        request_payload = {
            "file_hash": file_hash,
            "question": question,
            "limit": limit,
            "use_llm": use_llm,
            "use_reranker": use_reranker
        }

        if collection_name:
            request_payload["collection_name"] = collection_name

        # Send POST request to /ask-document endpoint
        url = f"{self.base_url}/ask-document"
        response = requests.post(url, json=request_payload, timeout=self.timeout)
        response.raise_for_status()

        # Parse and return the response
        return response.json()


# ========== Main pipeline ==========
if __name__ == "__main__":

    retrieve_prompt = "Answer the question based on given context. Elements of context are presented after the question. Each piece of context starts with "

    # Load file hash comparison dictionary
    HNKdict = read_json(f"{os.getcwd()}/file_hash_comparison.json")

    # Load or initialize dataset
    filename = f"{os.getcwd()}/MMLongDoc_answers.json"
    if os.path.exists(filename):
        dataset = read_json(filename)
    else:
        dataset = read_json('/home/sunveil/Documents/projects/laba/graph-m-rag/data/MMLongBench-Doc/data/samples.json')

    reserve_check = True

    # Load extraction prompt
    with open(f"{os.getcwd()}/MMLongDocEval/prompt_for_answer_extraction.md", 'r') as f:
        extract_prompt = f.read()

    # Initialize client for main application API
    # Point to your actual main application URL
    client = MainAppClient(base_url="http://0.0.0.0:9191")

    limit = 30

    for k, caser in enumerate(dataset):
        start_time = time.time()
        model_answer_time = 0
        model_extract_time = 0

        # Skip already processed or filtered cases
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
        embeds = None
        search_result = None
        model_answer = None
        retrieves = None
        score = 0
        extracted_res = None

        try:
            # Call the main application API
            print(f"\n\nQuery: {query}\n\n")

            # Use the API to get answers and retrievals
            # Set use_llm=False to get only retrieved chunks without LLM generation
            # We'll handle LLM separately if needed
            result = client.ask_document(
                file_hash=fileHash,
                question=query,
                limit=limit,
                use_llm=False,  # We'll handle LLM separately
                use_reranker=False
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

                    # Prepare context for LLM
                    # Build messages for LLM from retrieved chunks
                    import utils.qwen_qa_utils as custom_qwen

                    tt = custom_qwen.ModelMessageDict()
                    tt.add_text_content(f"Question: {query}")

                    for elem in answers:
                        element_type = elem.get('element_type', '')

                        if element_type != 'image':
                            tt.add_text_content(elem.get('text', ''))
                        elif element_type == 'image' and elem.get('image_base64'):
                            # Add image content if available
                            tt.add_img_content_base64(elem.get('image_base64'))

                    # Call LLM for answer generation
                    try:
                        llm_result = custom_qwen.send_messasge(
                            messages=[tt],
                            base_url='http://192.168.19.127:8888/v1'
                        )
                    except Exception as e:
                        caser['status'] = 'failed to connect to LLM'
                        print(f"Failed to connect to model: {e}")
                        continue

                    model_answer_time = time.time() - start_time
                    start_time = time.time()

                    if str(llm_result[1]) == 'None':
                        break

                    model_answer = llm_result[1][0]

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
        with open('proceeding_time.txt', 'a') as timesF:
            timesF.write(f"{model_answer_time};{model_extract_time}\n")

        # Save intermediate results
        if reserve_check:
            with open('MMLongDoc_answers.json', 'w') as f:
                json.dump(dataset, f)

    # Save final results
    with open('MMLongDoc_answers.json', 'w') as f:
        json.dump(dataset, f)