import requests

path_doc = "../mineru/test/document3.png"
path_doc = "../mineru/test/document2.pdf"

with open(path_doc, "rb") as f:
    response = requests.post(
        f"http://localhost:8001/process",
        files={"file": f},
        params={"backend": "vllm", "lang": "en"}
    )


print(('='*30)+"TASK" +('='*30) )    
print(response.json())
task_id = response.json()["task_id"]

result = requests.get(f"http://localhost:8001/status/{task_id}").json()
print(('='*30)+"RESULT" +('='*30) ) 
print(result)    

if 'markdown' in result['results']['download_links']:
    md_content = requests.get(
        result['results']['download_links']['content_list']
    ).text

    print(('='*30)+"Files" +('='*30) ) 
    print(md_content)