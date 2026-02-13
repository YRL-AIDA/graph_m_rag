import requests
from config.settings import settings

path_doc = "test/document3.png"
path_doc = "test/document2.pdf"

with open(path_doc, "rb") as f:
    response = requests.post(
        f"{settings.mineru.host}:{settings.mineru.port}/process",
        files={"file": f},
        params={"backend": "pipeline", "lang": "ru"}
    )


print(('='*30)+"TASK" +('='*30) )    
print(response.json())
task_id = response.json()["task_id"]

result = requests.get(f"http://localhost:8000/status/{task_id}").json()
print(('='*30)+"RESULT" +('='*30) ) 
print(result)    

if 'markdown' in result['results']['download_links']:
    md_content = requests.get(
        result['results']['download_links']['content_list']
    ).text

    print(('='*30)+"Files" +('='*30) ) 
    print(md_content)