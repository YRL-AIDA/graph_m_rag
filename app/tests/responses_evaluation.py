from MMLongDocEval.eval_score import eval_score, eval_acc_and_f1, show_results
import re
import json




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


if __name__=='__main__':
    dataset=read_json('MMLongDoc_answers.json')
    nonecount=0
    for sample in dataset:
        if(str(sample['response'])=='None'):
            print('AAAAAAAAAAAAAAAAAAAAAA')
        if('extracted_res' not in sample.keys()):
            nonecount+=1
            continue
        extracted_res=sample['extracted_res']
        if(str(extracted_res)=='None'):
            nonecount+=1
        else:

            try:
                pred_ans = extracted_res.split("Answer format:")[0].split("Extracted answer:")[1].strip()
                score = eval_score(sample["answer"], pred_ans, sample["answer_format"])
            except:
                pred_ans = "Failed to extract"
                score=0.0
                
                
            sample["pred"] = pred_ans
            sample["score"] = score     
    
    print(nonecount)
    show_results(dataset, show_path=re.sub("\.json$", ".txt", 'MMLongDoc_answers_evaluated.json'))
    with open('MMLongDoc_answers_evaluated.json', 'w') as f:
        json.dump(dataset, f)