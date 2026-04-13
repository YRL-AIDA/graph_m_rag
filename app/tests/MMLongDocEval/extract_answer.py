from openai import OpenAI

client = OpenAI(
                base_url = 'http://localhost:11434/v1',     #for Ollama compatibility
                api_key='ollama',
                )


def extract_answer(question, output, prompt, model_name="gpt-4o"):
    #try:
        print(type(output))
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
                {
                "role": "assistant",
                "content": "\n\nQuestion:{}\nAnalysis:{}\n".format(question, output)
                }
            ],
            temperature=0.0,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        response = response.choices[0].message.content
    #except:
    #    response = "Failed"
    
        return response
    
    
if __name__=='__main__':
    with open("/home/user/RAG/MMLongBench_Doc/eval/prompt_for_answer_extraction.md") as f:
        prompt = f.read()
    res=extract_answer(question='According to the report, how do 5% of the Latinos see economic upward mobility for their children?',
                   output='Got it, let\'s look at the pie chart and the text. The pie chart is titled "Latinos see economic upward mobility for their children" and the segments are "Better off" (72%), "About the same" (16%), and "Less well off" (5%). So the 5% segment is "Less well off". The question is asking how 5% of Latinos see economic upward mobility for their children. So we need to find which category that is. The chart labels "Less well off" as 5%. Let me check the text: the pie chart has "Less well off" with 5%. So the answer should be "Less well off".\n</think>\n\nTo determine how 5% of Latinos see economic upward mobility for their children, we refer to the pie chart titled *Latinos see economic upward mobility for their children*. The chart shows the following segments:  \n- “Better off” (72%)  \n- “About the same” (16%)  \n- “Less well off” (5%)  \n\nThus, 5% of Latinos see **less well off** economic upward mobility for their children.',
                   prompt=prompt,
                   model_name='llama3.1')
    print(res)