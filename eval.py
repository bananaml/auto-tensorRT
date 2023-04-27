from human_eval.data import write_jsonl, read_problems
import requests


problems = read_problems()
print(len(problems))
def generate_one_completion(prompt,id):
    print("Running- ",id)
    model_payloads = {"prompt":"Complete the provided code.","input":prompt,"max_new_tokens":1024}
    rs = requests.post("http://185.32.161.60:43412/",json=model_payloads)
    output = rs.json()["output"]
    out = output.split("Response:")[1]
    print(out)
    return out
num_samples_per_task = 20

samples = [
    dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"],task_id))
    for task_id in problems
    for _ in range(num_samples_per_task)
]
write_jsonl("samples.jsonl", samples)