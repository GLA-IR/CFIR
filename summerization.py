
from dataclass import atomic_text
from transformers import pipeline
from tqdm.auto import tqdm
import json

if __name__ == '__main__':


    atomic = atomic_text([0, 10134744])

    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0, num_workers=0, batch_size=8)
    summarizer.tokenizer.model_max_length = 1000

    summaries = []
    save_freq = 20000
    for out in tqdm(summarizer(atomic, max_length=64, min_length=8, do_sample=False), total=len(atomic)):
        summaries.append(out[0]['summary_text'])

        if len(summaries) % save_freq == 0:
            print(len(summaries))
            with open(f'./atomic_summary/atomic_summaries_{len(summaries)+save_freq}.json', 'w') as f:
                json.dump(summaries[len(summaries)-save_freq:len(summaries)], f)
    
