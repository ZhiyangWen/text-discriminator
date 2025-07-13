from datasets import load_dataset #fetch Xsum
import pandas as pd
import os
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig,pipeline


def main():
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,      
        llm_int8_threshold=6.0,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant_config, device_map = "auto", torch_dtype = "auto", trust_remote_code=True)#specify device to load the model onto appropriate hardware
    summarizer = pipeline("text-generation", model = model, tokenizer=tokenizer, max_new_tokens = 50, min_length = 10, do_sample = False, device_map="auto")
    
    full   = load_dataset("xsum", trust_remote_code=True)["train"]
    dataset = full.select(range(1000)) 
    human_sum = [ds["summary"] for ds in dataset] 

    ai_sum = []
    for ds in dataset:
        article = ds["document"]
        prompt  = (
        "Please summarize this article in one sentence:\n\n" 
        f"{article}\n\nSummary:")
        out = summarizer(prompt)[0]["generated_text"]
        summary = out[len(prompt):].strip()
        ai_sum.append(summary)

    df = pd.DataFrame({
        "text": ai_sum + human_sum,
        "label": [1] * len(ai_sum) + [0] * len(human_sum)
    })
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)#shuffle step
    os.makedirs("data/processed",exist_ok=True)
    path = "data/processed/dataset.csv"
    df.to_csv(path, index=False)
    print(f"Wrote {len(df)} rows to {path}")
if __name__ == "__main__":
    main()