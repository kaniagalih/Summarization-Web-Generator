from pprint import pprint
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def summarizer(rawtext):
   
  # Load BERT tokenizer and model
  tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
  model = AutoModelForSeq2SeqLM.from_pretrained("patrickvonplaten/bert2bert_cnn_daily_mail")

  rawtext = rawtext

  # Split the text into chunks of maximum length 512
  max_len = 512
  chunks = [rawtext[i:i+max_len] for i in range(0, len(rawtext), max_len)]

  # Summarize each chunk separately
  summaries = []
  for chunk in chunks:
    inputs = tokenizer.encode(chunk, return_tensors="pt", max_length=max_len, truncation=True)
    summary_ids = model.generate(inputs, max_length=128, min_length=56, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    summaries.append(summary)

  # Combine the summaries into a single summary
  summary = " ".join(summaries)

  return summary, rawtext, len(rawtext.split(' ')), len(summary.split(' '))

