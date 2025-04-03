from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification


model_name="distilbert-base-uncased-finetuned-sst-2-english"
model=AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer=AutoTokenizer.from_pretrained(model_name)

cl=pipeline("sentiment-analysis",model=model,tokenizer=tokenizer)
res=cl("hey u look pretty")
print(res)

seq="Using transformer is good"
res=tokenizer(seq)
print(res)
tokens=tokenizer.tokenize(seq)
print(tokens)
ids=tokenizer.convert_tokens_to_ids(tokens)
print(ids)
decoded_string=tokenizer.decode(ids)
print(decoded_string)

