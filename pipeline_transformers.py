from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

#sentiment analysis default
cl=pipeline("sentiment-analysis")

res=cl("hey u look pretty")

print(res)


#using tokenizer and model
model_name="distilbert-base-uncased-finetuned-sst-2-english"
model=AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer=AutoTokenizer.from_pretrained(model_name)

cl=pipeline("sentiment-analysis",model=model,tokenizer=tokenizer)
res=cl("hey u look pretty")
print(res)


