#text generation
from transformers import pipeline
gen=pipeline("text-generation",model="distilgpt2")
res=gen("The course is about the technology used in real world applications. ",max_length=100,num_return_sequences=2)
print(res)
