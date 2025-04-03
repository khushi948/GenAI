#zero shot classification
from transformers import pipeline
cl=pipeline("zero-shot-classification")
res=cl("This is a course about cooking ",candidate_labels=['education','cook','politics','finance'])
print(res)