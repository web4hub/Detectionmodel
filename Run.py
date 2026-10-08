from transformers import pipeline

pipe = pipeline("text-generation", model="ibm-granite/granite-4.1-30b")
messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe(messages)
