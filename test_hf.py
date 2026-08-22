import traceback
try:
    from transformers import pipeline
    print("Loading pipeline...")
    p = pipeline('text2text-generation', model='google/flan-t5-small')
    print('Success')
except Exception as e:
    print("Error:", e)
    print(traceback.format_exc())
