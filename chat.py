import openai
openai.api_key="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"
while True:
    quetion = input('bot : ')
    answer = openai.Completion.create(
        model = "text-davinci-003",
        prompt = quetion,
        temperature=0.9,
        max_tokens=2000,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.6,
    )
    if 'choices' not in answer:
        print("I'm sorry, I don't have an answer for that")
    else:
        text = answer['choices'][0]['text']
        print('...' + text)
        print("="*50)    

