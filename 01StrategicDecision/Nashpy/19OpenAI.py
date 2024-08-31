import openai




def get_completion(prompt, model="gpt-3.5-turbo", temp=0.):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(model=model, messages=messages, temperature=temp)
    return response.choices[0].message['content'].strip()


returned_string1 = get_completion(prompt="Write 20 words about moon", temp=0)
print(f'Factual moon: {returned_string1}')

returned_string2 = get_completion(prompt="Write 20 words about moon", temp=.5)
print(f'General moon: {returned_string2}')

returned_string3 = get_completion(prompt="Write 20 words about moon", temp=1)
print(f'Wonderful moon: {returned_string3}')
