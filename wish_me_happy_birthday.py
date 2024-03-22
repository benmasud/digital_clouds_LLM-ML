
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import gradio as gr

model_name = "sberbank-ai/rugpt3small_based_on_gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

model.eval()

def generate_birthday_greeting(name):
    input_text = f"С днём рождения, {name}! Пусть этот день будет наполнен радостью и счастьем. "
    input_ids = tokenizer.encode(input_text, return_tensors='pt')


    with torch.no_grad():
        output = model.generate(input_ids, max_length=200, num_return_sequences=1,
                                temperature=0.9, do_sample=True, pad_token_id=tokenizer.eos_token_id)


    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    lines = generated_text.split('\n')
    relevant_part = ""
    for line in lines:
        if line.startswith("С днём рождения") or line.startswith("Пусть этот день"):
            relevant_part += line + "\n"
        if len(relevant_part.split('\n')) >= 5:  # can adjust lines :)
            break

    return relevant_part.strip()


example_names = ["Елена", "Андрей", "Сергей", "Наташа", "Ольга"]


iface = gr.Interface(fn=generate_birthday_greeting, inputs="text", outputs="text",
                     title="Birthday Greeting Generator", description="Введите имя, кого вы хотите пожелать на день рождения)",
                     examples=[[name] for name in example_names])


iface.launch()

"""### Description | Описание

Итак, я использовал `PyTorch` для функций глубокого обучения (мне просто было удобнее использовать графический процессор - GPU )
и библиотеку преобразователей HuggingFace для доступа к предварительно обученным языковым моделям, 
в частности (https://huggingface.co/ai-forever/rugpt3small_based_on_gpt2) GPT-2 (sberbank-ai/rugpt3small_based_on_gpt2). 
Загрузив модель GPT-2 и токенизатор, я смог генерировать поздравления с днем рождения на основе введенных пользователем имен. 
Модель оптимизирована для генерации поздравлений с днем рождения, что делает ее подходящей для этой задачи. 
(https://www.gradio.app/)Gradio был использован для создания интуитивно понятного интерфейса, позволяющего пользователям легко взаимодействовать с моделью.


"""

