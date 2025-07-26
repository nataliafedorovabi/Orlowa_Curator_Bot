from transformers import pipeline
import pandas as pd

# Путь к файлу на Google Диске
file_path = '/content/drive/MyDrive/chat_girls.xlsx'

# Загружаем zero-shot классификатор (универсальный, работает и с русским)
classifier = pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli")

labels = ["питание", "личное", "служебное"]

df = pd.read_excel(file_path, sheet_name="Sheet1")

def get_category(text):
    if pd.isna(text) or not str(text).strip():
        return "пусто"
    result = classifier(str(text), labels)
    return result["labels"][0]  # самая вероятная категория

# Категоризация question
df.insert(df.columns.get_loc('question') + 1, 'категория question', df['question'].apply(get_category))
# Категоризация answer
df.insert(df.columns.get_loc('answer') + 1, 'категория answer', df['answer'].apply(get_category))

# Сохраняем результат на диск
out_path = '/content/drive/MyDrive/chat_girls_llm_categorized.xlsx'
df.to_excel(out_path, index=False)
print(f'✅ Готово! Файл с категориями сохранён как {out_path}')