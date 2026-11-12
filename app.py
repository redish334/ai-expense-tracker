import os
import time
import pandas as pd
import streamlit as st
from PIL import Image
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# 1. Опис структури відповіді (JSON Schema)
class ExpenseItem(BaseModel):
    name: str = Field(description="Назва товару, послуги або статті витрат")
    quantity: float = Field(description="Кількість (за замовчуванням 1, якщо не вказано)")
    price: float = Field(description="Загальна сума витрати")
    category: str = Field(description="Категорія: Продукти, Кафе/Ресторани, Побутова хімія, Транспорт, Комунальні/Зв'язок, Розваги/Інше")

class ExpenseReport(BaseModel):
    source_title: str = Field(description="Назва магазину, банку або заголовок звіту (напр. 'Витрати за місяць')")
    period_or_date: str = Field(description="Дата або період (напр. 'Листопад 2026' або РРРР-ММ-ДД)")
    currency: str = Field(description="Валюта (напр. UAH, USD, EUR)")
    total_amount: float = Field(description="Загальна сума всіх витрат")
    items: list[ExpenseItem]

# 2. Налаштування сторінки Streamlit
st.set_page_config(page_title="AI Expense Tracker", page_icon="💸", layout="centered")
st.title("💸 Розумний трекер витрат (Gemini API)")
st.write("Аналізуйте витрати за місяць із довільного тексту/нотаток або завантажуйте фото чеків та виписок.")

api_key = st.sidebar.text_input("Введіть Gemini API Key", type="password", value=os.getenv("GEMINI_API_KEY", ""))

tab1, tab2 = st.tabs(["📝 Список витрат текстом", "🧾 Фото чека / скріншот"])

input_content = None
mode = None

with tab1:
    sample_text = (
        "Витрати за листопад 2026:\n"
        "- Закупка продуктів у Сільпо (м'ясо, овочі, сир) — 1850 грн\n"
        "- Кава та круасан перед парами — 145 грн\n"
        "- Поповнення проїзного на метро та автобус — 300 грн\n"
        "- Оплата мобільного зв'язку та інтернету в гуртожитку — 450 грн\n"
        "- Побутова хімія та пральний порошок в Eva — 620 грн\n"
        "- Вечеря з друзями у піцерії — 780 грн\n"
        "- Таксі додому ввечері — 260 грн\n"
        "- Квитки в кіно — 340 грн\n"
        "- Замовлення кави в зернах та солодощів на Rozetka — 890 грн"
    )
    user_text = st.text_area("Вставте список покупок або нотатки за місяць у довільному форматі:", value=sample_text, height=220)
    if st.button("Проаналізувати текст", key="btn_text"):
        input_content = user_text
        mode = "text"

with tab2:
    uploaded_file = st.file_uploader("Оберіть зображення чека або виписки (JPG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Завантажене зображення", width=350)
        if st.button("Розпізнати зображення", key="btn_img"):
            input_content = image
            mode = "image"

# 3. Обробка запиту через Gemini API
if input_content is not None:
    if not api_key:
        st.error("Будь ласка, введіть Gemini API Key у бічній панелі зліва.")
    else:
        with st.spinner("Структуруємо та категоризуємо витрати через Gemini API..."):
            try:
                client = genai.Client(api_key=api_key)
                prompt = (
                    "Проаналізуй ці дані про витрати (текстовий список за місяць або фото чека/виписки). "
                    "Витягни всі позиції, суми, загальний підсумок, період/дату та розподіли кожну витрату по категоріях."
                )

                available_models = []
                for m in client.models.list():
                    name = m.name.replace("models/", "")
                    if "gemini" in name and "embedding" not in name and "aqa" not in name:
                        available_models.append(name)

                if "gemini-3.8-flash" in available_models:
                    available_models.remove("gemini-3.8-flash")
                    available_models.insert(0, "gemini-3.8-flash")

                response = None
                last_error = None
                used_model = None

                for model_name in available_models:
                    for attempt in range(2):
                        try:
                            response = client.models.generate_content(
                                model=model_name,
                                contents=[input_content, prompt],
                                config=types.GenerateContentConfig(
                                    response_mime_type="application/json",
                                    response_schema=ExpenseReport,
                                    temperature=0.1,
                                ),
                            )
                            used_model = model_name
                            break
                        except Exception as model_err:
                            last_error = model_err
                            if "503" in str(model_err):
                                time.sleep(2)
                                continue
                            break
                    if response is not None:
                        break

                if response is None:
                    raise last_error

                report = ExpenseReport.model_validate_json(response.text)

                st.caption(f"Оброблено моделлю: `{used_model}`")
                st.success(f"Джерело: **{report.source_title}** | Період: **{report.period_or_date}** | Загалом: **{report.total_amount} {report.currency}**")

                # Таблиця витрат
                df = pd.DataFrame([item.model_dump() for item in report.items])
                st.subheader("Структурована таблиця витрат")
                st.dataframe(df, use_container_width=True)

                # Аналітика за категоріями
                st.subheader("Розподіл витрат за категоріями")
                category_summary = df.groupby("category")["price"].sum()
                st.bar_chart(category_summary)

                # Експорт у CSV
                csv = df.to_csv(index=False).encode("utf-8-sig")
                st.download_button("Завантажити місячний звіт у CSV", data=csv, file_name="monthly_expenses.csv", mime="text/csv")

            except Exception as e:
                st.error(f"Помилка при зверненні до API: {e}")