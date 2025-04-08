import os
import streamlit as st
import torch
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    DistilBertConfig,
)
from typing import Tuple, List

nltk.download("stopwords")
nltk.download("wordnet")

torch.classes.__path__ = []

st.set_page_config(page_title="Классификация статей arXiv", layout="centered")

st.markdown(
    """
    <style>
        /* Общий фон */
        .reportview-container {
            background: linear-gradient(135deg, #f6f8fa, #e9eff5);
        }
        /* Заголовки и текст */
        .main > .block-container h1, .main > .block-container h2 {
            font-family: 'Helvetica Neue', sans-serif;
        }
        .main > .block-container {
            padding: 2rem;
        }
        /* Кнопки */
        div.stButton > button {
            background-color: #2ecc71;
            color: #ffffff;
            border-radius: 10px;
            border: none;
            padding: 10px 20px;
            font-size: 18px;
            transition: background-color 0.3s ease;
        }
        div.stButton > button:hover {
            background-color: #27ae60;
        }
        /* Поля ввода */
        .stTextInput>div>div>input {
            border: 2px solid #ced4da;
            border-radius: 5px;
            padding: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("О приложении")
st.sidebar.info(
    """
    Это приложение использует модель DistilBERT для классификации статей arXiv по тематике.
    Введите название статьи и (опционально) аннотацию, а модель выдаст топ-классы, суммарная вероятность которых 
    превышает заданный порог.
    """
)

user_threshold = st.sidebar.slider(
    "Порог суммарной вероятности для выбора топ-классов",
    min_value=0.5,
    max_value=1.0,
    value=0.95,
    step=0.01,
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Входные данные**")
st.sidebar.markdown("- *Название статьи* (обязательное)")
st.sidebar.markdown("- *Аннотация (abstract)* (необязательное)")


def clean_text(text: str) -> str:
    """
    Осуществляет предварительную обработку текста:
    - Приведение к нижнему регистру
    - Удаление пунктуации и лишних пробелов
    - Токенизация (split по пробелам)
    - Удаление английских стоп-слов
    - Лемматизация
    Возвращает очищенный текст.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)


@st.cache_resource(show_spinner=False)
def load_model(
    model_dir: str = "./models/distilbert-base-cased",
) -> Tuple[DistilBertForSequenceClassification, DistilBertTokenizerFast]:
    """
    Загружает токенизатор, конфигурацию модели и веса,
    сохранённые через torch.load.
    В директории model_dir должны быть файлы модели и токенизатора в формате Hugging Face,
    а также файл с весами (pytorch_model.pt).
    """
    model_dir = os.path.abspath(model_dir)

    tokenizer = DistilBertTokenizerFast.from_pretrained(
        model_dir, local_files_only=True
    )
    config = DistilBertConfig.from_pretrained(model_dir, local_files_only=True)

    config.num_labels = 10

    state_dict_path = os.path.join(model_dir, "pytorch_model.pt")
    if not os.path.exists(state_dict_path):
        st.error(f"Файл с весами не найден: {state_dict_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = torch.load(state_dict_path, map_location=device)

    if "distilbert.embeddings.word_embeddings.weight" in state_dict:
        saved_vocab_size = state_dict[
            "distilbert.embeddings.word_embeddings.weight"
        ].shape[0]
        config.vocab_size = saved_vocab_size
    else:
        config.vocab_size = tokenizer.vocab_size

    model = DistilBertForSequenceClassification(config)

    model.load_state_dict(state_dict)

    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
    return model, tokenizer


def predict(
    text: str,
    model: DistilBertForSequenceClassification,
    tokenizer: DistilBertTokenizerFast,
) -> np.ndarray:
    """
    Выполняет предсказание для заданного текста и возвращает вероятностное распределение по классам.
    Обратите внимание, что текст должен проходить ту же предварительную обработку, что и обучающие данные.
    """
    inputs = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits.cpu().numpy()[0]
    probs = np.exp(logits) / np.sum(np.exp(logits))
    return probs


def select_top95(probs: np.ndarray, class_names: List[str]) -> List[Tuple[str, float]]:
    """
    Отбирает классы по правилу: суммарная вероятность выбранных классов должна быть ≥95%.
    Возвращает список кортежей (имя класса, вероятность).
    """
    sorted_indices = np.argsort(probs)[::-1]
    selected = []
    accumulated = 0.0
    for idx in sorted_indices:
        prob = probs[idx]
        accumulated += prob
        selected.append((class_names[idx], prob))
        if accumulated >= 0.95:
            break
    return selected


st.title("Классификация статей arXiv")
st.markdown(
    """
    Введите название статьи и (опционально) её аннотацию в поля ниже.
    Модель выполнит предсказание вероятностей принадлежности к тематикам и выведет топ-классы.
    """
)

col1, col2 = st.columns(2)
with col1:
    title_input = st.text_area(
        "Название статьи",
        value="",
        placeholder="Например: A Novel Approach to Visual Recognition",
    )
with col2:
    abstract_input = st.text_area(
        "Аннотация (abstract)",
        value="",
        placeholder="Например: В данной статье представлен новый метод обработки изображений...",
    )

if st.button("Предсказать тематику"):
    if title_input.strip() == "":
        st.error("Пожалуйста, введите хотя бы название статьи.")
    else:
        if abstract_input.strip() == "":
            input_text = title_input.strip()
        else:
            input_text = f"{title_input.strip()} [SEP] {abstract_input.strip()}"

        input_text = clean_text(input_text)

        with st.spinner("Загрузка модели и выполнение предсказания..."):
            model, tokenizer = load_model()
            probs = predict(input_text, model, tokenizer)

        class_mapping = {
            2: "Computer Science",
            9: "Statistics",
            5: "Mathematics",
            8: "Quantitative Biology",
            7: "Physics",
            1: "Computational Linguistics",
            6: "Other",
            4: "Electrical Engineering",
            3: "Condensed Matter",
            0: "Astrophysics",
        }

        num_classes = len(probs)
        class_names = [class_mapping.get(i, f"Class_{i}") for i in range(num_classes)]
        top_classes = select_top95(probs, class_names)

        tab1, tab2, tab3 = st.tabs(
            ["Результаты", "Подробная статистика", "Информация о модели"]
        )

        with tab1:
            st.subheader("Предсказанная тематика (топ-классы)")
            for label, prob in top_classes:
                st.markdown(f"- **{label}** — {prob * 100:.2f}%")

            st.markdown("### Визуализация топ-классов")
            df_top = pd.DataFrame(top_classes, columns=["Класс", "Вероятность"])
            df_top = df_top.set_index("Класс")
            st.bar_chart(df_top)

        with tab2:
            st.subheader("Полное распределение вероятностей по классам")
            df_full = pd.DataFrame({"Класс": class_names, "Вероятность": probs})
            df_full["Вероятность (%)"] = df_full["Вероятность"] * 100
            st.dataframe(df_full.sort_values(by="Вероятность", ascending=False))
            st.markdown("### График распределения вероятностей")
            df_full_sorted = df_full.sort_values(by="Вероятность", ascending=True)
            df_full_chart = df_full_sorted.set_index("Класс")["Вероятность (%)"]
            st.bar_chart(df_full_chart)

        with tab3:
            st.subheader("Информация о модели")
            st.write("**Архитектура модели:** DistilBERT для классификации")
            st.write(f"**Число меток (labels):** {model.config.num_labels}")
            st.write("**Конфигурация модели:**")
            st.json(model.config.to_dict())
