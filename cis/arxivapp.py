import os
import streamlit as st
import torch
import numpy as np
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    DistilBertConfig,
)
from typing import Tuple, List

torch.classes.__path__ = []

st.set_page_config(page_title="Классификация статей arXiv", layout="centered")

st.markdown(
    """
    <style>
    /* Изменение стиля заголовков */
    .main > .block-container h1 {
        font-family: 'Helvetica Neue', sans-serif;
        color: #2c3e50;
    }
    .main > .block-container h2 {
        color: #34495e;
    }
    /* Изменение стиля кнопок */
    div.stButton > button {
        background-color: #2ecc71;
        color: white;
        font-size:18px;
        border-radius:10px;
        border: none;
        padding: 10px 20px;
    }
    div.stButton > button:hover {
        background-color: #27ae60;
        color: white;
    }
    /* Настройка боковой панели */
    .sidebar .sidebar-content {
        font-family: 'Arial', sans-serif;
        color: #2c3e50;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("О приложении")
st.sidebar.info(
    """
    Это приложение использует модель DistilBERT для классификации статей arXiv по тематике.
    Введите название статьи и (опционально) аннотацию, и модель выведет топ-классы, суммарная вероятность которых ≥95%.
    """
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Входные данные**")
st.sidebar.markdown("- *Название статьи* (обязательное)")
st.sidebar.markdown("- *Аннотация (abstract)* (необязательное)")


@st.cache_resource(show_spinner=False)
def load_model(
    model_dir: str = "./models/distilbert-base-cased",
) -> Tuple[DistilBertForSequenceClassification, DistilBertTokenizerFast]:
    """
    Загружает токенизатор, конфигурацию модели и веса, сохранённые через torch.load.
    В директории model_dir должны быть:
      - файлы модели и токенизатора в формате Hugging Face (config.json, tokenizer.*, и т.п.)
      - файл с весами, сохранёнными через torch.save(model.state_dict(), "pytorch_model.pt")
    """
    model_dir = os.path.abspath(model_dir)
    print(model_dir)
    tokenizer = DistilBertTokenizerFast.from_pretrained(
        model_dir, local_files_only=True
    )
    config = DistilBertConfig.from_pretrained(
        model_dir, local_files_only=True
    )
    config.num_labels = 10
    model = DistilBertForSequenceClassification.from_pretrained(
        model_dir, config=config, local_files_only=True
    )

    state_dict_path = os.path.join(model_dir, "pytorch_model.pt")
    if os.path.exists(state_dict_path):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        state_dict = torch.load(state_dict_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        st.error(f"Файл с весами не найден: {state_dict_path}")

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


# Основной интерфейс
st.title("Классификация статей arXiv")
st.markdown(
    """
    Введите название статьи и (опционально) её аннотацию в поля ниже.
    
    Модель выполнит предсказание вероятностей принадлежности к тематикам и выведет топ-классы, суммарная вероятность которых превышает **95%**.
    """
)

# Основные поля ввода
col1, col2 = st.columns(2)
with col1:
    title_input = st.text_input(
        "Название статьи", value="", placeholder="Введите название статьи"
    )
with col2:
    abstract_input = st.text_input(
        "Аннотация (abstract)",
        value="",
        placeholder="Введите аннотацию (необязательно)",
    )

if st.button("Предсказать тематику"):
    if title_input.strip() == "":
        st.error("Пожалуйста, введите хотя бы название статьи.")
    else:
        if abstract_input.strip() == "":
            input_text = title_input.strip()
        else:
            input_text = f"{title_input.strip()} [SEP] {abstract_input.strip()}"

        with st.spinner("Загрузка модели и выполнение предсказания..."):
            model, tokenizer = load_model()
            probs = predict(input_text, model, tokenizer)

        if hasattr(model.config, "id2label"):
            id2label = model.config.id2label
            num_classes = len(probs)
            class_names = [
                id2label.get(str(i), f"Class_{i}") for i in range(num_classes)
            ]
        else:
            class_names = [
                "Computer Science",
                "Statistics",
                "Mathematics",
                "Quantitative Biology",
                "Physics",
                "Astrophysics",
                "Condensed Matter",
                "Electrical Engineering",
                "Nonlinear Sciences",
                "Computational Linguistics",
                "Other",
            ]

        top_classes = select_top95(probs, class_names)

        st.subheader("Предсказанная тематика (топ-95% вероятности):")
        for label, prob in top_classes:
            st.markdown(f"- **{label}** — {prob * 100:.2f}%")

        st.markdown("---")
        with st.expander("Посмотреть полный список вероятностей"):
            for name, p in zip(class_names, probs):
                st.write(f"{name}: {p * 100:.2f}%")
