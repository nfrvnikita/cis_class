import os
import logging
import argparse
import ast
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import torch

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import Dataset

torch.cuda.empty_cache()

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """
    Устанавливает seed для воспроизводимости.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def map_tag(tag: str) -> str:
    """
    Группирует исходный тег в высокоуровневую категорию по его префиксу.
    """
    if tag.startswith("cs."):
        return "Computer Science"
    elif tag.startswith("stat."):
        return "Statistics"
    elif tag.startswith("math."):
        return "Mathematics"
    elif tag.startswith("q-bio"):
        return "Quantitative Biology"
    elif tag.startswith("physics."):
        return "Physics"
    elif tag.startswith("astro-ph"):
        return "Astrophysics"
    elif tag.startswith("cond-mat"):
        return "Condensed Matter"
    elif tag.startswith("eess."):
        return "Electrical Engineering"
    elif tag.startswith("nlin"):
        return "Nonlinear Sciences"
    elif tag == "cmp-lg":
        return "Computational Linguistics"
    elif tag == "quant-ph":
        return "Physics"
    else:
        return tag


def load_and_preprocess_data(
    json_file: str, rare_threshold: int = 50
) -> Tuple[pd.DataFrame, Dict[str, int], Dict[int, str]]:
    """
    Загружает данные из JSON-файла, выполняет парсинг тегов, группирует схожие классы
    и объединяет редкие классы в категорию "Other".
    """
    df: pd.DataFrame = pd.read_json(json_file)
    logger.info(f"Загружено {len(df)} примеров")

    df = df.dropna(subset=["title", "summary", "tag"]).reset_index(drop=True)

    def parse_first_tag(tag_str: str) -> Any:
        try:
            tags = ast.literal_eval(tag_str)
            if tags and isinstance(tags, list):
                return tags[0].get("term", None)
            return None
        except Exception as e:
            logger.error(f"Ошибка при разборе тега: {e}")
            return None

    df["parsed_tag"] = df["tag"].apply(parse_first_tag)
    df = df.dropna(subset=["parsed_tag"]).reset_index(drop=True)
    df["mapped_tag"] = df["parsed_tag"].apply(map_tag)
    df["text"] = df["title"] + " [SEP] " + df["summary"]

    tag_counts = df["mapped_tag"].value_counts()
    logger.info("Распределение классов после группировки:")
    logger.info(f"\n{tag_counts}")

    rare_tags = tag_counts[tag_counts < rare_threshold].index.tolist()
    if rare_tags:
        logger.info(
            f"Классы с менее чем {rare_threshold} примерами будут заменены на 'Other': {rare_tags}"
        )
        df.loc[df["mapped_tag"].isin(rare_tags), "mapped_tag"] = "Other"
        tag_counts = df["mapped_tag"].value_counts()
        logger.info("Распределение классов после объединения в 'Other':")
        logger.info(f"\n{tag_counts}")

    unique_tags = sorted(df["mapped_tag"].unique())
    tag2id: Dict[str, int] = {tag: idx for idx, tag in enumerate(unique_tags)}
    id2tag: Dict[int, str] = {idx: tag for tag, idx in tag2id.items()}
    df["label"] = df["mapped_tag"].map(tag2id)

    logger.info(f"Будет использоваться {len(unique_tags)} классов: {unique_tags}")
    return df, tag2id, id2tag


def tokenize_function(
    example: Dict[str, Any], tokenizer: DistilBertTokenizerFast
) -> Dict[str, Any]:
    """
    Токенизирует входной текст с ограничением в 512 токенов.
    """
    return tokenizer(example["text"], truncation=True, max_length=512)


def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """
    Вычисляет accuracy и macro F1.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average="macro")
    return {"accuracy": acc, "f1_macro": f1_macro}


def main(args: argparse.Namespace) -> None:
    """
    Основная функция: загрузка данных, обучение и сохранение модели.
    """
    set_seed(args.seed)
    df, tag2id, id2tag = load_and_preprocess_data(
        args.data_path, rare_threshold=args.rare_threshold
    )
    before_filtering = len(df)
    df = df[df.groupby("label")["label"].transform("count") > 1]
    after_filtering = len(df)
    logger.info(
        f"После фильтрации редких классов: удалено {before_filtering - after_filtering} примеров, осталось {after_filtering}"
    )

    train_df, test_df = train_test_split(
        df, test_size=0.1, random_state=args.seed, stratify=df["label"]
    )
    train_dataset = Dataset.from_pandas(train_df[["text", "label"]])
    test_dataset = Dataset.from_pandas(test_df[["text", "label"]])

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-cased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-cased", num_labels=len(tag2id)
    )
    if torch.cuda.is_available():
        logger.info("Использование CUDA: GPU найден!")
        model.to("cuda")
    else:
        logger.info("CUDA не найдена, используется CPU.")

    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer), batched=True
    )
    test_dataset = test_dataset.map(
        lambda x: tokenize_function(x, tokenizer), batched=True
    )
    train_dataset = train_dataset.remove_columns(["text"])
    test_dataset = test_dataset.remove_columns(["text"])
    train_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "label"]
    )
    test_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "label"]
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    logger.info("Начало обучения модели...")
    train_result = trainer.train()

    model_path = os.path.join(args.output_dir, "pytorch_model.pt")
    torch.save(model.state_dict(), model_path)
    tokenizer.save_pretrained(args.output_dir)

    metrics = train_result.metrics
    logger.info(f"Метрики обучения: {metrics}")
    logger.info("Оценка модели на тестовой выборке...")
    eval_result = trainer.evaluate()
    logger.info(f"Результаты оценки: {eval_result}")

    metrics_file = os.path.join(args.output_dir, "metrics.txt")
    with open(metrics_file, "w") as f:
        f.write("Training metrics:\n")
        f.write(str(metrics) + "\n")
        f.write("Evaluation metrics:\n")
        f.write(str(eval_result) + "\n")

    logger.info(f"Обученная модель и tokenizer сохранены в {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Классификация статей arXiv с использованием DistilBERT"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/arxivData.json",
        help="Путь к JSON-файлу с данными (например, data/arxivData.json)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models",
        help="Директория для сохранения модели и логов",
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=3, help="Количество эпох обучения"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Размер батча (на устройстве)"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-5, help="Скорость обучения"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Коэффициент weight decay"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed для воспроизводимости"
    )
    parser.add_argument(
        "--rare_threshold",
        type=int,
        default=50,
        help="Порог для объединения редких классов в 'Other'",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
