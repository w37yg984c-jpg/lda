import argparse
import csv
import json
import unicodedata
from pathlib import Path

from qhchina.analytics.topicmodels import LDAGibbsSampler


def tokenize_poem(poem: dict) -> list[str]:
    text = "".join(poem["paragraphs"])
    return [
        char
        for char in text
        if not unicodedata.category(char).startswith("P") and not char.isspace()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train character-based LDA topic models on 全唐詩."
    )
    parser.add_argument(
        "corpus",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("全唐诗.json"),
        help="Path to the poetry JSON file (default: 全唐诗.json beside this script)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Gibbs sampling iterations per model (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible models (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("topic_model_results.csv"),
        help="Output CSV path (default: topic_model_results.csv beside this script)",
    )
    args = parser.parse_args()

    with args.corpus.open(encoding="utf-8") as corpus_file:
        poems = json.load(corpus_file)

    documents = [tokenize_poem(poem) for poem in poems]
    print(
        f"Loaded {len(poems):,} poems; "
        f"{sum(map(len, documents)):,} character tokens after cleaning."
    )

    results = []
    for n_topics in (10, 20):
        print(f"\n=== {n_topics} topics ===", flush=True)
        model = LDAGibbsSampler(
            n_topics=n_topics,
            iterations=args.iterations,
            random_state=args.seed,
        )
        model.fit(documents)

        print(f"Perplexity: {model.perplexity():.4f}")
        for topic_id in range(n_topics):
            top_words = "、".join(
                word for word, _ in model.get_topic_words(topic_id, n_words=10)
            )
            print(f"\nTopic {topic_id + 1}: {top_words}")

            row = {
                "n_topics": n_topics,
                "topic": topic_id + 1,
                "top_10_characters": top_words,
            }
            for rank, (doc_id, probability) in enumerate(
                model.get_top_documents(topic_id, n_docs=2), start=1
            ):
                poem = poems[doc_id]
                author = poem.get("author") or "作者未載"
                row[f"poem_{rank}_title"] = poem["title"]
                row[f"poem_{rank}_author"] = author
                row[f"poem_{rank}_topic_probability"] = probability
                print(
                    f"  {poem['title']} — {author} "
                    f"(θ={probability:.4f})"
                )
            results.append(row)

    fieldnames = [
        "n_topics",
        "topic",
        "top_10_characters",
        "poem_1_title",
        "poem_1_author",
        "poem_1_topic_probability",
        "poem_2_title",
        "poem_2_author",
        "poem_2_topic_probability",
    ]
    with args.output.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCSV results written to {args.output}")


if __name__ == "__main__":
    main()
