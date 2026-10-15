"""
Lesson 09 Exercises:
1. Load GLUE (mrpc config) and inspect first 5 examples
2. Stream a large dataset (e.g. c4 subset or rotten_tomatoes) and benchmark rows processed
3. Convert dataset to Parquet vs CSV and compare compression ratio
4. Create a 70/15/15 train/val/test split with fixed random seed
"""

import time
from pathlib import Path
from datasets import load_dataset


def exercise_1():
    print("\n--- Exercise 1: Load GLUE (MRPC) ---")
    ds = load_dataset("nyu-mll/glue", "mrpc", split="train")
    print(f"Total rows in MRPC train: {len(ds)}")
    for i in range(min(5, len(ds))):
        ex = ds[i]
        print(f"[{i+1}] Label: {ex['label']}")
        print(f"     Sentence 1: {ex['sentence1'][:70]}...")
        print(f"     Sentence 2: {ex['sentence2'][:70]}...")


def exercise_2():
    print("\n--- Exercise 2: Benchmark Dataset Streaming ---")
    # Stream a dataset (using Salesforce/wikitext with streaming=True)
    stream_ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="train", streaming=True)
    start_time = time.time()
    count = 0
    duration = 5  # Run for 5 seconds
    print(f"Streaming rows for {duration} seconds...")
    for _ in stream_ds:
        count += 1
        if time.time() - start_time >= duration:
            break
    elapsed = time.time() - start_time
    print(f"Processed {count} rows in {elapsed:.2f} seconds ({count/elapsed:.1f} rows/sec)")


def exercise_3():
    print("\n--- Exercise 3: Parquet vs CSV File Size Comparison ---")
    ds = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="train")
    out_dir = Path("outputs/format_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "dataset.csv"
    parquet_path = out_dir / "dataset.parquet"

    ds.to_csv(str(csv_path))
    ds.to_parquet(str(parquet_path))

    csv_size = csv_path.stat().st_size
    parquet_size = parquet_path.stat().st_size

    print(f"CSV Size:     {csv_size:,} bytes")
    print(f"Parquet Size: {parquet_size:,} bytes")
    print(f"Compression:  Parquet is {csv_size / parquet_size:.2f}x smaller than CSV")


def exercise_4():
    print("\n--- Exercise 4: 70 / 15 / 15 Train / Val / Test Split ---")
    ds = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="train")
    total_len = len(ds)

    # 70% train, 30% for val+test
    split1 = ds.train_test_split(test_size=0.30, seed=42)
    train_ds = split1["train"]

    # Split the remaining 30% equally into val (15%) and test (15%)
    split2 = split1["test"].train_test_split(test_size=0.50, seed=42)
    val_ds = split2["train"]
    test_ds = split2["test"]

    print(f"Total rows: {total_len}")
    print(f"  Train: {len(train_ds)} ({len(train_ds)/total_len:.1%}) - Target: 70%")
    print(f"  Val:   {len(val_ds)} ({len(val_ds)/total_len:.1%}) - Target: 15%")
    print(f"  Test:  {len(test_ds)} ({len(test_ds)/total_len:.1%}) - Target: 15%")


if __name__ == "__main__":
    print("=" * 60)
    print("Lesson 09: Data Management Exercises")
    print("=" * 60)
    exercise_1()
    exercise_3()
    exercise_4()
    exercise_2()
    print("\nAll exercises completed successfully!")
