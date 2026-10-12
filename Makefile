.PHONY: build sample dense-demo test evaluate benchmark-100k run clean

build:
	cmake -S cpp -B build/cpp -DCMAKE_BUILD_TYPE=Release
	cmake --build build/cpp -j

sample:
	PYTHONPATH=. python scripts/build_sample_corpus.py

dense-demo:
	PYTHONPATH=. python scripts/build_dense_index.py --backend hashing

test: build sample dense-demo
	ctest --test-dir build/cpp --output-on-failure
	PYTHONPATH=. pytest -q tests/unit tests/integration

evaluate: build sample dense-demo
	PYTHONPATH=. python scripts/evaluate.py --dense-backend hashing --out artifacts/eval-demo.json

benchmark-100k: build
	PYTHONPATH=. python scripts/generate_synthetic_corpus.py --count 100000 --out data/synthetic-100k.jsonl
	PYTHONPATH=. python scripts/build_corpus.py --input data/synthetic-100k.jsonl --output data/synthetic-100k.normalized.tsv --workers 4
	PYTHONPATH=. python benchmarks/native_benchmark.py --tsv data/synthetic-100k.normalized.tsv --queries 1000 --out artifacts/native-100k.json

run: build sample
	PYTHONPATH=. python api/app.py

clean:
	rm -rf build .pytest_cache artifacts data/index data/synthetic-100k* **/__pycache__
