# Local LLM Evaluator

An application-driven benchmarking platform for evaluating open-source large language models across different inference runtimes, configurations, and hardware environments.

The goal is to help answer:

> Given the hardware, requirements, and constraints of a particular application, which open-source model should I use, and why?

Rather than producing a generic model leaderboard, this project aims to evaluate the tradeoffs between model quality, inference performance, resource consumption, and reliability across different workloads.

## Project Goals

The evaluator is designed around four primary evaluation dimensions:

- **Quality:** Task-specific correctness and output quality.
- **Performance:** Time to first token (TTFT), total latency, and generation throughput.
- **Resource Usage:** VRAM, RAM, and GPU utilization.
- **Reliability:** Success rates, errors, timeouts, and consistency across repeated executions.

These dimensions will be evaluated independently rather than blindly combined into a single score. The eventual goal is to provide application-specific recommendations that explain the tradeoffs behind each model selection.

## Architecture

The project separates three important dimensions:

- **Models:** Qwen, Llama, Gemma, Mistral, and other open-source models.
- **Inference Runtimes:** Ollama, llama.cpp, LM Studio, and potentially other inference engines.
- **Configurations:** Quantization, temperature, context length, GPU offloading, and other inference settings.

This separation makes it possible to compare models across runtimes without unnecessarily coupling the evaluation logic to a particular inference engine.

### Current Benchmark Pipeline

1. **Request Configuration:** Define the model, runtime, prompt, and inference parameters.
2. **Inference:** Execute warmup and measured inference requests through a common runtime interface.
3. **Measurement:** Capture client-observed latency and runtime-provided inference metrics.
4. **Failure Handling:** Record failed requests without terminating the entire benchmark.
5. **Analysis:** Calculate statistical summaries and reliability metrics.
6. **Environment Collection:** Capture hardware and software environment metadata.
7. **Persistence:** Serialize the complete benchmark execution into a structured JSON result.

## Implemented Features

### Inference Runtime Abstraction

- Common interface for inference runtimes.
- Initial Ollama runtime implementation.
- Structured generation requests and responses.
- Runtime-independent error handling with categorized exceptions.
- Streaming response processing and inference metric collection.

### Benchmark Execution

- Configurable warmup runs and measured iterations.
- Individual measurement tracking.
- Failure-aware execution that continues after unsuccessful requests.
- Separate tracking of successful and failed iterations.

### Performance and Reliability Analysis

- Mean, median, minimum, maximum, and standard deviation.
- Client-observed inference latency.
- Time to first token.
- Model loading and prompt evaluation durations.
- Generation duration and tokens per second.
- Success and failure rates.
- Failed request duration statistics.
- Error counts grouped by category.

### Environment Metadata

- Operating system and Python version.
- CPU information.
- Total system RAM.
- NVIDIA GPU model, VRAM capacity, and driver version.

Environment collection is best-effort: unavailable hardware information generates warnings rather than preventing benchmark execution.

### JSON Persistence

Every benchmark execution is saved as a self-contained JSON document containing:

- Execution metadata and configuration.
- Environment information.
- Warmup summary.
- Individual measured results.
- Performance and reliability analysis.
- Collection warnings.

Results include a unique execution ID, UTC timestamp, and schema version to support reproducibility and future analysis.

Local benchmark results are separated from curated published results.

## Initial Benchmark Baseline

The first benchmark uses **Qwen2.5-Coder 7B** through Ollama.

| Configuration | Value |
|---|---|
| Model | Qwen2.5-Coder 7B |
| Runtime | Ollama |
| GPU | NVIDIA RTX 3080 10GB |
| System RAM | 32GB |
| Workload | Basic code generation |
| Warmup runs | 2 |
| Measured iterations | 5 |
| Temperature | 0.0 |

### Results

| Metric | Mean |
|---|---:|
| Client latency | 426.3 ms |
| Time to first token | 37.5 ms |
| Generation duration | 387.3 ms |
| Generation throughput | 126.55 tokens/sec |
| Runtime total duration | 400.4 ms |
| Success rate | 100% |

All five measured requests completed successfully.

These results establish an initial baseline, not a comparative model ranking. More workloads, configurations, and runtimes are needed before drawing broader conclusions.

## Project Structure

```text
local-llm-evaluator/
├── benchmark/
│   ├── analysis.py
│   ├── environment.py
│   ├── models.py
│   ├── persistence.py
│   ├── run.py
│   └── runner.py
├── runtimes/
│   ├── base.py
│   └── ollama.py
├── tests/
│   ├── test_benchmark_analysis.py
│   ├── test_benchmark_persistence.py
│   ├── test_benchmark_runner.py
│   ├── test_ollama_runtime.py
│   └── test_runtime_base.py
├── results/
│   ├── local/
│   └── published/
├── requirements.txt
└── README.md
```

## Getting Started

### Requirements

- Python 3.12+
- Ollama
- A locally available Ollama model

### Installation

Clone the repository:

```bash
git clone https://github.com/StephenLee129/Local-LLM-Evaluator.git
cd Local-LLM-Evaluator
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/Scripts/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Pull the initial benchmark model:

```bash
ollama pull qwen2.5-coder:7b
```

### Run a Benchmark

Ensure Ollama is running, then execute:

```bash
python -m benchmark.run
```

Benchmark results are saved under `results/local/`.

### Run Tests

```bash
python -m pytest -v
```

## Roadmap

- [x] Runtime abstraction and initial Ollama integration
- [x] Benchmark execution with warmup and measured iterations
- [x] Failure-aware benchmark execution
- [x] Statistical performance and reliability analysis
- [x] Environment metadata collection
- [x] Structured JSON result persistence
- [ ] Dynamic resource monitoring (peak VRAM, RAM, and GPU utilization)
- [ ] Improved benchmark methodology and repeatability
- [ ] Task-specific quality evaluation
  - [ ] Code generation
  - [ ] Code understanding
  - [ ] Structured extraction
- [ ] Additional inference runtime integrations
  - [ ] llama.cpp
  - [ ] LM Studio
  - [ ] Hugging Face Transformers
- [ ] Application-level evaluation integrations
  - [ ] LangChain
  - [ ] LangGraph
- [ ] Application-driven model recommendations
- [ ] Read-only benchmark comparison dashboard

## Future Direction

The long-term goal is to evolve this project from a local benchmarking tool into an application-driven model evaluation platform.

Given a particular workload, hardware environment, and operational constraints, the evaluator should help identify suitable models and configurations while making the underlying performance, quality, and reliability tradeoffs transparent.