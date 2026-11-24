# Local LLM Evaluator

An application-driven benchmarking and evaluation platform for evaluating open-source large language models across different inference runtimes, configurations, and hardware environments.

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

### Experiment Framework

A YAML-driven experiment framework provides a reusable way to define and execute experiments.

The configuration defines:

- Which models to evaluate.
- Which experiment types to execute.
- Experiment-specific parameters and workloads.

The Python implementation determines how each experiment runs, while the orchestrator coordinates execution, captures results, and isolates experiment failures.

Experiments currently execute sequentially. This avoids concurrent workloads contaminating hardware measurements, particularly when evaluating models on resource-constrained GPUs.

Example experiment suite:

```yaml
name: benchmark-stress-test

models:
  - qwen2.5-coder:7b

experiments:
  - type: benchmark
    config:
      task_id: code-generation-stress-test
      prompt: |
        Write a Python function called `is_palindrome`
        that accepts a string and returns True if the
        string is a palindrome, ignoring capitalization
        and spaces.

        Include a short explanation of your implementation.
      warmup_runs: 2
      iterations: 100
      temperature: 0.0
      max_tokens: 300
```

The framework is designed to support multiple experiment types while keeping their result schemas and evaluation logic independent.

### Current Benchmark Pipeline

1. **Request Configuration:** Define the model, runtime, prompt, and inference parameters.
2. **Inference:** Execute warmup and measured inference requests through a common runtime interface.
3. **Measurement:** Capture client-observed latency and runtime-provided inference metrics.
4. **Resource Monitoring:** Capture GPU and system memory measurements during execution.
5. **Failure Handling:** Record failed requests without terminating the entire benchmark.
6. **Analysis:** Calculate statistical summaries and reliability metrics.
7. **Environment Collection:** Capture hardware and software environment metadata.
8. **Persistence:** Serialize the complete experiment execution into a structured JSON result.

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
- Sequential experiment execution through the YAML-driven orchestrator.

### Performance and Reliability Analysis

- Mean, median, minimum, maximum, and standard deviation.
- Client-observed inference latency.
- Time to first token.
- Model loading and prompt evaluation durations.
- Generation duration and tokens per second.
- Success and failure rates.
- Failed request duration statistics.
- Error counts grouped by category.
- Longest consecutive failure sequence.
- Failure recovery and consecutive-failure transition analysis.

### Hardware Monitoring and Profiling

- GPU VRAM usage.
- GPU utilization.
- System RAM usage and availability.
- Periodic resource sampling during inference.
- Initial and final resource snapshots.
- Pre-warmup and post-warmup resource profiling.
- Per-iteration resource statistics.
- Graceful handling of unavailable monitoring capabilities.

Hardware monitoring is best-effort: unavailable hardware information generates warnings rather than preventing benchmark execution.

### Environment Metadata

- Operating system and Python version.
- CPU information.
- Total system RAM.
- NVIDIA GPU model, VRAM capacity, and driver version.

### YAML-Driven Experiment Execution

- Reusable experiment suite configurations.
- Multiple model definitions per suite.
- Multiple experiment definitions per suite.
- Experiment-specific configuration validation.
- Experiment registration and factory-based construction.
- Sequential orchestration with per-experiment failure isolation.
- Shared experiment result metadata with experiment-specific result payloads.

### JSON Persistence

Every experiment execution is saved as a structured JSON document containing:

- Execution metadata and configuration.
- Environment information.
- Warmup summary.
- Individual measured results.
- Performance and reliability analysis.
- Resource monitoring data.
- Collection warnings.

Results include unique execution identifiers and UTC timestamps to support reproducibility and future analysis.

Local benchmark results are separated from curated published results.

## Benchmark Results

### Initial Benchmark Baseline

The initial benchmark uses **Qwen2.5-Coder 7B** through Ollama.

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

#### Results

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

### Sustained Inference Stress Test

A separate 100-iteration benchmark was executed using the YAML experiment framework to validate sustained inference, resource monitoring, reliability analysis, and result persistence.

| Configuration | Value |
|---|---|
| Model | Qwen2.5-Coder 7B |
| Runtime | Ollama |
| GPU | NVIDIA RTX 3080 10GB |
| System RAM | 32GB |
| Workload | Repeated code-generation requests |
| Warmup runs | 2 |
| Measured iterations | 100 |
| Maximum generated tokens | 300 |
| Temperature | 0.0 |
| Total execution time | 247.76 seconds |

#### Performance Results

| Metric | Mean | Minimum | Maximum |
|---|---:|---:|---:|
| Client latency | 2.394 sec | 2.329 sec | 2.529 sec |
| Time to first token | 29.36 ms | 15.23 ms | 32.46 ms |
| Generation throughput | 127.18 tokens/sec | 120.25 tokens/sec | 130.85 tokens/sec |

Client-observed latency had a standard deviation of approximately 47 ms, indicating relatively consistent performance across the repeated requests.

#### Reliability Results

| Metric | Result |
|---|---:|
| Total iterations | 100 |
| Successful iterations | 100 |
| Failed iterations | 0 |
| Success rate | 100% |
| Failure rate | 0% |
| Longest consecutive failures | 0 |
| Warnings | 0 |

All 100 requests completed successfully, and all individual results were persisted.

The recovery rate was not applicable because no failures occurred during the test. Failure recovery is covered by analysis tests but was not exercised through an actual runtime failure in this stress-test execution.

#### Resource Usage

| Measurement | Pre-warmup | Post-warmup |
|---|---:|---:|
| GPU memory | 2,089.84 MB | 6,856.65 MB |
| GPU utilization | 0% | 97% |
| System RAM used | 12,856.68 MB | 13,657.67 MB |
| System RAM available | 19,037.21 MB | 18,236.21 MB |

During measured inference:

- Average GPU memory usage was approximately 6.89 GB.
- Peak GPU memory usage averaged approximately 6.89 GB.
- Average GPU utilization was approximately 94.7%.
- Average system RAM usage was approximately 13.75 GB.
- No obvious progressive memory accumulation or sustained performance degradation was observed.

Resource snapshots represent system-wide measurements, not exclusively model-attributed memory usage.

#### Stress-Test Assessment

This experiment validates that the existing benchmark pipeline can sustain 100 sequential inference requests while collecting performance and resource measurements, calculating reliability statistics, and persisting the complete execution.

All requests generated the configured maximum of 300 tokens. Consequently, this test evaluates sustained inference behavior rather than task correctness or code quality.

The experiment also does not test concurrent inference, multiple models, or actual runtime failure and recovery.

The YAML configuration is retained as a reusable stress-test and regression-testing example for future changes to the benchmark runner, monitoring, runtime integrations, and experiment orchestration.

## Project Structure

```text
local-llm-evaluator/
├── benchmark/
│   ├── analysis.py
│   ├── environment.py
│   ├── models.py
│   ├── monitoring.py
│   ├── persistence.py
│   ├── run.py
│   └── runner.py
├── evaluation/
├── experiments/
│   ├── base.py
│   ├── benchmark.py
│   ├── benchmark_config.py
│   ├── config_loader.py
│   ├── factory.py
│   ├── models.py
│   ├── orchestrator.py
│   ├── persistence.py
│   ├── registry.py
│   └── run.py
├── monitoring/
├── runtimes/
│   ├── base.py
│   └── ollama.py
├── examples/
│   └── benchmark/
│       ├── basic-code-generation.yaml
│       └── stress-test.yaml
├── tests/
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
- NVIDIA GPU recommended for GPU resource monitoring

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

### Run an Experiment Suite

Ensure Ollama is running, then execute an example YAML experiment suite:

```bash
python -m experiments.run examples/benchmark/basic-code-generation.yaml
```

To run the sustained inference stress test:

```bash
python -m experiments.run examples/benchmark/stress-test.yaml
```

Experiment results are saved under `results/local/`.

### Run Tests

```bash
python -m pytest -v
```

## Roadmap

### Phase 1 — Benchmarking Foundation

- [x] Runtime abstraction and initial Ollama integration.
- [x] Benchmark execution with warmup and measured iterations.
- [x] Failure-aware benchmark execution.
- [x] Statistical performance and reliability analysis.
- [x] Environment metadata collection.
- [x] Structured JSON result persistence.

### Phase 2 — Hardware Benchmarking and Reliability

- [x] GPU VRAM and utilization monitoring.
- [x] System RAM monitoring.
- [x] Periodic resource sampling and per-iteration statistics.
- [x] Initial and final resource snapshots.
- [x] Pre-warmup and post-warmup profiling.
- [x] Failure recovery and consecutive-failure analysis.
- [x] YAML-driven experiment execution.
- [x] Sustained inference stress testing.
- [x] End-to-end experiment execution and result persistence.

Hardware monitoring and profiling are considered complete for the current feature scope. Hardware capacity profiling and additional workload scenarios are deferred until broader model and runtime support is available.

### Phase 3 — Model Quality Evaluation

- [ ] Dataset-driven task evaluation.
- [ ] Code generation evaluation.
- [ ] Code understanding evaluation.
- [ ] Structured extraction evaluation.
- [ ] Task-specific correctness and quality metrics.
- [ ] Isolated execution and testing of generated code.
- [ ] Model comparisons across evaluation workloads.

Quality evaluation will use its own experiment implementations and result schemas rather than being folded into performance benchmarking.

### Future Phases

- [ ] Additional inference runtime integrations.
  - [ ] llama.cpp
  - [ ] LM Studio
  - [ ] Hugging Face Transformers
- [ ] Standardized evaluation framework integrations.
  - [ ] LM Evaluation Harness
- [ ] Application-level evaluation integrations.
  - [ ] LangChain
  - [ ] LangGraph
- [ ] Broader model and runtime support.
- [ ] Hardware capacity profiling.
- [ ] Application-driven model recommendations.
- [ ] Read-only benchmark comparison dashboard.

## Future Direction

The long-term goal is to evolve this project from a local benchmarking tool into an application-driven model evaluation platform.

Given a particular workload, hardware environment, and operational constraints, the evaluator should help identify suitable models and configurations while making the underlying performance, quality, resource consumption, and reliability tradeoffs transparent.

Rather than optimizing for a single benchmark score, the project aims to help answer a more practical question: **Which model is the best fit for this application, and what tradeoffs come with that choice?**