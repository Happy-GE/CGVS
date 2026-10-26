# Task-Aware Schema Benchmark Artifact

This artifact contains a self-contained, cleaned code package for reproducing the task-aware schema experiments for the `Exhaustive-best` and `Consensus-Val` methods.

## Entry Points

- `experiments/methods/run_classification_methods.py`
- `experiments/methods/run_regression_methods.py`

## Shared Experiment Modules

- `experiments/methods/shared_method_utils.py`
- `experiments/schema_gate_base.py`
- `experiments/schema_gate_consensus.py`

## Included Data

- `data/University`
- `data/Biodegradability`

## Example Commands

```bash
cd task_aware_schema_benchmark_artifact
python experiments/methods/run_classification_methods.py --methods Exhaustive-best,Consensus-Val
python experiments/methods/run_regression_methods.py --methods Exhaustive-best,Consensus-Val
```
