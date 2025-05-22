# Test Scripts for vLLM

## Some Useful Scripts
- `cache_evicion.py`: Set up several sequences to fill the cache and observe the evictor's behavior under hit and miss conditions.
- `blocksize_batch_exp.sh`, `blocksize_batch_exp.sh`: for block size experiment.
- `positional_dependency.sh`, `positional_dependency.py`: for positional dependency experiment.
- `wikiQA_2Q_valid.py`, `wikiQA_2Q_valid_cut.py`
- `./synthesis/`: for HotSpot and Distribution Shift experiments.
- `./data_process/`: process wikiQA and view results.

## HotSpot Test (Example: SQuAD)
### Step 0: Prepare Environment
Download [my vLLM Eviction Strategy Integration](https://github.com/Y-aang/vllm.git). Build vLLM from source (Reference to vLLM official manual).
### Step 1: HotSpot sampling
```
python squad_hotspot_sample.py > ./sample/squad_hotspot_sample.txt 2>&1
```
### Step 2: Run Experiment
Manually set cache strategy in `Qwen2.5_script.sh` and vLLM's `make_evictor()`. 

Fill in the configuration in `model_config.py` as instructed in the code comments.
```
python squad_hotspot_sample.py > ./sample/squad_hotspot_sample.txt 2>&1
```
### Step 3: Collect Data
```
python collect_result.py
```
### Step 4: Plot the charts
Copy the `.csv`. Use `view_graph.ipynb`. (From [Simulator tool box](https://github.com/Y-aang/vLLM-Eviction-Simulator.git)) 