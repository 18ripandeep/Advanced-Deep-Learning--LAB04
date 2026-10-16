# Advanced-Deep-Learning--LAB04
# What Recurrence Can and Cannot Remember

Measuring gradients through time and comparing recurrent cells on a delayed-recall task.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
## Run
```bash
# Run your Jupyter Notebook containing your lab solutions
## Gate Verification
We confirmed the gate bias was applied correctly

## Prediction
**Prediction:** I expect gradient clipping to have little to no effect on the plain RNN at sequence length 30 because plain RNNs suffer from vanishing gradients rather than exploding gradients over long sequences.

## Results

### 1. Benchmark Table
```text
--- BENCHMARK RESULTS ---
Config: Plain RNN       | Seq Len: 10 | Seeds: ['0.491', '1.000', '0.495'] | Mean: 0.662
Config: Plain RNN       | Seq Len: 30 | Seeds: ['0.248', '0.251', '0.264'] | Mean: 0.254
Config: Plain RNN       | Seq Len: 60 | Seeds: ['0.250', '0.240', '0.233'] | Mean: 0.241
Config: LSTM (default)  | Seq Len: 10 | Seeds: ['1.000', '1.000', '1.000'] | Mean: 1.000
Config: LSTM (default)  | Seq Len: 30 | Seeds: ['0.248', '0.251', '0.255'] | Mean: 0.251
Config: LSTM (default)  | Seq Len: 60 | Seeds: ['0.249', '0.251', '0.254'] | Mean: 0.251
Config: LSTM (bias=3)   | Seq Len: 10 | Seeds: ['1.000', '1.000', '1.000'] | Mean: 1.000
Config: LSTM (bias=3)   | Seq Len: 30 | Seeds: ['1.000', '1.000', '1.000'] | Mean: 1.000
Config: LSTM (bias=3)   | Seq Len: 60 | Seeds: ['1.000', '1.000', '1.000'] | Mean: 1.000
Config: GRU (default)   | Seq Len: 10 | Seeds: ['1.000', '1.000', '1.000'] | Mean: 1.000
Config: GRU (default)   | Seq Len: 30 | Seeds: ['0.241', '0.257', '0.254'] | Mean: 0.251
Config: GRU (default)   | Seq Len: 60 | Seeds: ['0.248', '0.249', '0.240'] | Mean: 0.246
Config: GRU (bias=3)    | Seq Len: 10 | Seeds: ['1.000', '1.000', '1.000'] | Mean: 1.000
Config: GRU (bias=3)    | Seq Len: 30 | Seeds: ['1.000', '1.000', '1.000'] | Mean: 1.000
Config: GRU (bias=3)    | Seq Len: 60 | Seeds: ['1.000', '1.000', '1.000'] | Mean: 1.000
...
### 2. Gradient-Through-Time Measurement
```text
Steps back: [0, 9, 29, 59]
Plain RNN gradients:        ['0.00000', '0.00000', '0.00000', '0.00269']
LSTM default gradients:     ['0.00000', '0.00000', '0.00000', '0.00061']
LSTM bias=3 gradients:      ['0.00022', '0.00015', '0.00016', '0.00063']
...

### 3. Two Controlled Fixes
```text
--- PART A: FORGET-BIAS SWEEP ---
LSTM Forget Bias 0 | Seeds: ['0.247', '0.251', '0.258'] | Mean: 0.252
LSTM Forget Bias 1 | Seeds: ['0.495', '0.496', '1.000'] | Mean: 0.664
LSTM Forget Bias 2 | Seeds: ['1.000', '1.000', '1.000'] | Mean: 1.000
LSTM Forget Bias 3 | Seeds: ['1.000', '1.000', '1.000'] | Mean: 1.000

--- PART B: CLIPPING COMPARISON ---
Plain RNN | Clipping ON (1.0)    | Seeds: ['0.248', '0.251', '0.264'] | Mean: 0.254
Plain RNN | Clipping OFF (None)  | Seeds: ['0.248', '0.251', '0.264'] | Mean: 0.254
...

--- TIMING LIMITS ---
```text
Sequence Length  50 | Best Time: 0.0148 seconds
Sequence Length 100 | Best Time: 0.0240 seconds
Sequence Length 200 | Best Time: 0.0429 seconds
Sequence Length 400 | Best Time: 0.0874 seconds
...
## Conclusion
- **Gates:** Kept error signals from vanishing over long sequences.
- **Initialization:** Opened the gates right from step zero so training could begin successfully.
- **Clipping:** Had minimal impact because vanishing gradients (not explosions) were the main issue.
- **Limits of Recurrence:** Sequential bottlenecks and memory fading over long time spans. Direct connections help solve these by allowing gradients to skip steps.
