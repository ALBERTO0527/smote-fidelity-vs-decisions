# Is faithful synthetic data good for decisions?

Code for our project in II2202 Research Methodology and Scientific Writing (KTH, autumn 2026), by Alberto Tonni and Greta Severi.

When one class is rare, practitioners often generate synthetic examples of it and must choose how to configure the generator. The project asks what is lost when a SMOTE configuration is chosen by the fidelity of its synthetic data rather than by the expected cost of the decisions it leads to. The case study is root-cause diagnosis of hardware failures in microwave networks.

## Contents

| Folder | What it is |
|---|---|
| [`quantitative-exercise/`](quantitative-exercise/) | Course quantitative exercise (November 2026): exploratory analysis, learning curve, scarcity of the rare class, and how small a difference in expected cost the experiment can detect |

The code of the main experiment will be added later.

## Data

We use the public dataset of Di Cicco et al., *Machine Learning for Failure Management in Microwave Networks: A Data-Centric Approach*, IEEE TNSM 21(5), 2024, [doi:10.1109/TNSM.2024.3406934](https://doi.org/10.1109/TNSM.2024.3406934). It is released under GPL-3.0 at [bonsai-lab-polimi/tnsm2024-data-centric](https://github.com/bonsai-lab-polimi/tnsm2024-data-centric) and is not copied here. Download it with:

```bash
mkdir -p data
curl -L -o data/dataset_TNSM_06995_v2.csv https://raw.githubusercontent.com/bonsai-lab-polimi/tnsm2024-data-centric/main/dataset_TNSM_06995_v2.csv
```

## Running

Tested with Python 3.14 on macOS (Apple M5).

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python quantitative-exercise/analysis.py data/dataset_TNSM_06995_v2.csv           # about 9 min
python quantitative-exercise/pilot_group_split.py data/dataset_TNSM_06995_v2.csv  # about 2 min
```

All seeds are fixed. `analysis.py` writes the five figures of the report to `quantitative-exercise/figures/` and every number to `quantitative-exercise/results.json`; `pilot_group_split.py` writes `quantitative-exercise/pilot_group_split.json`. The outputs used in the report are committed, so they can be checked without running anything.

## Use of AI tools

The code was written with the help of a large language model (Claude, Anthropic) and checked by the authors, who reran it and verified the figures and numbers against the data.

## License

Code: MIT (see [LICENSE](LICENSE)). The dataset keeps its own GPL-3.0 license.
