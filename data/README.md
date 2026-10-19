# Data

## Jiaxing charging transactions

- Official source: [Figshare, version 3](https://doi.org/10.6084/m9.figshare.28182251.v3)
- Repository mirror: [jiaxing_figshare_v3.zip](jiaxing/jiaxing_figshare_v3.zip)
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

Included files:

- `jiaxing/jiaxing_transactions_clean.parquet`: cleaned transaction stream used by the replay.
- `jiaxing/jiaxing_figshare_v3.zip`: original Figshare v3 archive containing `Dataset.zip` and `Code.zip`.
- `jiaxing/run_jiaxing_event_dispatch.py`: replay-data and topology builders.
- `jiaxing/run_charging_core_benchmark.py`: shared benchmark utilities.
- `jiaxing/jiaxing_audit_summary.json`: data audit summary.

The cleaned Parquet file is included in this repository and is sufficient for
`examples/run_vehicles_demo.py`.

## UrbanEV

- Official source: [Dryad](https://doi.org/10.5061/dryad.np5hqc04z)
- Latest complete archive: [UrbanEVDataset.zip](https://datadryad.org/downloads/file_stream/4587805)
- License: [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)

Included files:

- `urbanev/station_information.csv`: station metadata.
- `urbanev/station_distance.csv`: inter-station distance matrix.
- `urbanev/pile_rated_power.csv`: station rated-power information.

The full five-minute charging traces are available from the official dataset
and are not duplicated in this repository.
