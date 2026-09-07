# Recorded H100 execution

The 7 September 2026 audit records 970 distinct completed runs totalling **808.589722222 H100 GPU-hours**. Rounded to the precision appropriate for describing overall resources, this is **at least about 809 H100 GPU-hours**.

Each retained source record reports one NVIDIA H100 80GB HBM3 and wall-clock seconds. Hours are `wall_seconds * accelerator_count /3600`. The run clock begins before per-cell data/model preparation and ends after generation, fitting, evaluation and output preparation. This is elapsed accelerator allocation during those runs, not a measurement of GPU utilisation. Queue time and execution without a complete timing record are omitted.

The count is a lower bound for the project's saved development and evaluation work. It is not the cost of only the final comparisons, and a run need not support a scientifically valid result to have consumed compute. Scientifically superseded and invalid development configurations are therefore included when their execution and timing records are complete.

Deduplication retains one record per scientific tag, using the smaller recorded duration where repeated local copies differ. The original 886-record audit accounts for 750.102222222 hours. The additional local archives contribute 84 distinct records and 58.4875 hours. Another 107 files duplicate retained tags, and 33 older records lack explicit accelerator-hour metadata. These files are not added. No missing timing field is treated as a measured zero.

Every included source JSON and its completion receipt were read locally on the audit date. Checks verify complete status, equal identity and cell fingerprint, a receipt-listed source file with matching SHA-256 and byte count, explicit hardware, and the seconds-to-hours identity. The CSV stores the source and receipt hashes; the JSON stores its own checksum of the CSV and totals by study. Original run artifacts are outside this compact ledger. Portable path aliases identify their original archive roots without personal filesystem paths.

Run `python3 verify_compute_ledger.py` to verify the shipped ledger's arithmetic, uniqueness, path form, study totals and binding. This replays the frozen accounting; it does not claim to re-read original source files when those files are absent from the current checkout.
