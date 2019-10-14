# MLPerf training logging summarization

This directory contains a set of rules for compliance checker
producing summary of hyperparameters present in the compliant log file.

## Usage
To get the summary for the compliant log, run the following command:

    python mlp_compliance.py --config hp_extraction_0.6.0/common.yaml FILENAME  # for log file of all benchmarks except for the minigo
    python mlp_compliance.py --config hp_extraction_0.6.0/minigo.yaml FILENAME  # for minigo log file

The command will print the summary in a json format to stdout.
