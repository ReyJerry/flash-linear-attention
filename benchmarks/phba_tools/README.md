# PHBA diagnostic tools

`python benchmarks/inspect_phba_gemm.py --output candidates.json` enumerates cuBLASLt candidates on the validated H20 runtime. This development tool builds its own candidate-enumeration helper; the installed PHBA backend performs no heuristic search. Inspection output is not a validated algorithm choice. Production descriptors require separate byte-equality gates and complete-call measurements, including all allocation and dispatch costs.

`python benchmarks/profile_phba.py` accepts the PHBA benchmark arguments and exports a Chrome trace with local-attention, metadata, history, merge and prefix ranges. Profiling hooks are installed only within that command and restored afterward. Use the uninstrumented training benchmark for throughput.
