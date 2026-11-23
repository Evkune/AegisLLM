# AegisLLM — Live End-to-End Proxy Load Test Report

- **Runtime Backend:** `onnx:int8_dynamic` (`onnx_loaded=True`)
- **Policy Mode:** `enforce` (block threshold = `0.987907350063324`)
- **Peak Proxy RSS:** `697.5 MB` (Budget `< 500 MB`: **FAIL**)

| Concurrency (`C`) | Requests | Throughput (req/s) | E2E P50 (ms) | E2E P95 (ms) | E2E P99 (ms) | Short Benign P99 (ms) | Peak RSS (MB) | HTTP Statuses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | 64 | 90.9 | 12.35 | 15.71 | **16.77** | 15.33 | 693.2 | `{'200': 52, '403': 12}` |
| 4 | 64 | 168.2 | 26.97 | 33.13 | **33.70** | 33.08 | 693.3 | `{'200': 52, '403': 12}` |
| 16 | 64 | 176.4 | 91.17 | 121.88 | **143.78** | 144.13 | 697.1 | `{'200': 52, '403': 12}` |
| 32 | 128 | 171.6 | 164.78 | 297.64 | **373.91** | 379.04 | 697.5 | `{'200': 104, '403': 24}` |
