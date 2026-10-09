# AegisLLM Architecture Specification

## 1. System Overview

AegisLLM is an enterprise-grade AI Application Security Gateway designed to operate in-line as a reverse proxy protecting downstream Large Language Model (LLM) services and agentic workflows. 

The primary objective is the real-time mitigation of adversarial inputs mapped to **OWASP Top 10 for LLM Applications (LLM01: Prompt Injection and Jailbreak)**.

## 2. In-Line Interception Flow

```
                      +------------------------------------------------+
                      |                 Client Traffic                 |
                      +-----------------------+------------------------+
                                              |
                                              v (HTTP POST /v1/chat/completions)
+-----------------------------------------------------------------------------------------------+
| AegisLLM Security Gateway                                                                     |
|                                                                                               |
|  +---------------------+       +---------------------+       +-----------------------------+  |
|  | Input Normalization | ----> |  Inspection Engine  | ----> |      Decision Gateway       |  |
|  | - Zero-width strip  |       |  - Baseline Regex   |       | - Score >= 0.75: BLOCK      |  |
|  | - Unicode NFKC      |       |  - ONNX INT8 SLM    |       | - Score >= 0.45: SUSPICIOUS |  |
|  +---------------------+       +---------------------+       | - Score <  0.45: ALLOW      |  |
|                                                              +--------------+--------------+  |
+-----------------------------------------------------------------------------|-----------------+
                                                                              |
                                     +----------------------------------------+-------------------+
                                     | (ALLOW)                                                    | (BLOCK)
                                     v                                                            v
                    +--------------------------------+                          +----------------------------------+
                    | Upstream LLM Endpoint          |                          | Immediate Rejection              |
                    | (e.g. OpenAI / vLLM / Ollama)  |                          | HTTP 403 Forbidden               |
                    |                                |                          | CEF/Syslog Security Audit Log    |
                    +--------------------------------+                          +----------------------------------+
```

## 3. Requirements Specification

### Functional Requirements (FR)
- **FR-1 (Traffic Interception):** Transparently intercept all inbound requests targeting `/v1/chat/completions` and `/v1/completions`.
- **FR-2 (Adversarial Normalization):** Strip zero-width non-printable characters, decode basic obfuscation vectors, and normalize Unicode to canonical form before semantic evaluation.
- **FR-3 (Multi-Class Threat Classification):** Categorize threats into Direct Prompt Injection, System Prompt Leakage, and Jailbreak personas.
- **FR-4 (Deterministic Gating):** Enforce strict policy-based routing with explicit confidence thresholds (`ALLOW`, `SUSPICIOUS`, `BLOCK`).
- **FR-5 (Audit Trail & SIEM Logging):** Generate structured security audit events including correlation IDs, matched indicators, threat scores, and latency metrics.

### Non-Functional Requirements (NFR)
- **NFR-1 (Latency Budget):** In-line inspection P99 latency must remain strictly below **15.0 ms** on commodity multi-core CPU hardware.
- **NFR-2 (Resource Footprint):** Operational memory usage must not exceed 512 MB per worker process; zero runtime reliance on dedicated GPU infrastructure.
- **NFR-3 (Fail-Safe Behavior):** In the event of an unhandled internal exception or timeout, the gateway enters configurable fail-safe mode (default: fail-closed with alert notification).
- **NFR-4 (Stateless Scalability):** The inspection engine is fully stateless, allowing horizontal autoscaling across multi-node Kubernetes clusters.
