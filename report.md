# Project report

The overview of the project and conceptual plan to scale the data collection process.

## Overview
This project implements a modular, asynchronous data pipeline that streams large datasets, performs online quality control and save jsonL with the metadata of why the decision to save the line was made, trains a byte-level BPE tokenizer if needed, tokenizes text, buckets by sequence length, shards to Parquet, and emits manifests and weighted mixtures for downstream training.
The orchestration relies on Python’s asyncio to parallelize dataset downloads and post-processing, while Hugging Face Datasets streaming enables iterating over massive corpora without full downloads, and PyArrow writes columnar Parquet shards for efficient storage and training I/O.

### Workflow
- The data acquisition stream loads training splits from Hugging Face Datasets using streaming=True. It can optionally apply dataset-level filtering before records undergo quality-control checks. The process includes activity logging and the capability to resume from the point of failure.
- Post-processing trains or reuses a ByteLevelBPETokenizer, tokenizes normalized text, buckets sequences by length, writes sharded Parquet files, and generates a manifest plus an optional mixtures.json describing weighted combinations for training phases.


## Data acquisition
The data acquisition module streams large datasets with Hugging Face Datasets, applies online preprocessing and filtering per record, writes append‑friendly JSONL in batches, and persists periodic checkpoints for fault‑tolerant resume, ensuring scalable and efficient ingestion.
Concurrency and non‑blocking I/O are achieved with asyncio and aiofiles, while blocking steps (e.g., advancing iterators) are delegated to thread pool executors to keep the event loop responsive.

### Core workflow
- Stream source data via load_dataset(..., streaming=True) to iterate without full downloads or cache pressure, optionally filtering by a dataset attribute before processing.
- For each streamed record, run preprocessing checks and normalization, accumulate clean examples in memory, then flush to disk as JSON Lines in batches to amortize I/O.
- After each batch write, update a progress checkpoint to record counts and output file path, enabling recovery after interruptions without reprocessing completed records.
- Batched writes: buffers records to batch_size=1000 and writes in one joined string, minimizing syscalls and JSON serialization overhead.
-  Size-guarded loop: stops when the output file reaches the configured size_mb, preventing overshoot and wasted

### Checkpointing and resume
- The downloader maintains a JSON progress file that tracks record_count, output_file, and last_updated; on restart, it resumes from that state, validating the file’s actual line count and recovering safely if mismatched.
- This implements practical, application‑level checkpointing for streaming pipelines: save current position and outputs regularly so a failure can restart from the last consistent point, avoiding duplicate work.
- If frequent restarts are expected, deeper checkpointing (e.g., reader offsets and granular consumption metadata) can further reduce rework, similar to lake/stream frameworks’ checkpoint semantics.

### Metadata captured
- The downloader captures start/end timestamps, output path, final size (MB), and accepted record counts to provide traceability of what was ingested and written per run.
- Emitting manifest‑like metadata early in the pipeline supports SRE practices for monitoring throughput, data completeness, and diagnosing regressions in upstream data sources.

### Inline cleaning during download
- Each record is validated and cleaned with the configured QualityControl before persistence, ensuring only domain‑appropriate, normalized texts are appended to the JSONL output.
- In streaming contexts, applying map/filter‑style transforms during iteration is supported by the Datasets streaming interface and avoids post‑hoc full‑dataset passes.

### Performance strategies
- Use streaming ingestion to avoid full downloads, reduce disk usage, and start processing immediately, especially for multi‑terabyte corpora.
- Delegate blocking iterator advancement to run_in_executor to keep the asyncio event loop unblocked while reading the streaming dataset.
- Write in batches to minimize syscall overhead; buffering larger chunks significantly improves throughput versus many tiny writes.
- Use aiofiles for async file I/O so writes don’t block the loop, preserving concurrency with other I/O‑bound tasks.
- Persist append‑only JSONL to enable efficient, incremental writes and simple line‑by‑line recovery without reparsing entire files.

### Error handling and consistency
- On resume, if the output file exists but has fewer lines than recorded, the pipeline recreates the file to avoid partial record inconsistencies, prioritizing data integrity.
- If the previous output file is missing, the pipeline starts a new timestamped JSONL file and resets counters, ensuring a deterministic, auditable recovery path.

### Configuration knobs
- Required configuration includes dataset_type, dataset_name, and a size limit (MB) to cap streaming writes per run, aligning ingestion windows with storage budgets.
- Optional filter_condition supports pre‑filtering the streaming dataset, and cleaning configuration toggles inline QualityControl and domain‑specific normalization.

### Outputs produced
- Append‑only JSONL shards with normalized text and a meta field recording cleaning decisions provide a stream‑friendly, audit‑ready artifact for downstream post‑processing.
- A progress checkpoint (JSON) is stored alongside outputs, with record_count and output_file enabling deterministic restart from the last persisted point on failure.

## Preprocessing
### Purpose
The preprocessing step ensures only high‑quality, policy‑compliant text flows downstream by filtering for language/domain fit, schema and metadata integrity, PII/profanity redaction criteria, and duplicate suppression before normalization and export.
It combines lightweight, deterministic heuristics with rule‑based patterns and dictionary methods to operate efficiently during streaming ingestion at scale.

### Where it runs
QualityControl is applied online during dataset acquisition so that low‑quality or non‑compliant records are filtered before they are persisted, minimizing wasted I/O and post‑hoc remediation.
The DataDownloader invokes QualityControl.clean_record per record, returning a boolean “accept/drop” decision and a compact metadata trace of the checks applied.

### High‑level flow
- Validate exact‑duplicate status via SHA‑256 hashing and an in‑memory set to avoid repeated content.
- Validate schema/fields and minimum content length to ensure records are structurally usable.
- Apply content filters for PII, profanity, and copyright indicators using regular expressions with density thresholds.
- Detect language and domain suitability via langdetect probabilities or domain‑specific heuristics (medical vocabulary / code heuristics).

### Language and domain detection
- General language check: detect_langs returns candidate languages and probabilities; acceptance requires expected_language (default en) with probability ≥ 0.9, and exceptions are handled for short/ambiguous inputs.
- Medical domain: dictionary matching over a configurable JSON vocabulary computes total medical term hits, medical term density, and category coverage thresholds to decide domain fit.
- Code domain: a heuristic classifier estimates code‑likeness using indentation density, punctuation ratio, and multilingual code keyword hits on a sampled prefix, an approach aligned with established code–text feature separations.


`Note that` : `Sample-bounded` scan and `Early exit gate` make the process fast

### Content filtering
- PII: rule‑based detection for emails, phone numbers, and similar entities; if matched density exceeds a small threshold, the sample is rejected to reduce privacy risk.
- Profanity: a regex‑driven check flags offensive terms and obfuscated variants and rejects text when profane-token density exceeds a conservative cutoff.
- Copyright: simple indicators (e.g., “copyright”, “©”, “all rights reserved”) trigger rejection to avoid license violations in training corpora.

### Schema and metadata validation
- Schema: checks for required_fields and enforces a minimum text length to ensure that trivial or malformed samples do not pass downstream.
- Metadata: verifies required_metadata fields and, when enabled, enforces an allowed_licenses list so only permitted license strings are retained.

### Duplicate suppression
- Exact deduplication: each text is hashed with SHA‑256 and dropped if the hash has been seen, yielding O(1) average lookups and preventing over‑representation of repeated content.
- This exact matching complements later fuzzy/near‑duplicate strategies if desired, and is chosen here for speed during ingestion.

### Code detection heuristic
- Early‑exit checks operate on a small prefix to short‑circuit obviously non‑code inputs lacking indentation or structural punctuation, minimizing cost at scale.
- The final score combines indent_ratio (weight 0.45), punct_ratio (0.30), and keyword_ratio (0.25), accepting code when score ≥ 0.43, reflecting known discriminators like braces, operators, and common keywords.

### Text normalization utilities
- Unicode normalization: NFC normalization composes characters into canonical forms to reduce equivalent-variant drift across sources.
- Whitespace and control characters: collapses runs of whitespace, trims, and removes non‑printing control characters to standardize tokenization inputs.
- Punctuation and “fancy” quotes: maps curly quotes/dashes to ASCII and fixes spacing around punctuation to improve readability and downstream tokenization consistency.

### Cleaning for code text
- Code lines are normalized to Unix newlines, trailing blank lines are pruned, and NUL bytes are removed to ensure robust serialization and shard writing.
- These steps avoid cross‑platform newline inconsistencies and reduce spurious diffs or shard integrity issues.

### Configuration surface
- Domain: selects between general language detection, medical vocabulary heuristics, or code heuristics to tailor acceptance rules to the dataset’s intent.
- Required fields and metadata: enforce presence of critical keys and license policy; allowed_licenses constrains ingestion to compliant sources.
- Thresholds: density/score thresholds (e.g., PII/profanity ≤ 1%, code score ≥ 0.43) balance recall and precision and can be tuned per corpus.

### Outputs and tracing
- For accepted records, clean_record returns True and a compact meta dict aggregating signals (e.g., domain, densities, language probabilities) to aid auditability and downstream sampling.
- For rejected records, it returns False with a failed_check tag (duplicates, schema, content, or language_domain) that explains the earliest failing gate for clear diagnostics.

### Rationale and trade‑offs
- Lightweight heuristics and regexes keep the data path fast enough for streaming ingestion while addressing the most impactful failure modes (privacy, licensing, noise).
- Dictionary‑based domain checks and exact hashing provide predictable behavior and easy tuning, with the option to layer ML/near‑duplicate detection later if needed.

### What’s configurable and extensible
- Vocabularies can be swapped (e.g., MeSH/UMLS derived lists) and thresholds adjusted to tighten or relax acceptance per subdomain.
- PII/profanity detectors can be upgraded to frameworks like Presidio or custom detectors as corpus risk profiles evolve.

### Summary of guarantees
- Ensures samples match the expected language/domain, carry necessary fields, avoid obvious PII/profanity, respect licensing constraints, and are normalized for stable downstream processing.
- Provides deterministic, auditable decisions with clear rejection reasons and minimal overhead suitable for large‑scale streaming pipelines.

## Post‑processing and exports
This module implements end-to-end post-processing: it selects the latest JSONL input, trains or reuses a byte-level BPE tokenizer, tokenizes and normalizes text, buckets sequences by length, shards to Parquet, generates integrity-checked manifests, and builds curriculum-style mixtures with reproducible seeds and lightweight checkpointing.

### Inputs and prerequisites
- Input discovery uses latest_data_file(directory, base_name) to select the most recently created JSONL file, enabling consistent ingestion of append-only streams where files are timestamped or rotated.
- The pipeline expects newline-delimited JSON records with at least a text field and optional meta; JSON Lines is preferred for append-friendly streaming and recovery semantics.

### Checkpointing model
- CheckpointManager persists a small JSON state (export_state.json) holding a tokenizer_fingerprint; when the fingerprint matches, tokenizer retraining is skipped to save time and ensure determinism.
- This aligns with stream/pipeline checkpointing patterns: persist minimal state to resume or skip expensive steps safely after interruptions or configuration no-ops.

### TokenizerTrainingStage
- The stage parses the JSONL, keeps rows where text is a string, shuffles, and splits 90/10 into train/val, materializing train.txt and val.txt as sources for training and validation.
- A ByteLevelBPETokenizer is trained on train.txt with configurable vocab_size and min_frequency plus common special tokens [<s>, </s>, <pad>, <unk>, <mask>], matching standard practice for byte-level BPE tokenizers
- Byte-level subword tokenization avoids out-of-vocabulary issues and is widely used in multilingual and noisy text, providing robust coverage across scripts.
- The tokenizer is saved to tokenizer-vocab.json and tokenizer-merges.txt, which are the standard artifacts for BPE tokenizers in the Hugging Face tokenizers stack.
- A fingerprint is computed as SHA-256 over the concatenated vocab and merges bytes to detect tokenizer changes; this fingerprint drives the checkpoint’s valid() logic.

### TokenizationStage
- Text is normalized by whitespace collapsing to stabilize downstream tokenization; then encode_batch is used for efficient batch tokenization over the normalized texts.
- The stage collects ids and offsets per Encoding, returning records with tokens, offsets, and augmented meta that includes token_count for subsequent length-based grouping.
- The API used here mirrors the tokenizers guidance that encode_batch returns a list of Encoding objects with fields like ids and offsets for each input sequence.

### Length bucketing and sharding
- Records are bucketed by token_count using configurable ranges (e.g., 0–128, 129–256, …), which reduces padding and improves batching efficiency in training workloads.
- Within each bucket, records are shuffled with a deterministic Random(seed + bucket_index) to ensure reproducibility while avoiding local ordering bias.
- Shard construction packs records into shards up to shard_size_bytes by estimating rec_bytes as 4 bytes per token, then merges small tail shards when appropriate to reduce extreme size variance.
- Each shard is written as a PyArrow Table with columns text, tokens, and meta to Parquet via pyarrow.parquet.write_table, which is optimized for columnar analytics and efficient I/O during training.
- Parquet settings allow control over row groups and compression (e.g., Snappy), enabling trade-offs between size and read speed across downstream toolchains.

### TSV summaries and integrity checks
- For each shard, a parallel TSV is written with per-record index, length, token_sum, and sha256 of the original text to support quick spot checks and data validation outside Parquet readers.
- A file-level checksum (file_sha256) is computed over each Parquet file’s bytes and stored in the manifest to support integrity verification and content-addressable workflows.

### Manifest generation
- The pipeline emits a manifest.json enumerating all shards with metadata: shard_id, file path, number of records, bucket id, source label, file_sha256, and the seed used for shuffling.
- Manifests provide a lightweight index for dataset introspection and reproducible sampling, complementing Parquet’s columnar storage and metadata footprint.

### MixtureExportStage and curriculum mixtures
- The mixture stage assembles mixtures.json by grouping shards per source and assigning phase-specific weights (e.g., phase_1, phase_2, final), encoding a curriculum-like sampling schedule.
- Length- or source-aware curriculum schedules can accelerate training and stabilize optimization, and the provided phase weights illustrate a simple, declarative configuration.

### Determinism and reproducibility
- A fixed seed is applied consistently for record shuffling within buckets and across shards, ensuring stable outputs given the same inputs and configuration.
- The tokenizer checkpointing via fingerprint prevents unnecessary retraining and allows deterministic reuse of the same vocabulary/merges when the underlying text basis has not changed.

### Storage formats and rationale
- Parquet is chosen for sharded outputs to leverage columnar compression, predicate pushdown, and efficient parallel scans during training or analysis, which is superior to row-oriented formats at scale.
- JSON Lines is used for the raw input due to its append-friendly nature, simple line-based recovery, and compatibility with streaming data acquisition.

### Extensibility points
- Tokenizer configuration supports vocab_size, min_frequency, and special tokens; advanced users can build custom tokenizers or modify byte-level components per the tokenizers API.
- Shard parameters (shard_size_bytes) and length_buckets are configurable to match downstream batch sizes, sequence length targets, and storage constraints.
- Compression settings and Parquet row group sizes can be tuned for specific training hardware, data loaders, and distributed file systems.

### Operational considerations
- The latest_data_file pattern enforces predictable input selection in environments where ingestion produces rolling JSONL files, reducing ambiguity about training baselines.
- Manifest checksums, TSV summaries, and deterministic seeds provide practical guardrails for validation, regression testing, and data lineage tracking across runs.

### Outputs
- Tokenizer artifacts: tokenizer-vocab.json and tokenizer-merges.txt saved to the output directory for reproducible tokenization across stages and runs.
- Data shards: shard_b{bucket}_s{index}.parquet plus TSV summaries with record-level stats and hashes for simple verification workflows.
- Indexes: manifest.json listing shard metadata and mixtures.json specifying phase-wise sampling weights for curriculum-style training.

## Design decisions
Streaming‑first ingestion was chosen to handle web‑scale datasets immediately and avoid disk‑space bottlenecks, aligning with the Datasets library’s streaming iterator model.
Byte‑level BPE was selected for broad character coverage and simple training/saving semantics, while Parquet was chosen for columnar compression and efficient read paths during training.

## Configuration summary
Core acquisition knobs include dataset_type, dataset_name, and per‑run size caps (MB), plus optional streaming filters; preprocessing config governs domain, PII/profanity thresholds, required fields, and license policy.
Post‑processing config includes tokenizer vocab size/min frequency, length bucket ranges, and shard size in bytes to balance file counts and training I/O efficiency.

## Operations and reliability
Periodic checkpointing enables deterministic resume after failures, and conservative file integrity checks on restart prevent partial‑write corruption from propagating.
Rotating logs keep historical traces without unbounded growth, aiding diagnosis of ingest anomalies or upstream changes.

## Performance notes
Async file I/O and batching reduce syscall overhead, while streaming mode avoids full downloads and enables immediate processing, which is critical at multi‑TB scales.
JSON Lines enables incremental appends and easy recovery via line counts, making it well‑suited to streamed ingestion pipelines.

## Interface module

### Purpose
This module provides shared infrastructure for components in the pipeline, including structured logging, unified file and directory management, progress persistence via JSON checkpoints, and extensible base classes for download, cleaning, and post-processing stages.
By centralizing these cross-cutting concerns, downstream components can focus on data logic while inheriting consistent reliability, observability, and restart behavior.

### Logging setup
- The module configures a console StreamHandler and a size-based RotatingFileHandler with up to 5 MB per log file and multiple backups, preventing unbounded log growth.
- The LOG_FORMAT includes timestamps, levels, module/class context, process/thread identifiers, and source location, aligning with logging best practices for observability and incident analysis.
- Using a consistent formatter across console and file handlers yields uniform, structured logs that are easy to filter and correlate across environments and runs.

### BaseComponent
- BaseComponent standardizes naming, config copying, and logger instantiation per concrete class, enabling scoped log categories and consistent initialization metadata.
- Capturing an _initialized_at timestamp at construction allows coarse timing and lifecycle tracing for each component during orchestration and debugging.

### BaseFileComponent
- BaseFileComponent extends BaseComponent with an output_dir Path and helper methods to ensure directories exist and to construct safe output paths within a component’s sandbox.
- Centralizing directory creation avoids race conditions and ad-hoc path handling, improving repeatability across pipelines and environments.

### BaseProgressComponent
- BaseProgressComponent adds a simple JSON checkpoint mechanism with a bind/save pattern to persist progress such as record_count, output_file, and last_updated.
- On initialize, it attempts to load an existing checkpoint; if corrupted or missing, it initializes a default structure and persists it for deterministic resumes.
- This pattern implements application-level checkpointing so long-running tasks can restart after failure without reprocessing completed work.

### Checkpointing model
- Checkpointing stores minimal state needed to resume safely at consistent boundaries, which reduces recovery time and compute waste while preserving correctness.
- This aligns with common data pipeline patterns where tasks save state periodically to enable restart from the last known-good point, improving fault tolerance.

### BaseDownloader
- BaseDownloader binds a dataset-specific checkpoint file and maintains metadata fields like start/end time, final_size_mb, and record_count for auditability and run reporting.
- It defines an abstract download() to be implemented by concrete downloaders, ensuring a consistent interface and enabling orchestration across multiple sources.
- The log_metadata utility prints a normalized summary at completion, aiding SRE-style reviews and postmortems with consistent structured context.

### BaseCleaner
- BaseCleaner mirrors BaseDownloader’s checkpoint and metadata patterns but focuses on cleaning/validation operations over records, exposing an abstract clean_record interface.
- Centralizing cleaning metadata and progress allows component-level tracking of throughput, failures, and acceptance rates for data hygiene.
- This unified structure simplifies swapping different cleaning strategies while keeping observability and state management intact.

### PostProcessing
- PostProcessing extends the progress model with richer stage tracking (tokenizing, bucketing, sharding, manifest, completed) and detailed counters for records, buckets, shards, and tokens.
- The update_stage_progress method provides granular, resumable milestones and logs stage summaries, which improves transparency and debuggability in multi-phase jobs.
- add_completed_shard records shard paths and metadata to the checkpoint, supporting idempotent restarts and simplifying recovery logic after partial completion.

### Metadata conventions
- Metadata objects attach structured fields such as processor/downloader name, config, timestamps, and outputs, providing a standard shape for logs and run summaries.
- Including tokenizer_config placeholders in PostProcessing metadata improves traceability of model artifacts used downstream and supports reproducibility.

### Resume information
- get_resume_info exposes a compact summary of last_stage, completed_buckets/shards, and processed records, allowing orchestration layers to decide safe restart points.
- By surfacing resume state explicitly, operators and automated runners can design restart policies without inspecting raw checkpoint files.

### Operational considerations
- RotatingFileHandler ensures logs remain bounded in size and retained across rotations, which is essential for reliability in long-running data jobs.
- Consistent, context-rich logging formats and timestamps improve triage speed, enable structured log ingestion, and reduce mean time to recovery during incidents.
- Lightweight, JSON-based checkpointing balances simplicity and robustness, providing fault tolerance without introducing heavyweight dependencies.

### Extensibility
- New components can derive from these bases to inherit logging, file management, and checkpointing with minimal boilerplate, accelerating development and standardizing behavior.
- Additional fields can be added to metadata/progress structures as needs grow, while preserving backward compatibility through default progress creation.

## Conceptual Plan

# Scaling Data Collection for LLMs - Concise Plan

## Current Foundation
- **Streaming ingestion** via Hugging Face Datasets for multi-TB processing
- **Modular pipeline** with data acquisition → quality control → post-processing stages  
- **Async processing** using asyncio for non-blocking I/O operations
- **Checkpoint recovery** with JSON progress files for fault tolerance
- **Quality filtering** with inline PII/profanity detection and normalization

## Core Scaling Strategy

### Infrastructure Transformation
- **Containerize components** → Deploy as independent Kubernetes pods with auto-scaling
- **Event-driven messaging** → Replace direct coupling with Apache Kafka/Redis queues
- **Distributed storage** → Migrate from local JSONL to cloud object storage (S3/GCS)
- **Multi-cloud deployment** → Geographic distribution for latency reduction and vendor risk

### Microservices Architecture
- **Data ingestion service** with configurable source adapters
- **Quality control service** with pluggable validation rules  
- **Tokenization service** with model-specific processors
- **Storage orchestration service** with format-specific writers
- **Auto-scaling** based on queue depth and resource utilization

### Performance Optimization
- **Parallel processing** using Apache Spark/Dask for distributed compute
- **Intelligent caching** with Redis Cluster for metadata and tokenizer artifacts
- **Load balancing** routing by data type and system load
- **Streaming optimization** with advanced partitioning and prefetching

### Quality at Scale
- **AI-powered monitoring** for automatic anomaly and drift detection
- **Real-time feedback loops** adjusting thresholds based on model performance
- **Automated lineage tracking** for audit trails and compliance

## Implementation Phases

### Containerization
- Package components as Docker containers
- Deploy Kubernetes clusters with monitoring/logging
- Integrate Apache Kafka for inter-service communication

### Horizontal Scaling  
- Split into specialized microservices
- Migrate to distributed object storage
- Implement load testing and optimization

###  Advanced Features
- Multi-cloud federation with data synchronization
- AI-powered predictive scaling and optimization
- Real-time analytics dashboard

### Production Hardening
- Complete security audit and zero-trust implementation
- Disaster recovery procedures
- Final performance tuning



## Key Technologies
- **Orchestration**: Kubernetes with HPA/VPA
- **Messaging**: Apache Kafka, Redis
- **Storage**: S3/GCS with intelligent tiering
- **Processing**: Apache Spark/Dask
- **Monitoring**: OpenTelemetry, ELK stack
- **Security**: Zero-trust architecture, encryption at rest/transit