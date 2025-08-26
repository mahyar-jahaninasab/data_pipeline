# Mainpipe - Senior Data Engineer Take-Home

## Intro

**Role: Senior Data Engineer**

This take-home is your chance to showcase your expertise as a Senior Data Engineer. You'll be building a simplified version of a real data pipeline used in LLM pre-training: we call it 'mainpipe'. We will evaluate your submission on system architecture, code quality, performance, and the clarity of your written report, which should clearly communicate your approach and results to the Maincode team.

## Your Task

Build an end-to-end data pipeline focused on filtering & data preparation. We provide the starting point (sample of unprepared data) and the end result needs to be a dataset ready for LLM pre-training.

The following elements are **must-haves** of the submission, and a good submission will have additional elements of your choice:

1. **Data acquisition**
2. **Data cleaning, normalisation and tokenisation**
3. **Training ready exports** (i.e. tokenised, mixtures, shards, etc.)
4. **Inspectability** (e.g. histograms of length, lang scores, dup markers, PII hit-rates, drop reasons)
5. **Conceptual plan for scaling**


The brief intentionally leaves room for interpretation - your choices and rationale are an important part of the evaluation. There are existing open source pipelines for the preparation of LLM training datasets, and we encourage you to draw inspiration from them for best practices.

Keep the solution self-contained, but feel free to explain what you would do differently at real scale.
The take-home is designed to be completed in roughly four hours of focused work.

## Ground Rules

- **Language:** Python 3.10+
- **Containerised pipeline** that runs end-to-end
- **Data:** Use the dataset provided below

## Dataset Instructions

For this assignment, you'll work with a curated, multi-domain slice assembled from two sources:

- **monology/pile-uncopyrighted** (for PubMed, GitHub, Wikipedia)
- **allenai/c4** (for web text; replacement for OpenWebText)

Please prepare approximately **170 MB** of text data in total by downloading the following subsets:

1. **PubMed Abstracts** (biomedical text, semi-structured) - about **50 MB**
2. **Web crawl (C4, English configuration)** - about **50 MB**
3. **GitHub** (code data) - about **50 MB**
4. **Wikipedia (English)** - about **20 MB**

### Download Guidelines

- Use the Hugging Face dataset library, or any other method you prefer.
- For *pile-uncopyrighted*, filter records by the `meta.pile_set_name` field to extract the PubMed, GitHub, and Wikipedia subsets.
- For *C4*, use the English configuration.
- Aim for the approximate sizes listed above, exact values aren't critical.
- Save each subset locally in **JSONL format** (one JSON record per line). If you prefer an alternative (e.g., Parquet, Arrow), explain why in your report.

## Deliverables

1. **GitHub repository** with your data pipeline and README explaining how to run it
2. **A written report** summarising your work and design decisions

## How We Evaluate Your Take-Home Submission

- **Pipeline design & correctness** (containerisation, filters, dedupe, attributes, mixing)
- **Scalability & Systems thinking** (Spark/Ray configs for scale up plan, partitioning, shuffle strategy, small-file mitigation, failure modes etc.)
- **Observability & reproducibility** (logs, metrics, deterministic seeds etc.)
- **Code quality and engineering hygiene**
- **Quality of your project report**
- **Creativity**

---

## How to Submit

Send us a link to your repository along with your written report. Make sure the instructions in your README allow us to run the pipeline end-to-end without additional setup. If we like your submission, we will invite you to a 30-minute call for an in-depth discussion of your work with our technical team.

## Closing Note

We're excited to see your submission! This is your chance to show us your approach, creativity, and engineering craftsmanship. We're looking forward to reviewing your work and hope to talk to you soon.

*The Maincode Team*
