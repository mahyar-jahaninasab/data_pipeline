# Data Pipeline Project

This repository contains the **Data Acquisition Pipeline** for curated datasets. Check the report.md for technical details.


## Setup & Run

1. Clone the repository  
   ```bash
   git clone https://github.com/mahyar-jahaninasab/data_pipeline.git
   ```

2. Change into the project directory  
   ```bash
   cd data_pipeline
   ```

3. (Optional) Open in Visual Studio Code  
   ```bash
   code .
   ```

4. Build and start the pipeline using Docker Compose  
   ```bash
   docker compose up --build
   ```

5. Verify output folders  
   - **Data**: Check the newly created `data/` directory for acquired datasets.  
   - **Logs**: Inspect the `logs/` directory for pipeline run logs.

***
