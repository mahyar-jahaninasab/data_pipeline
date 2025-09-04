#src.managerpy

import asyncio
import os
import json
from pathlib import Path

from pipeline.stages.data_acquisition import DataDownloader
from pipeline.stages.postprocessing import TrainingExportBuilder
from pipeline.stages.interface import logger
from dotenv import load_dotenv

load_dotenv()

class DataPipelineManager:
    REQUIRED_KEYS = ["pubmed", "c4", "github", "wikipedia"]

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.datasets = self.config.get("datasets", {})
        self.post_processing = self.config.get("post_processing", {})

        self._validate_config()

    @staticmethod
    def _load_config(path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _validate_config(self):
        missing_datasets = [k for k in self.REQUIRED_KEYS if k not in self.datasets]
        if missing_datasets:
            raise KeyError(f"Missing dataset config for: {missing_datasets}")

        missing_post = [k for k in self.REQUIRED_KEYS if k not in self.post_processing]
        if missing_post:
            raise KeyError(f"Missing post_processing config for: {missing_post}")

    async def _download_all(self):
        tasks = []
        for key, cfg in self.datasets.items():
            downloader = DataDownloader(cfg)
            tasks.append(asyncio.create_task(self._run_download(key, downloader)))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for key, result in zip(self.datasets, results):
            if isinstance(result, Exception):
                raise RuntimeError(f"Download failed for '{key}': {result}")

    async def _run_download(self, name: str, downloader: DataDownloader):
        
        logger.info(f"Downloading dataset '{name}'...")
        await downloader.download()
        logger.info(f"Download complete for '{name}'.")

    async def _postprocess_all(self):
        for name in self.REQUIRED_KEYS:
            cfg = self.post_processing[name]
            output_dir = os.path.join(cfg["output_dir"], name)
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            builder = TrainingExportBuilder(
                name=name,
                config=cfg,
                output_dir=output_dir
            )
            manifest = await builder.process()
            logger.info(f"Post-processing complete for '{name}'. Manifest at: {manifest}")

    async def run(self):
        await self._download_all()
        await self._postprocess_all()


if __name__ == "__main__":
    config_path = os.getenv("CONFIG_PATH")
    if not config_path:
        raise EnvironmentError("CONFIG_PATH environment variable is not set")
    manager = DataPipelineManager(config_path)
    asyncio.run(manager.run())
