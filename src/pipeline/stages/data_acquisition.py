# src/pipeline.data_acquisiton.py
from datetime import datetime
from pathlib import Path
from typing import Dict, Any,  Set, Tuple
import asyncio
import json
import os
import re
from datasets import load_dataset
import aiofiles

from pipeline.stages.interface import BaseDownloader, logger
from pipeline.stages.preprocessing import QualityControl

class DataDownloader(BaseDownloader):
    """
    The class automatically continues where you stopped using progress file, 
    concurrencey is used to increase the writing speed and to use it in a bigger system if scaling was considered
    """
    def __init__(self, config: Dict[str, Any], output_dir: str = './data/raw'):
        self.REQUIRED_KEYS = ["dataset_type", "dataset_name", "size_mb"]
        missing = [k for k in self.REQUIRED_KEYS 
                   if k not in config or config[k] is None]
        if missing:
            raise ValueError(
                f"Config is missing required keys: {missing}."
                f"Got: {config}"
            )
        final_output_dir = (
            output_dir
            if output_dir is not None
            else config.get("output_dir", "./data/raw")
        )
        self.cleaner = None
        if config["cleaning"]["clean"]:
            self.cleaner = QualityControl(f"{config.get('dataset_type')}Cleaner",
                                config=config["cleaning"],
                                )
            final_output_dir = self.cleaner.output_dir

        super().__init__(
            f"{config.get('dataset_type')}Downloader",
            config=config,
            output_dir=final_output_dir
        )



    async def download(self) -> Path:

        """
        The class automatically continues where you stopped using progress file, 
        concurrencey is used to increase the writing speed and to use it in a bigger system if scaling was considered.
        Note that in this case resuming function is note effient, if resuming is going to happen a lot in the system its better to use a different data structure
        """
        self.metadata["start_time"] = datetime.now().isoformat()
        logger.info(f"Starting download of {self.config['dataset_type']}")
        is_resuming = self.progress["record_count"] > 0 and self.progress["output_file"]
        if is_resuming:
            logger.info(f"Resuming from checkpoint: {self.progress['record_count']} records")
            out_path = Path(self.progress["output_file"])
            if out_path.exists():
                with open(out_path, 'r') as f:
                    actual_lines = sum(1 for _ in f)
                if actual_lines < self.progress["record_count"]:
                    logger.warning(f"File has fewer records ({actual_lines}) than expected ({self.progress['record_count']}). Starting from beginning.")
                    count = 0
                    out_path.unlink() 
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    out_path = self.output_dir / f"{self.config['dataset_type']}_{timestamp}.jsonl"
                else:
                    count = self.progress["record_count"]
            else:
                logger.warning("Previous output file not found. Starting from beginning.")
                count = 0
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = self.output_dir / f"{self.config['dataset_type']}_{timestamp}.jsonl"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = self.output_dir / f"{self.config['dataset_type']}_{timestamp}.jsonl"
            count = 0
        kwargs = {"name": self.config["dataset_config"]} if self.config["dataset_config"] else {}
        ds = load_dataset(self.config['dataset_name'], split="train", streaming=True, **kwargs)
        if self.config.get("filter_condition"):
            ds = ds.filter(
                lambda x: x.get("meta", {}).get("pile_set_name") == self.config.get("filter_condition")
            )
        
        size_limit = self.config.get("size_mb", 50)
        loop = asyncio.get_event_loop()
        async with aiofiles.open(out_path, "a" if is_resuming and count > 0 else "w") as f:
            size_mb = os.path.getsize(out_path) / 1024**2 if out_path.exists() else 0
            if count > 0:
                logger.info(f"Skipping {count} already downloaded records...")
                def skip_records(dataset, num_to_skip):
                    iterator = iter(dataset)
                    for _ in range(num_to_skip):
                        next(iterator)
                    return iterator
                iterator = await loop.run_in_executor(None, skip_records, ds, count)
            else:
                iterator = iter(ds)
            batch_size = 1000
            batch = []
            while size_mb < size_limit:
                try:
                    record = await loop.run_in_executor(None, next, iterator)
                    is_clean, reasons = self.cleaner.clean_record(record)
                    if is_clean:
                        if self.config.get("cleaning", {}).get("domain") == "code":
                            record['text'] = QualityControl.clean_code_text(record['text']) 
                        record['meta'] = reasons
                        record['text'] = QualityControl.normalize_text(record['text']) 
                        batch.append(record)
                    if len(batch) >= batch_size:
                        batch_text = "\n".join(json.dumps(r, ensure_ascii=False) for r in batch) + "\n"
                        await f.write(batch_text)
                        count += len(batch)
                        batch = []
                        size_mb = os.path.getsize(out_path) / 1024**2
                        logger.info(f"[{self.name}] {count} records, {size_mb:.2f} MB")
                        self.save_progress(count, str(out_path))
                        await asyncio.sleep(0)

                except StopIteration:
                    break
            if batch:
                batch_text = "\n".join(json.dumps(r, ensure_ascii=False) for r in batch) + "\n"
                await f.write(batch_text)
                count += len(batch)
        
        final_size = os.path.getsize(out_path) / (1024 * 1024)
        self.metadata.update({
            "end_time": datetime.now().isoformat(),
            "output_file": str(out_path),
            "final_size_mb": final_size,
            "record_count": count
        })
        logger.info(f"[{self.config['dataset_type']}] Completed {final_size:.2f} MB, {count} records")
        self.log_metadata()
        self.save_progress(count, str(out_path))
        return out_path

