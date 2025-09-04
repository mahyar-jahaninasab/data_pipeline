# # src/pipeline.interface.py

from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from logging.handlers import RotatingFileHandler
import abc
import json 
import logging 




LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | "
    "pid=%(process)d tid=%(thread)d | "
    "%(filename)s:%(lineno)d | %(message)s"
)

log_dir = Path("./logs")
log_dir.mkdir(parents=True, exist_ok=True)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(LOG_FORMAT, "%Y-%m-%d %H:%M:%S %z"))
file_handler = RotatingFileHandler(
    log_dir / "pipeline.log", maxBytes=5*1024*1024, backupCount=5
)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT, "%Y-%m-%d %H:%M:%S %z"))
logging.basicConfig(level=logging.INFO, handlers=[console_handler, file_handler])
logger = logging.getLogger("pipeline")


class BaseComponent(abc.ABC):
    def __init__(self, 
                name:str, 
                config: Dict[str, Any]):
        self.name:str = name 
        self.config:Dict[str, Any] = config.copy()
        self.logger = logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )
        self._initialized_at = datetime.now()


class BaseFileComponent(BaseComponent):
    def __init__(self, 
                name: str, 
                config: Dict[str, Any], 
                output_dir: str):

        super().__init__(name,
                         config)

        self.output_dir: Path = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def ensure_dir(self, p: Path) -> Path:
        p = Path(p)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def path_in_output(self, *parts: str) -> Path:
        return self.output_dir.joinpath(*parts)




class BaseProgressComponent(BaseFileComponent):
    def __init__(self, 
                name: str, 
                config: Dict[str, Any], 
                output_dir: str):

        super().__init__(name,config,output_dir)

        self.checkpoint_file: Optional[Path] = None
        self.progress: Dict[str, Any] = {}
    
    def _create_default_progress(self):
        return {
            "record_count": 0,
            "completed_chunks": [],
            "output_file": None,
            "last_updated": None,            
        }


    def _bind_checkpoint(self, checkpoint_file: Path) -> None:
        self.checkpoint_file = checkpoint_file
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                    self.progress = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load checkpoint: {e}. Creating new one.")
                self.progress = self._create_default_progress()
        else:
            self.progress = self._create_default_progress()
            self._save_progress()


    def _save_progress(self) -> None:
        if not self.checkpoint_file:
            logger.error("Checkpoint file not set; cannot save progress")
            return
        try:
            with open(self.checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(self.progress, f, indent=2)
        except IOError as e:
            logger.error(f"Failed to save progress: {e}")

    def save_progress(self,
                      record_count: int,
                      output_file: str = None) -> None:
        
        self.progress.update({
            "record_count": record_count,
            "output_file": output_file,
            "last_updated": datetime.now().isoformat(),
        })
        self._save_progress()



class BaseDownloader(BaseProgressComponent):
    def __init__(self, name: str, config: Dict[str, Any], output_dir: str = "./data/raw"):
        super().__init__(name,
                        config,
                        output_dir)

        dataset_type = self.config.get("dataset_type", "unknown")
        checkpoint = self.output_dir / f"{dataset_type}_progress.json"
        self._bind_checkpoint(checkpoint)
        self.metadata: Dict[str, Any] = {
            "downloader": self.name,
            "source": None,
            "config": self.config,
            "start_time": None,
            "end_time": None,
            "final_size_mb": self.config.get("size_mb", None),
            "output_file": None,
            "record_count": 0,
        }

    @abc.abstractmethod
    async def download(self) -> Path:
        """
        Implement in concrete downloaders; must return the output Path.
        """
        raise NotImplementedError

    def log_metadata(self) -> None:
        logger.info(f"[{self.name} Download finished]")
        for k, v in self.metadata.items():
            logger.info(f" {k}: {v}")

    def __repr__(self) -> str:
        return f"<Downloader {self.name} (config={self.config})>"


class BaseCleaner(BaseProgressComponent):
    def __init__(self, name: str, config: Dict[str, Any], output_dir: str = "./data/cleaned"):
        
        
        super().__init__(name, 
                        config, 
                        output_dir)

        dataset_type = self.config.get("dataset_type", "unknown")
        checkpoint = self.output_dir / f"{dataset_type}_progress.json"
        self._bind_checkpoint(checkpoint)
        self.metadata: Dict[str, Any] = {
            "downloader": self.name,
            "source": None,
            "config": self.config,
            "start_time": None,
            "end_time": None,
            "final_size_mb": self.config.get("size_mb", None),
            "output_file": None,
            "record_count": 0,
        }

    @abc.abstractmethod
    def clean_record(self, 
                    record: Dict[str, Any], 
                    domain: str) -> bool:
        """
        Return cleaned record (dict) if it passes all checks, or None if it should be dropped.
        """
        raise NotImplementedError

    def log_metadata(self) -> None:
        logger.info(f"[{self.name} Download finished]")
        for k, v in self.metadata.items():
            logger.info(f" {k}: {v}")
    def __repr__(self) -> str:
        return f"<Downloader {self.name} (config={self.config})>"
    


class PostProcessing(BaseProgressComponent):
    def __init__(self, name: str, config: Dict[str, Any], output_dir: str = "./data/tokenized"):
        super().__init__(name, config, output_dir)
        
        # Bind checkpoint for this specific post-processor
        dataset_type = self.config.get("dataset_type", "unknown")
        checkpoint = self.output_dir / f"{dataset_type}_postprocess_progress.json"
        self._bind_checkpoint(checkpoint)



        # Enhanced metadata for post-processing
        self.metadata: Dict[str, Any] = {
            "processor": self.name,
            "config": self.config,
            "start_time": None,
            "end_time": None,
            "input_file": None,
            "output_shards": [],
            "total_records": 0,
            "total_tokens": 0,
            "buckets_created": 0,
            "shards_created": 0,
            "manifest_path": None,
            "tokenizer_config": {
                "vocab_path": self.config.get("tokenizer_vocab"),
                "merges_path": self.config.get("tokenizer_merges")
            }
        }

    def _create_default_progress(self):
        """Enhanced progress tracking for tokenization pipeline"""
        return {
            "stage": "not_started",  # not_started, tokenizing, bucketing, sharding, manifest, completed
            "records_processed": 0,
            "buckets_completed": 0,
            "shards_completed": 0,
            "current_bucket": None,
            "current_shard": None,
            "output_shards": [],
            "manifest_path": None,
            "last_updated": None,
            "total_tokens_processed": 0,
        }

    @abc.abstractmethod
    async def process(self, input_path: str) -> Path:
        """
        Main processing method that implementations must override.
        Should return the path to the manifest file or main output.
        """
        raise NotImplementedError

    def update_stage_progress(self, 
                            stage: str, 
                            records_processed: int = None,
                            buckets_completed: int = None,
                            shards_completed: int = None,
                            current_bucket: int = None,
                            current_shard: int = None,
                            total_tokens: int = None) -> None:
        """Update progress tracking with detailed stage information"""
        
        updates = {"stage": stage, "last_updated": datetime.now().isoformat()}
        
        if records_processed is not None:
            updates["records_processed"] = records_processed
        if buckets_completed is not None:
            updates["buckets_completed"] = buckets_completed
        if shards_completed is not None:
            updates["shards_completed"] = shards_completed
        if current_bucket is not None:
            updates["current_bucket"] = current_bucket
        if current_shard is not None:
            updates["current_shard"] = current_shard
        if total_tokens is not None:
            updates["total_tokens_processed"] = total_tokens
            
        self.progress.update(updates)
        self._save_progress()
        
        # Log progress update
        self.logger.info(
            f"Stage: {stage} | Records: {self.progress.get('records_processed', 0)} | "
            f"Buckets: {self.progress.get('buckets_completed', 0)} | "
            f"Shards: {self.progress.get('shards_completed', 0)}"
        )

    def add_completed_shard(self, shard_path: str, shard_info: Dict[str, Any]) -> None:
        """Track completed shards for resume capability"""
        if "output_shards" not in self.progress:
            self.progress["output_shards"] = []
        
        self.progress["output_shards"].append({
            "path": shard_path,
            "info": shard_info,
            "completed_at": datetime.now().isoformat()
        })
        self._save_progress()

    def log_metadata(self) -> None:
        """Enhanced metadata logging for post-processing results"""
        self.logger.info(f"[{self.name} Post-Processing Completed]")
        for k, v in self.metadata.items():
            if k == "tokenizer_config":
                self.logger.info(f" {k}:")
                for tk, tv in v.items():
                    self.logger.info(f"   {tk}: {tv}")
            elif k == "output_shards" and isinstance(v, list):
                self.logger.info(f" {k}: {len(v)} shards created")
            else:
                self.logger.info(f" {k}: {v}")

    def get_resume_info(self) -> Dict[str, Any]:
        """Get information needed to resume processing from checkpoint"""
        return {
            "can_resume": self.progress.get("stage") not in ["completed", "not_started"],
            "last_stage": self.progress.get("stage"),
            "completed_buckets": self.progress.get("buckets_completed", 0),
            "completed_shards": self.progress.get("shards_completed", 0),
            "records_processed": self.progress.get("records_processed", 0)
        }

    def __repr__(self) -> str:
        return f"<PostProcessor {self.name} (stage={self.progress.get('stage', 'unknown')})>"
