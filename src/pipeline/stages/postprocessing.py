import json
import random
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import ByteLevelBPETokenizer, Tokenizer

from pipeline.stages.interface import PostProcessing


def latest_data_file(
    directory: str,
    base_name: str,
    ignore_suffix: str = "_progress"
) -> Optional[Path]:
    """
    Return the Path to the most recently created matching file,
    or None if none found.
    """
    dir_path = Path(directory)
    pattern = f"{base_name}_*"
    candidates = [
        p for p in dir_path.glob(pattern)
        if p.is_file() and not p.stem.endswith(ignore_suffix)
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda p: p.stat().st_ctime or p.stat().st_mtime,
        reverse=True
    )
    return candidates[0]


class CheckpointManager:
    def __init__(self, ckpt_path: Path):
        self.ckpt_path = ckpt_path
        self.state: Dict[str, Any] = {}
        if ckpt_path.exists():
            self.state = json.loads(ckpt_path.read_text(encoding="utf8"))

    def save(self, key: str, value: Any):
        self.state[key] = value
        self.ckpt_path.write_text(json.dumps(self.state, indent=2), encoding="utf8")

    def load(self, key: str) -> Any:
        return self.state.get(key)

    def valid(self, fingerprint: str) -> bool:
        return self.state.get("tokenizer_fingerprint") == fingerprint


class TokenizerTrainingStage(PostProcessing):
    """
    Reads and validates JSONL input
    Splits/shuffle data into train/validation sets (90/10)
    Trains tokenizer with special tokens
    Saves vocabulary and merges files
    Creates a fingerprint for change detection
    """
    async def process(self, input_file: str) -> Tuple[Path, Path, str]:
        lines: List[str] = []
        
        for raw in Path(input_file).read_text(encoding="utf8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
                if "text" in obj and isinstance(obj["text"], str):
                    lines.append(raw)
            except json.JSONDecodeError:
                continue

        random.shuffle(lines)
        split = int(len(lines) * 0.9)
        train, val = lines[:split], lines[split:]

        tmp_dir = Path(self.output_dir) / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        train_txt = tmp_dir / "train.txt"
        val_txt = tmp_dir / "val.txt"

        train_txt.write_text(
            "\n".join(json.loads(l)["text"] for l in train),
            encoding="utf8"
        )
        val_txt.write_text(
            "\n".join(json.loads(l)["text"] for l in val),
            encoding="utf8"
        )

        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(
            files=[str(train_txt)],
            vocab_size=self.config["vocab_size"],
            min_frequency=self.config.get("min_freq", 2),
            special_tokens=["<s>", "</s>", "<pad>", "<unk>", "<mask>"]
        )

        tokenizer.save_model(str(self.output_dir), prefix="tokenizer")
        vocab_path = Path(self.output_dir) / "tokenizer-vocab.json"
        merges_path = Path(self.output_dir) / "tokenizer-merges.txt"

        combined = vocab_path.read_bytes() + merges_path.read_bytes()
        fingerprint = hashlib.sha256(combined).hexdigest()
        return vocab_path, merges_path, fingerprint


class TokenizationStage(PostProcessing):
    def __init__(
        self,
        name: str,
        config: Dict[str, Any],
        output_dir: str,
        tokenizer: Tokenizer
    ):
        super().__init__(name, config, output_dir)
        self.tokenizer = tokenizer

    """
    Normalizes text (collapses whitespace)
    Tokenizes text using pre-trained tokenizer
    Returns records with tokens, offsets, and metadata
    """
    async def process(self, input_file: str) -> List[Dict[str, Any]]:
        def normalize(text: str) -> str:
            return " ".join(text.strip().split())

        texts: List[str] = []
        metas: List[Dict[str, Any]] = []

        for line in Path(input_file).read_text(encoding="utf8").splitlines():
            obj = json.loads(line)
            texts.append(normalize(obj.get("text", "")))
            metas.append(obj.get("meta", {}))

        encodings = self.tokenizer.encode_batch(texts)
        records: List[Dict[str, Any]] = []

        for encoding, meta, text in zip(encodings, metas, texts):
            rec = {
                "text": text,
                "tokens": encoding.ids,
                "offsets": encoding.offsets,
                "meta": {**meta, "token_count": len(encoding.ids)}
            }
            records.append(rec)

        return records


class MixtureExportStage(PostProcessing):
    def process(self, manifests: List[Dict[str, Any]]) -> Path:
        mixtures: Dict[str, Any] = {}
        """
        Defines hardcoded phase weights for different data sources
        Groups shards by source and assigns weights
        Saves mixture configuration to JSON
        """
        curriculum_schedule = {
                    "phase_1": {
                    "pubmed": 0.4,
                    "c4": 0.3,
                    "github": 0.2,
                    "wikipedia": 0.1
                    },
                    "phase_2": {
                    "pubmed": 0.3,
                    "c4": 0.4,
                    "github": 0.2,
                    "wikipedia": 0.1
                    },
                    "final": {
                    "pubmed": 0.25,
                    "c4": 0.25,
                    "github": 0.25,
                    "wikipedia": 0.25
                    }
                }
        
        for phase, weights in curriculum_schedule.items():
            mixtures[phase] = []
            for src, weight in weights.items():
                shards = [m for m in manifests if m["source"] == src]
                mixtures[phase].append({
                    "source": src,
                    "weight": weight,
                    "shards": shards
                })

        out = Path(self.output_dir) / "mixtures.json"
        out.write_text(json.dumps(mixtures, indent=2), encoding="utf8")
        return out


class TrainingExportBuilder(PostProcessing):

    """
    Trains tokenizer if needed (with checkpointing)
    Tokenizes all text data
    Buckets sequences by length for efficient batching
    Creates shards of appropriate size
    Saves data to Parquet format with metadata
    Generates manifest files
    Creates curriculum mixtures
    Logs all metadata
    """
    
    def __init__(
        self,
        name: str,
        config: Dict[str, Any],
        output_dir: str
    ):
        super().__init__(name, config, output_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.input_jsonl = latest_data_file(config["reading_dir"], name)
        if self.input_jsonl is None:
            raise FileNotFoundError(
                f"No input file for {name} in {config['reading_dir']}"
            )

        self.tokenizer_stage = TokenizerTrainingStage(name, config, output_dir)
        self.vocab_path: Optional[Path] = None
        self.merges_path: Optional[Path] = None
        self.tokenizer_fp: Optional[str] = None
        self.ckpt = CheckpointManager(self.output_dir / "export_state.json")
        self.shard_size = config.get("shard_size_bytes", 100 * 1024 * 1024)
        self.length_buckets = config.get("length_buckets") or [
            (0, 128),
            (129, 256),
            (257, 512),
            (513, 1024),
            (1025, 1e9),
        ]
        self.seed = config.get("seed", 0)
        random.seed(self.seed)

    async def process(self, input_path: str = None) -> Path:
        input_file = input_path or str(self.input_jsonl)
        self.metadata["start_time"] = datetime.now().isoformat()
        self.metadata["input_file"] = input_file
        if not self.ckpt.valid(self.tokenizer_fp or ""):
            self.vocab_path, self.merges_path, self.tokenizer_fp = (
                await self.tokenizer_stage.process(input_file)
            )
            self.ckpt.save("tokenizer_fingerprint", self.tokenizer_fp)
        tokenizer = ByteLevelBPETokenizer(
            str(self.vocab_path),
            str(self.merges_path)
        )

        token_stage = TokenizationStage(
            self.name, self.config, str(self.output_dir), tokenizer
        )
        records = await token_stage.process(input_file)
        self.metadata["total_tokens"] = sum(len(r["tokens"]) for r in records)
        buckets: Dict[int, List[Dict[str, Any]]] = {
            i: [] for i in range(len(self.length_buckets))
        }
        for rec in records:
            length = rec["meta"]["token_count"]
            for idx, (mn, mx) in enumerate(self.length_buckets):
                if mn <= length <= mx:
                    buckets[idx].append(rec)
                    break
        manifests: List[Dict[str, Any]] = []
        for bidx, recs in buckets.items():
            random.Random(self.seed + bidx).shuffle(recs)
            cur_shard: List[Dict[str, Any]] = []
            cur_bytes = 0
            shards: List[List[Dict[str, Any]]] = []
            for rec in recs:
                rec_bytes = len(rec["tokens"]) * 4
                if cur_shard and (cur_bytes + rec_bytes > self.shard_size):
                    shards.append(cur_shard)
                    cur_shard, cur_bytes = [], 0
                cur_shard.append(rec)
                cur_bytes += rec_bytes
            if cur_shard:
                shards.append(cur_shard)
            if len(shards) > 1 and len(shards[-1]) < len(shards[0]) // 2:
                tail = shards.pop()
                shards[-1].extend(tail)

            for sidx, shard in enumerate(shards):
                random.Random(self.seed + bidx + sidx).shuffle(shard)
                pf = self.output_dir / f"shard_b{bidx}_s{sidx}.parquet"
                table = pa.Table.from_pylist([
                    {"text": r["text"], "tokens": r["tokens"], "meta": r["meta"]}
                    for r in shard
                ])
                pq.write_table(table, str(pf))
                tv = self.output_dir / f"shard_b{bidx}_s{sidx}.tsv"
                with open(tv, "w", encoding="utf8") as tsv:
                    tsv.write("idx\tlength\ttoken_sum\tsha256\n")
                    for i, r in enumerate(shard):
                        b = r["text"].encode("utf8")
                        h = hashlib.sha256(b).hexdigest()
                        tsv.write(f"{i}\t{len(r['tokens'])}\t{sum(r['tokens'])}\t{h}\n")
                # manifest entry
                file_sha = hashlib.sha256(Path(pf).read_bytes()).hexdigest()
                manifests.append({
                    "shard_id": pf.stem,
                    "file": str(pf),
                    "num_records": len(shard),
                    "bucket": bidx,
                    "source": self.name,
                    "file_sha256": file_sha,
                    "seed": self.seed
                })

        mix_stage = MixtureExportStage(self.name, self.config, str(self.output_dir))
        mix_path = mix_stage.process(manifests)
        mf = self.output_dir / "manifest.json"
        mf.write_text(json.dumps(manifests, indent=2), encoding="utf8")
        self.metadata.update({
            "vocab": str(self.vocab_path),
            "merges": str(self.merges_path),
            "manifest": str(mf),
            "mixtures": str(mix_path),
            "total_records": sum(m["num_records"] for m in manifests),
            "buckets": len(buckets),
            "shards": len(manifests),
            "end_time": datetime.now().isoformat()
        })
        self.log_metadata()
        self.save_progress(sum(m["num_records"] for m in manifests), str(self.output_dir))
        return mf

    def build(self) -> Path:
        import asyncio
        return asyncio.run(self.process())
