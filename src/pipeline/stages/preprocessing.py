# src/pipeline.cleaning.py

from datetime import datetime
from pathlib import Path
from typing import Dict, Any , Set, Tuple,List
import json
import os
import re
import hashlib
import unicodedata

from langdetect import detect_langs
from langdetect.lang_detect_exception import LangDetectException
import ast

from pipeline.stages.interface import  BaseCleaner

class QualityControl(BaseCleaner):
    """
    Applied while downloading the dataset to ensure the quality and text is right
    """

    def __init__(self, name: str, config: Dict[str, Any], output_dir: str = "./data/cleaned"):

        super().__init__(name, 
                        config, 
                        output_dir)

        self.config = config
        self.seen_hashes: Set[str] = set()

        self.pii_patterns = [
            re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), 
            re.compile(r'\b\d{3}\s\d{2}\s\d{4}\b'),  
            re.compile(r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b', re.I),  
            re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),  
        ]
        
        # :|
        self.profanity_patterns = [
            re.compile(r"""\b(fuck|shit|asshole|arsehole|bitch|
            bastard|cunt|dick|piss|damn|bloody|
            bugger|wanker|tosser|prick|dickhead|
            fuckwit|shithead|dumbarse|smartarse|
            drongo|galah|bogan|yobbo|derro)\b""", re.I)
        ]

        self.copyright_patterns = [
            re.compile(r'copyright', re.I),
            re.compile(r'©'),
            re.compile(r'all rights reserved', re.I),
        ]

        
    

    def detect_language_domain(self, text: str, expected_domain: str) -> Tuple[bool, Dict[str, Any]]:
        try:
            # If the collected data is actually for that domain?!, so a bigger threshold can be used if the source is unknow but since the source is known
            # we go for smaller threholds because of that.
            match expected_domain:
                ## we care about the text to  be in english
                # there are some cases in medical dataset like talking about molecular dynamics which we dont need
                # Also some code in github are not useful since they dont have any code inside them
                # we can use a samll model (dl_based approaches or maybe UMLS) to categorize the text 
                case 'medical':
                    cfg_path = self.config["vocabulary"]
                    with open(cfg_path, "r") as f:
                        content =  f.read()
                        medical_vocabularies = json.loads(content)
                    text_lower = text.lower()
                    text_clean = re.sub(r'[^a-zA-Z0-9\s-]', ' ', text_lower)
                    category_scores = {}
                    total_medical_terms = 0
                    for category, terms in medical_vocabularies.items():
                        score = 0
                        for term in terms:
                            pattern = r'\b' + re.escape(term) + r'\b'
                            matches = len(re.findall(pattern, text_clean))
                            score += matches
                            total_medical_terms += matches
                        category_scores[category] = score
                    text_length = len(text_clean.split())
                    if text_length == 0:
                        return False, {"Failed": "short_length"}

                    medical_density = total_medical_terms / text_length
                    category_score = len([score for score in category_scores.values() if score > 0])
                    is_medical = (
                        total_medical_terms >= 3 and
                        medical_density >= 0.05 and
                        category_score >= 2
                    )
                    return is_medical, {"length": text_length,"medical_density":medical_density, "total_medical_terms":total_medical_terms,"category_score": category_score}

                case 'code':
                    return self.is_code(text) 

                case _:
                    try:
                        lang_results = detect_langs(text)
                        if not lang_results:
                            return False , {"Failed": "Unknown"}
                        
                        expected_lang = self.config.get('expected_language', 'en')
                        if lang_results[0].lang != expected_lang and lang_results[0].prob < 0.9:
                            return False , {"Failed": "Not English"}
                        return True, {"language": lang_results[0].lang, "probabilty": lang_results[0].prob}
                    except LangDetectException:
                        return False , {"Failed": "Unknown"}
        except Exception:
            return False , {"Failed": "Unknown"}
    
    def apply_content_filtering(self, text: str) -> Tuple[bool, Dict[str, Any]]:
        tokens = text.split()
        total_tokens = max(len(tokens), 1)
        for pat in self.copyright_patterns:
            if pat.search(text):
                return False , {"Failed": "copyright"}

        pii_hits = sum(len(pat.findall(text)) for pat in self.pii_patterns)
        pii_density = pii_hits / total_tokens
        if pii_density > 0.01:  
            return False , {"Failed": "pii"}

        prof_hits = sum(len(pat.findall(text)) for pat in self.profanity_patterns)
        prof_density = prof_hits / total_tokens
        if prof_density > 0.01:  # >1% of tokens
            return False , {"Failed": "prof"}
        return True, {"pii_density": pii_density, "profanity_density": prof_density, "copyright": False}



    def validate_schema(self, record: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        required_fields = self.config.get('required_fields', [])
        for field in required_fields:
            if field not in record or not record[field]:
                return False , {"Failed": "Not relevent"}
        if len(record) < self.config.get('min_text_length'):
            return False , {"Failed": "min_text_length"}
        return True , {"length": len(record)}
    

    def check_exact_duplicates(self, content: str) -> Tuple[bool, Dict[str, Any]]:
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        if content_hash in self.seen_hashes:
            return False, {"Failed": "Exact_dup"}
        self.seen_hashes.add(content_hash)
        return True, {"exact_duplicate": 0}
    
    def validate_metadata(self, metadata: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        required_metadata = self.config.get('required_metadata', [])
        for field in required_metadata:
            if field not in metadata or not metadata[field]:
                return False, {"Failed": "meta_data"}
                
        if self.config.get('check_license', False):
            license = metadata.get('license', '')
            allowed_licenses = self.config.get('allowed_licenses', [])
            if license not in allowed_licenses:
                return False , {"Failed": "licenece not in allowed domains"}
        return True, {"license": False}

    def is_code(self,
        text: str,
        sample_bytes: int = 8192,       
        min_lines: int = 3,
        early_check_chars: int = 256,   
        score_threshold: float = 0.43
    ) -> Tuple[bool, Dict[str, Any]]:
        
        """
        APPLy three concepts to give fast signal to find the codes 
        first one is `indent_ratio` based on the fact that natural language rarely has consistent identaiton
        `punctuation ratio` should be relative to total size
        `keyword_ratio` that are supported in multiple languges
        Sample only first 8KB, the optimization is essential if we want scale the data collection
        Early exit check if there is no identition or no punctuation -> probably not code. 
        Min lines if the code is too short not useful
        """

        CODE_KEYWORDS = {
            "def","class","return","import","from","if","else","elif",
            "for","while","try","except","public","private","interface",
            "function","#include","namespace","let","const","var",
            "switch","case","break","continue","new","delete","template",
            "using","struct","enum","package","println","printf"
        }
        _PUNCT_REGEX = re.compile(r'[{}\(\)\[\];=<>:"\'#/\*\.,+\-_%&\|`\\]')
        _INDENT_REGEX = re.compile(r'^[ \t]{2,}')
        _TOKEN_REGEX = re.compile(r"[A-Za-z_#/\*!][A-Za-z0-9_#/\*\!\-\.]*")

        if not text:
            return False , {"Failed": "Empty"}

        buf = text[:sample_bytes]

        lines = buf.splitlines()
        nonblank_lines = sum(1 for ln in lines if ln.strip())
        if nonblank_lines < min_lines:
            return False , {"Failed": "min leng"}


        indent_lines = sum(1 for ln in lines if _INDENT_REGEX.match(ln))
        punct_count = len(_PUNCT_REGEX.findall(buf))

        early = buf[:early_check_chars]
        early_indent = sum(1 for ln in early.splitlines() if _INDENT_REGEX.match(ln))
        early_punct = len(_PUNCT_REGEX.findall(early))
        if early and (early_indent == 0 or early_punct == 0):
            return False , {"Failed": "early detection of not code block"}

        indent_ratio = indent_lines / max(nonblank_lines, 1)
        punct_ratio = punct_count / max(len(buf), 1)

        toks = _TOKEN_REGEX.findall(buf.lower())
        kw_hits = 0
        for t in toks:
            if t in CODE_KEYWORDS:
                kw_hits += 1
            elif t.startswith(("#!", "//", "/*")):
                kw_hits += 1
        keyword_ratio = kw_hits / max(len(toks), 1)

        score = 0.45 * indent_ratio + 0.30 * punct_ratio + 0.25 * keyword_ratio
        return score >= score_threshold, {"indent_ratio":indent_ratio, "punct_ratio":punct_ratio,"keyword_ratio":keyword_ratio}


    def clean_record(self, record: str) -> Tuple[bool, Dict[str, Any]]:
        
        text = record.get('text', '')
        metadata = record.get('metadata', {})
        domain = self.config["domain"]
        dup_passed, dup_meta = self.check_exact_duplicates(text)
        schema_passed, schema_meta = self.validate_schema(text)
        content_passed, content_meta = self.apply_content_filtering(text)
        lang_passed, lang_meta = self.detect_language_domain(text, domain)
        if not dup_passed:
            return False, {"failed_check": "duplicates", **dup_meta}
        if not schema_passed:
            return False, {"failed_check": "schema", **schema_meta}
        if not content_passed:
            return False, {"failed_check": "content", **content_meta}
        if not lang_passed:
            return False, {"failed_check": "language_domain", **lang_meta}

        combined_meta = {
            "domain" :  self.config["domain"],
            **dup_meta,
            **schema_meta,
            **content_meta,
            **lang_meta,
        }
        return True, combined_meta


    @staticmethod
    def clean_code_text(text: str) -> str:
        t = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
        lines = t.split("\n")
        while lines and lines[-1].strip() == "":
            lines.pop()
        return "\n".join(lines)

    @staticmethod
    def normalize_text(text: str) -> str:
        text = unicodedata.normalize('NFC', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        fancy_map = {
            '\u2018': "'", '\u2019': "'",
            '\u201C': '"', '\u201D': '"',
            '\u2013': '-', '\u2014': '-',
        }
        for fancy, ascii_eq in fancy_map.items():
            text = text.replace(fancy, ascii_eq)
        # We are doing three things here
        # Remove space before opening punctuation
        # Ensure space after punctuation if missing, 
        # Ensure space before closing punctuation if missing
        text = re.sub(r'([,\.!\?;:])([^\s\)\]\}])', r'\1 \2', text)
        text = re.sub(r'\s+([\(\[\{])', r'\1', text)
        text = re.sub(r'([^\s\(\[\{])([\,\.\!\?\;\:\)\]\}])', r'\1 \2', text)
        return text
    
