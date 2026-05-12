"""
Cache Manager for Music Generation
Task 3.5 – Implement intelligent caching system
"""

import os
import json
import time
import hashlib
import shutil
from pathlib import Path


class CacheManager:
    def __init__(
        self,
        cache_dir="backend_cache",
        expiry_seconds=3600,  # 1 hour
        max_files=50,
        max_size_mb=500
    ):
        self.cache_dir = Path(cache_dir)
        self.expiry_seconds = expiry_seconds
        self.max_files = max_files
        self.max_size_mb = max_size_mb

        self.cache_dir.mkdir(exist_ok=True)

        # stats tracking
        self.stats = {
            "hits": 0,
            "misses": 0,
            "last_eviction": None,
            "cache_size_mb": 0,
            "total_files": 0,
        }

    # -------- UTILS -------- #
    def _hash(self, prompt, params):
        raw = f"{prompt}|{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _file_path(self, key):
        return self.cache_dir / f"{key}.mp3"

    def _meta_path(self, key):
        return self.cache_dir / f"{key}.json"

    def _is_expired(self, meta):
        return (time.time() - meta.get("timestamp", 0)) > self.expiry_seconds

    def _get_cache_size_mb(self):
        total = 0
        for f in self.cache_dir.glob("*"):
            total += f.stat().st_size
        return total / (1024 * 1024)

    def _evict_if_needed(self):
        """LRU eviction — removes oldest files first"""

        all_items = list(self.cache_dir.glob("*.json"))
        total_files = len(all_items)
        total_size = self._get_cache_size_mb()

        self.stats["total_files"] = total_files
        self.stats["cache_size_mb"] = round(total_size, 2)

        if total_files <= self.max_files and total_size <= self.max_size_mb:
            return  # nothing to evict

        # sort by timestamp (oldest first)
        entries = []
        for meta_file in all_items:
            try:
                with open(meta_file, "r") as f:
                    meta = json.load(f)
                entries.append((meta.get("timestamp", 0), meta_file))
            except:
                continue

        entries.sort(key=lambda x: x[0])  # oldest first

        # delete until under limits
        for timestamp, meta_file in entries:
            key = meta_file.stem
            audio_file = self._file_path(key)

            try:
                meta_file.unlink()
            except:
                pass

            try:
                audio_file.unlink()
            except:
                pass

            # re-check size
            total_size = self._get_cache_size_mb()
            total_files -= 1

            if total_files <= self.max_files and total_size <= self.max_size_mb:
                break

        self.stats["last_eviction"] = time.time()

    # ---------- PUBLIC FUNCTIONS ---------- #
    def get_cache_key(self, prompt, params):
        return self._hash(prompt, params)

    def get(self, cache_key):
        """Return cached audio if exists and not expired"""
        meta_path = self._meta_path(cache_key)
        audio_path = self._file_path(cache_key)

        if not meta_path.exists() or not audio_path.exists():
            self.stats["misses"] += 1
            return None

        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
        except:
            self.stats["misses"] += 1
            return None

        # expiry check
        if self._is_expired(meta):
            try:
                meta_path.unlink()
                audio_path.unlink()
            except:
                pass
            self.stats["misses"] += 1
            return None

        # Cache HIT
        self.stats["hits"] += 1
        return {
            "audio_file": str(audio_path),
            "metadata": meta.get("metadata", {})
        }

    def set(self, cache_key, audio_file, metadata):
        """Store audio + metadata into cache"""
        try:
            dest_audio = self._file_path(cache_key)
            dest_meta = self._meta_path(cache_key)

            shutil.copyfile(audio_file, dest_audio)

            meta = {
                "timestamp": time.time(),
                "metadata": metadata
            }

            with open(dest_meta, "w") as f:
                json.dump(meta, f, indent=2)

            # check size limits
            self._evict_if_needed()

        except Exception as e:
            print("[CacheManager] Error writing cache:", e)

    def clear_cache(self):
        shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.stats = {k: 0 for k in self.stats}

    def export_cache(self, target="cache_export"):
        """Copy entire cache out for debugging"""
        target = Path(target)
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(self.cache_dir, target)

    def get_stats(self):
        """Return cache statistics"""
        self.stats["cache_size_mb"] = round(self._get_cache_size_mb(), 2)
        self.stats["total_files"] = len(list(self.cache_dir.glob("*.json")))
        return self.stats
