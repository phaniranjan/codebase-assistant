"""Incremental indexer tracking source revisions, file hashes, and stale chunk removal."""

from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from codebase_assistant.config import get_settings
from codebase_assistant.ingestion.chunker import LogicalChunker
from codebase_assistant.ingestion.models import (
    CodeChunk,
    FileMetadata,
    IndexManifest,
    SourceRevision,
)
from codebase_assistant.ingestion.scanner import RepositoryScanner

logger = logging.getLogger(__name__)


class IndexingReport(BaseModel):
    """Summary report of an indexing run."""

    repository_path: str
    source_revision: Optional[SourceRevision] = None
    indexed_at: str
    added_files: List[str] = Field(default_factory=list)
    modified_files: List[str] = Field(default_factory=list)
    deleted_files: List[str] = Field(default_factory=list)
    unchanged_files: List[str] = Field(default_factory=list)
    failed_files: Dict[str, str] = Field(default_factory=dict)  # rel_path -> error message
    total_files: int = 0
    total_chunks: int = 0
    rebuilt: bool = False

    def is_fresh(self) -> bool:
        """Return True if no files were added, modified, deleted, or failed."""
        return not (
            self.added_files
            or self.modified_files
            or self.deleted_files
            or self.failed_files
        )


class IncrementalIndexer:
    """Manages index persistence, change detection, and stale chunk eviction."""

    def __init__(
        self,
        repository_path: Path,
        cache_dir: Optional[Path] = None,
        scanner: Optional[RepositoryScanner] = None,
        chunker: Optional[LogicalChunker] = None,
    ) -> None:
        self.repository_path = Path(repository_path).resolve()
        settings = get_settings()
        if cache_dir is not None:
            c = Path(cache_dir)
            self.cache_dir = (self.repository_path / c).resolve() if not c.is_absolute() else c.resolve()
        else:
            self.cache_dir = (self.repository_path / settings.cache_dir).resolve()
        self.manifest_path = self.cache_dir / "manifest.json"
        self.chunks_path = self.cache_dir / "chunks.json"

        # Automatically exclude cache_dir from scanner if it resides within repo
        extra_ignores = []
        try:
            rel_cache = self.cache_dir.relative_to(self.repository_path).as_posix()
            extra_ignores.extend([rel_cache, f"{rel_cache}/**", self.cache_dir.name])
        except ValueError:
            pass

        if scanner is None:
            self.scanner = RepositoryScanner(
                self.repository_path,
                ignore_patterns=settings.default_ignore_patterns + extra_ignores,
            )
        else:
            self.scanner = scanner
            for pat in extra_ignores:
                if pat not in self.scanner.ignore_patterns:
                    self.scanner.ignore_patterns.append(pat)

        self.chunker = chunker or LogicalChunker()

    def load_manifest(self) -> Optional[IndexManifest]:
        """Load persisted manifest if it exists and matches repository identity."""
        try:
            if not self.manifest_path.is_file():
                return None
        except OSError:
            return None

        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            manifest = IndexManifest.model_validate(data)

            # Repository identity verification: avoid cross-repository index corruption
            expected_repo = self.repository_path.as_posix()
            if manifest.repository_path and manifest.repository_path != expected_repo:
                logger.warning(
                    "Repository mismatch in manifest: cached '%s' != current '%s'. Starting fresh index.",
                    manifest.repository_path,
                    expected_repo,
                )
                return None

            return manifest
        except Exception as e:
            logger.warning("Could not read index manifest at %s: %s. Starting fresh.", self.manifest_path, e)
            return None

    def load_chunks(self) -> Dict[str, CodeChunk]:
        """Load stored chunks from local cache."""
        try:
            if not self.chunks_path.is_file():
                return {}
        except OSError:
            return {}

        try:
            with open(self.chunks_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {k: CodeChunk.model_validate(v) for k, v in data.items()}
        except Exception as e:
            logger.warning("Could not read chunk store at %s: %s.", self.chunks_path, e)
            return {}

    @staticmethod
    def _atomic_write_json(target_path: Path, data: Any) -> None:
        """Write JSON data to a temp file in the same directory, then atomically replace.

        This guarantees that process interruption or crashes during disk writing
        cannot leave a partially written or corrupted cache.
        """
        target_path.parent.mkdir(parents=True, exist_ok=True)
        temp_file = target_path.with_name(f".{target_path.name}.{os.getpid()}.tmp")
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(str(temp_file), str(target_path))
        except BaseException:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass
            raise

    def save_index(self, manifest: IndexManifest, chunks: Dict[str, CodeChunk]) -> None:
        """Persist manifest and chunks to local cache directory with atomic write protection."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            os.makedirs(str(self.cache_dir), exist_ok=True)

        manifest.file_count = len(manifest.files)
        manifest.chunk_count = len(chunks)

        # 1. Atomically save chunks
        chunks_dict = {k: v.model_dump() for k, v in chunks.items()}
        self._atomic_write_json(self.chunks_path, chunks_dict)

        # 2. Atomically save manifest
        self._atomic_write_json(self.manifest_path, manifest.model_dump())

        logger.info(
            "Saved index atomically to %s (%d files, %d chunks)",
            self.cache_dir,
            manifest.file_count,
            manifest.chunk_count,
        )

    def index(self, rebuild: bool = False) -> IndexingReport:
        """Perform an incremental or full rebuild index of the target repository."""
        scanned_files, revision, _ = self.scanner.scan()
        timestamp = datetime.now(timezone.utc).isoformat()

        manifest = None if rebuild else self.load_manifest()
        chunks = {} if rebuild or manifest is None else self.load_chunks()

        repo_id = hashlib.sha256(self.repository_path.as_posix().encode("utf-8")).hexdigest()[:16]
        if manifest is None:
            manifest = IndexManifest(
                repository_path=self.repository_path.as_posix(),
                repository_id=repo_id,
                source_revision=revision,
                indexed_at=timestamp,
            )

        manifest.repository_id = repo_id
        manifest.source_revision = revision
        manifest.indexed_at = timestamp

        # Determine file diffs
        scanned_set: Set[str] = {p.as_posix() for p in scanned_files}
        manifest_set: Set[str] = set(manifest.files.keys())

        added_files: List[str] = []
        modified_files: List[str] = []
        unchanged_files: List[str] = []
        deleted_files: List[str] = sorted(list(manifest_set - scanned_set))
        failed_files: Dict[str, str] = {}

        # 1. Evict stale chunks from deleted files
        for del_path in deleted_files:
            file_meta = manifest.files.pop(del_path, None)
            if file_meta:
                for chunk_id in file_meta.chunk_ids:
                    chunks.pop(chunk_id, None)

        # 2. Process scanned files
        for rel_path in scanned_files:
            posix_path = Path(rel_path).as_posix()
            abs_path = self.repository_path / rel_path
            current_hash = self.scanner.compute_file_hash(abs_path)
            stat = abs_path.stat()
            lang = self.scanner.detect_language(rel_path) or "unknown"

            prev_meta = manifest.files.get(posix_path)

            if prev_meta is None or rebuild:
                # New or rebuild
                new_chunk_ids, err = self._index_single_file(
                    abs_path, posix_path, lang, revision.commit_sha, chunks
                )
                if err:
                    failed_files[posix_path] = err
                else:
                    added_files.append(posix_path)

                manifest.files[posix_path] = FileMetadata(
                    file_path=posix_path,
                    file_hash=current_hash,
                    file_size=stat.st_size,
                    last_modified=stat.st_mtime,
                    language=lang,
                    source_revision=revision.commit_sha,
                    chunk_ids=new_chunk_ids,
                )
            elif prev_meta.file_hash != current_hash:
                # Modified file: purge old chunks first to guarantee no stale chunks remain
                for old_id in prev_meta.chunk_ids:
                    chunks.pop(old_id, None)

                new_chunk_ids, err = self._index_single_file(
                    abs_path, posix_path, lang, revision.commit_sha, chunks
                )
                if err:
                    failed_files[posix_path] = err
                else:
                    modified_files.append(posix_path)

                manifest.files[posix_path] = FileMetadata(
                    file_path=posix_path,
                    file_hash=current_hash,
                    file_size=stat.st_size,
                    last_modified=stat.st_mtime,
                    language=lang,
                    source_revision=revision.commit_sha,
                    chunk_ids=new_chunk_ids,
                )
            else:
                unchanged_files.append(posix_path)

        # 3. Persist updated index
        self.save_index(manifest, chunks)

        return IndexingReport(
            repository_path=self.repository_path.as_posix(),
            source_revision=revision,
            indexed_at=timestamp,
            added_files=added_files,
            modified_files=modified_files,
            deleted_files=deleted_files,
            unchanged_files=unchanged_files,
            failed_files=failed_files,
            total_files=len(manifest.files),
            total_chunks=len(chunks),
            rebuilt=rebuild,
        )

    def _index_single_file(
        self,
        abs_path: Path,
        posix_path: str,
        lang: str,
        revision_sha: Optional[str],
        chunks_store: Dict[str, CodeChunk],
    ) -> Tuple[List[str], Optional[str]]:
        """Read, chunk, and store chunks for one file.

        Returns (chunk_ids, error_message).
        """
        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception as e:
            err_msg = f"ReadError: {type(e).__name__}: {e}"
            logger.error("Failed to read file %s: %s", abs_path, err_msg)
            return [], err_msg

        try:
            file_chunks = self.chunker.chunk_file(
                content=content,
                rel_file_path=posix_path,
                language=lang,
                source_revision=revision_sha,
            )
        except Exception as e:
            err_msg = f"ParseError: {type(e).__name__}: {e}"
            logger.error("Failed to chunk file %s: %s", posix_path, err_msg, exc_info=True)
            return [], err_msg

        chunk_ids: List[str] = []
        for c in file_chunks:
            chunks_store[c.chunk_id] = c
            chunk_ids.append(c.chunk_id)

        return chunk_ids, None
