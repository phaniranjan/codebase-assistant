"""Command-line interface for Codebase Assistant."""

from pathlib import Path
import sys
import click

# Ensure utf-8 terminal output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    from rich.console import Console
    from rich.table import Table

    rich_available = True
    console = Console()
except ImportError:
    rich_available = False
    console = None

from codebase_assistant.config import get_settings
from codebase_assistant.embeddings.gemini import GeminiEmbeddingProvider
from codebase_assistant.embeddings.manager import EmbeddingManager
from codebase_assistant.embeddings.mock import MockEmbeddingProvider
from codebase_assistant.ingestion.indexer import IncrementalIndexer
from codebase_assistant.ingestion.scanner import RepositoryScanner
from codebase_assistant.retrieval.context_assembler import ContextAssembler
from codebase_assistant.retrieval.hybrid import HybridRetriever


def print_info(title: str, value: str = ""):
    """Print formatted info line with rich or plain fallback."""
    if rich_available and console:
        console.print(f"[bold cyan]{title}[/bold cyan] {value}")
    else:
        print(f"{title} {value}")


def print_success(title: str, value: str = ""):
    if rich_available and console:
        console.print(f"[bold green]{title}[/bold green] {value}")
    else:
        print(f"{title} {value}")


@click.group()
@click.version_option(version="0.1.0")
def main():
    """Codebase Assistant - RAG-Powered Coding Agent."""
    pass


@main.command()
@click.argument("repo_path", default=".", type=click.Path(exists=True, file_okay=False))
def scan(repo_path: str):
    """Scan a repository and display detected files, revisions, and exclusions."""
    target = Path(repo_path).resolve()
    print_info("Scanning repository:", str(target))

    scanner = RepositoryScanner(target)
    eligible_files, revision, excluded_files = scanner.scan()

    print_success("Source Revision:", f"{revision.revision_id} (branch: {revision.branch or 'none'})")
    print_success("Eligible Files:", str(len(eligible_files)))
    print_info("Excluded Files / Rules:", str(len(excluded_files)))

    if rich_available and console:
        table = Table(title="Eligible Files Sample")
        table.add_column("Relative Path", style="cyan")
        table.add_column("Language", style="magenta")

        for p in eligible_files[:20]:
            lang = scanner.detect_language(p) or "unknown"
            table.add_row(p.as_posix(), lang)

        console.print(table)
    else:
        print("\n--- Eligible Files Sample (First 20) ---")
        for p in eligible_files[:20]:
            lang = scanner.detect_language(p) or "unknown"
            print(f"  [{lang:10}] {p.as_posix()}")

    if len(eligible_files) > 20:
        print(f"... and {len(eligible_files) - 20} more files.")


@main.command()
@click.argument("repo_path", default=".", type=click.Path(exists=True, file_okay=False))
@click.option("--rebuild", is_flag=True, help="Rebuild the entire index from scratch.")
@click.option("--cache-dir", default=None, help="Directory to store the index manifest and chunks.")
def index(repo_path: str, rebuild: bool, cache_dir: str | None):
    """Index a repository incrementally or rebuild from scratch."""
    target = Path(repo_path).resolve()
    print_info("Indexing repository:", f"{target} (rebuild={rebuild})")

    cache = Path(cache_dir).resolve() if cache_dir else None
    indexer = IncrementalIndexer(target, cache_dir=cache)
    report = indexer.index(rebuild=rebuild)

    metrics = [
        ("Repository", report.repository_path),
        ("Revision", str(report.source_revision.revision_id if report.source_revision else "unknown")),
        ("Added Files", str(len(report.added_files))),
        ("Modified Files", str(len(report.modified_files))),
        ("Deleted Files", str(len(report.deleted_files))),
        ("Failed Files", str(len(report.failed_files))),
        ("Unchanged Files", str(len(report.unchanged_files))),
        ("Total Indexed Files", str(report.total_files)),
        ("Total Chunks", str(report.total_chunks)),
    ]

    if rich_available and console:
        table = Table(title="Indexing Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        for k, v in metrics:
            table.add_row(k, v)
        console.print(table)
    else:
        print("\n--- Indexing Results ---")
        for k, v in metrics:
            print(f"  {k:22}: {v}")

    if report.failed_files:
        print("\n--- Indexing Failures ---")
        for fpath, err in report.failed_files.items():
            print(f"  [ERROR] {fpath}: {err}")


@main.command(name="inspect-chunks")
@click.option("--cache-dir", default=None, help="Directory containing the index.")
@click.option("--file", "file_filter", default=None, help="Filter chunks by file path.")
@click.option("--symbol", "symbol_filter", default=None, help="Filter chunks by symbol name.")
@click.option("--limit", default=10, type=int, help="Maximum number of chunks to show.")
def inspect_chunks(cache_dir: str | None, file_filter: str | None, symbol_filter: str | None, limit: int):
    """Inspect indexed chunks stored in the local cache."""
    cache = Path(cache_dir).resolve() if cache_dir else Path(".codebase_assistant").resolve()
    indexer = IncrementalIndexer(Path("."), cache_dir=cache)
    chunks = indexer.load_chunks()

    if not chunks:
        print(f"No chunks found in cache: {cache}")
        return

    filtered = list(chunks.values())
    if file_filter:
        filtered = [c for c in filtered if file_filter.lower() in c.file_path.lower()]
    if symbol_filter:
        filtered = [c for c in filtered if c.symbol_name and symbol_filter.lower() in c.symbol_name.lower()]

    print(f"\nStored Chunks ({len(filtered)} matching, showing up to {limit}):")
    for c in filtered[:limit]:
        symbol_str = f" ({c.symbol_name})" if c.symbol_name else ""
        print(f"  [{c.chunk_type.value.upper():12}]{symbol_str} {c.file_path}:{c.start_line}-{c.end_line} [hash: {c.content_hash[:8]}]")


@main.command(name="embed")
@click.argument("repo_path", default=".", type=click.Path(exists=True))
@click.option(
    "--provider",
    "provider_type",
    default="gemini",
    type=click.Choice(["gemini", "mock"], case_sensitive=False),
    help="Embedding provider ('gemini' or 'mock').",
)
@click.option("--rebuild", is_flag=True, help="Recompute all embeddings from scratch.")
@click.option("--cache-dir", default=None, help="Custom cache directory.")
def embed(repo_path: str, provider_type: str, rebuild: bool, cache_dir: str | None):
    """Generate and synchronize vector embeddings for indexed repository chunks."""
    target_repo = Path(repo_path).resolve()
    print_info("Target Repository", str(target_repo))
    print_info("Embedding Provider", provider_type.lower())

    cache = Path(cache_dir).resolve() if cache_dir else target_repo / get_settings().cache_dir
    print_info("Cache Directory", str(cache))

    if provider_type.lower() == "mock":
        provider = MockEmbeddingProvider()
    else:
        try:
            provider = GeminiEmbeddingProvider()
        except ValueError as e:
            print(f"\n[Warning] {e}")
            print("Falling back to MockEmbeddingProvider for offline execution.")
            provider = MockEmbeddingProvider()

    manager = EmbeddingManager(target_repo, cache_dir=cache, provider=provider)
    result = manager.sync(rebuild=rebuild)

    metrics = [
        ("Total Chunks", str(result["total_chunks"])),
        ("Newly Embedded Chunks", str(result["embedded_chunks"])),
        ("Evicted Vectors", str(result["evicted_chunks"])),
        ("Vector Dimension", str(result["dimension"])),
        ("Model Used", str(result["model"])),
    ]

    if rich_available and console:
        table = Table(title="Embedding Sync Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        for k, v in metrics:
            table.add_row(k, v)
        console.print(table)
    else:
        print("\n--- Embedding Sync Results ---")
        for k, v in metrics:
            print(f"  {k:22}: {v}")


@main.command(name="search")
@click.argument("query")
@click.option("--repo-path", default=".", type=click.Path(exists=True), help="Target repository path.")
@click.option(
    "--mode",
    default="hybrid",
    type=click.Choice(["hybrid", "dense_only", "sparse_only"], case_sensitive=False),
    help="Search mode: hybrid, dense_only, or sparse_only.",
)
@click.option(
    "--provider",
    "provider_type",
    default="gemini",
    type=click.Choice(["gemini", "mock"], case_sensitive=False),
    help="Embedding provider for dense search.",
)
@click.option("--top-k", default=5, type=int, help="Number of chunks to retrieve.")
@click.option("--budget", default=8192, type=int, help="Token budget limit for context assembly.")
@click.option("--cache-dir", default=None, help="Custom cache directory.")
def search(
    query: str,
    repo_path: str,
    mode: str,
    provider_type: str,
    top_k: int,
    budget: int,
    cache_dir: str | None,
):
    """Search indexed codebase using hybrid, dense, or sparse retrieval."""
    target_repo = Path(repo_path).resolve()
    cache = Path(cache_dir).resolve() if cache_dir else target_repo / get_settings().cache_dir

    indexer = IncrementalIndexer(target_repo, cache_dir=cache)
    chunks = indexer.load_chunks()

    if not chunks:
        print(f"Error: No indexed chunks found in {cache}. Please run 'codebase-assistant index' first.")
        return

    if provider_type.lower() == "mock":
        provider = MockEmbeddingProvider()
    else:
        try:
            provider = GeminiEmbeddingProvider()
        except ValueError:
            provider = MockEmbeddingProvider()

    manager = EmbeddingManager(target_repo, cache_dir=cache, provider=provider)
    vector_store = manager.load_vector_store()

    if mode in ("hybrid", "dense_only") and vector_store.count() == 0:
        print("Notice: Vector store is empty. Generating embeddings first...")
        manager.sync()
        vector_store = manager.load_vector_store()

    retriever = HybridRetriever(
        chunks=chunks,
        vector_store=vector_store,
        embedding_provider=provider,
    )

    results = retriever.search(query=query, top_k=top_k, mode=mode.lower())

    if not results:
        print(f"No matching code chunks found for query: '{query}'")
        return

    # Print search result table
    if rich_available and console:
        table = Table(title=f"Search Results (Mode: {mode.upper()}, Top {len(results)})")
        table.add_column("Rank", style="dim", width=4)
        table.add_column("Score", style="green", width=8)
        table.add_column("File", style="cyan")
        table.add_column("Symbol", style="magenta")
        table.add_column("Lines", style="yellow")
        table.add_column("Dense Rank", width=10)
        table.add_column("Sparse Rank", width=11)

        for rank, r in enumerate(results, start=1):
            table.add_row(
                str(rank),
                f"{r.score:.4f}",
                r.chunk.file_path,
                r.chunk.symbol_name or "-",
                f"{r.chunk.start_line}-{r.chunk.end_line}",
                str(r.dense_rank or "-"),
                str(r.sparse_rank or "-"),
            )
        console.print(table)
    else:
        print(f"\n--- Search Results (Mode: {mode.upper()}, Top {len(results)}) ---")
        for rank, r in enumerate(results, start=1):
            d_rank = f"dense:{r.dense_rank}" if r.dense_rank else "dense:-"
            s_rank = f"sparse:{r.sparse_rank}" if r.sparse_rank else "sparse:-"
            sym = f" ({r.chunk.symbol_name})" if r.chunk.symbol_name else ""
            print(f"  {rank}. [{r.score:.4f}] {r.chunk.file_path}:{r.chunk.start_line}-{r.chunk.end_line}{sym} [{d_rank}, {s_rank}]")

    # Assemble context block with budget
    assembler = ContextAssembler(default_max_tokens=budget)
    assembled = assembler.assemble(results, max_tokens=budget)

    print(f"\n--- Assembled Context (Estimated {assembled.estimated_tokens}/{budget} tokens, {len(assembled.included_chunks)} included) ---")
    print(assembled.formatted_text)


if __name__ == "__main__":
    main()
