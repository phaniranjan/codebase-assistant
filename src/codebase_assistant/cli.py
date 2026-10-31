"""Command-line interface for Codebase Assistant."""

import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
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

from codebase_assistant.agent import (
    AgentConfig,
    AgentCoordinator,
    DiagnosticEngine,
    PermissionEngine,
    PolicyEngine,
    RepairWorkflowCoordinator,
    RepairWorkflowReport,
    RepairWorkflowStatus,
    ToolRegistry,
)
from codebase_assistant.config import get_settings
from codebase_assistant.embeddings.gemini import GeminiEmbeddingProvider
from codebase_assistant.embeddings.manager import EmbeddingManager
from codebase_assistant.embeddings.mock import MockEmbeddingProvider
from codebase_assistant.evaluation.dataset import GoldenDataset
from codebase_assistant.evaluation.report import EvaluationReporter
from codebase_assistant.evaluation.runner import EvaluationRunner
from codebase_assistant.generation import (
    CodeProposalGenerator,
    ProposalResult,
    TestProposalGenerator,
)
from codebase_assistant.ingestion.indexer import IncrementalIndexer
from codebase_assistant.ingestion.scanner import RepositoryScanner
from codebase_assistant.llm.client import GeminiClient
from codebase_assistant.llm.mock_client import MockGeminiClient
from codebase_assistant.llm.qa import CodebaseQAEngine
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
@click.argument("repo_path", required=False, default=None, type=click.Path(exists=True, file_okay=False))
@click.option("--repo-path", "repo_path_opt", default=None, type=click.Path(exists=True, file_okay=False), help="Target repository path.")
def scan(repo_path: str | None, repo_path_opt: str | None):
    """Scan a repository and display detected files, revisions, and exclusions."""
    target = Path(repo_path_opt or repo_path or ".").resolve()
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
@click.argument("repo_path", required=False, default=None, type=click.Path(exists=True, file_okay=False))
@click.option("--repo-path", "repo_path_opt", default=None, type=click.Path(exists=True, file_okay=False), help="Target repository path.")
@click.option("--rebuild", is_flag=True, help="Rebuild the entire index from scratch.")
@click.option("--cache-dir", default=None, help="Directory to store the index manifest and chunks.")
def index(repo_path: str | None, repo_path_opt: str | None, rebuild: bool, cache_dir: str | None):
    """Index a repository incrementally or rebuild from scratch."""
    target = Path(repo_path_opt or repo_path or ".").resolve()
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
@click.argument("repo_path", required=False, default=None, type=click.Path(exists=True, file_okay=False))
@click.option("--repo-path", "repo_path_opt", default=None, type=click.Path(exists=True, file_okay=False), help="Target repository path.")
@click.option("--cache-dir", default=None, help="Directory containing the index.")
@click.option("--file", "file_filter", default=None, help="Filter chunks by file path.")
@click.option("--symbol", "symbol_filter", default=None, help="Filter chunks by symbol name.")
@click.option("--limit", default=10, type=int, help="Maximum number of chunks to show.")
def inspect_chunks(
    repo_path: str | None,
    repo_path_opt: str | None,
    cache_dir: str | None,
    file_filter: str | None,
    symbol_filter: str | None,
    limit: int,
):
    """Inspect indexed chunks stored in the local cache."""
    target = Path(repo_path_opt or repo_path or ".").resolve()
    cache = Path(cache_dir).resolve() if cache_dir else target / get_settings().cache_dir
    indexer = IncrementalIndexer(target, cache_dir=cache)
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
@click.argument("repo_path", required=False, default=None, type=click.Path(exists=True, file_okay=False))
@click.option("--repo-path", "repo_path_opt", default=None, type=click.Path(exists=True, file_okay=False), help="Target repository path.")
@click.option(
    "--provider",
    "provider_type",
    default="gemini",
    type=click.Choice(["gemini", "mock"], case_sensitive=False),
    help="Embedding provider ('gemini' or 'mock').",
)
@click.option("--rebuild", is_flag=True, help="Recompute all embeddings from scratch.")
@click.option("--allow-fallback", is_flag=True, default=False, help="Allow falling back to mock provider if Gemini initialization fails.")
@click.option("--cache-dir", default=None, help="Custom cache directory.")
def embed(
    repo_path: str | None,
    repo_path_opt: str | None,
    provider_type: str,
    rebuild: bool,
    allow_fallback: bool,
    cache_dir: str | None,
):
    """Generate and synchronize vector embeddings for indexed repository chunks."""
    target_repo = Path(repo_path_opt or repo_path or ".").resolve()
    print_info("Target Repository", str(target_repo))
    print_info("Embedding Provider", provider_type.lower())

    cache = Path(cache_dir).resolve() if cache_dir else target_repo / get_settings().cache_dir
    print_info("Cache Directory", str(cache))

    if provider_type.lower() == "mock":
        provider = MockEmbeddingProvider()
    else:
        try:
            provider = GeminiEmbeddingProvider()
            provider._get_client()
        except Exception as e:
            if allow_fallback:
                print(f"\n[Warning] {e}")
                print("Falling back to MockEmbeddingProvider for offline execution.")
                provider = MockEmbeddingProvider()
            else:
                print(f"\n[Error] {e}", file=sys.stderr)
                print("Failed to initialize Gemini embedding provider. Set GEMINI_API_KEY or specify --provider mock or --allow-fallback.", file=sys.stderr)
                sys.exit(1)

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
@click.argument("repo_path", required=False, default=None, type=click.Path(exists=True, file_okay=False))
@click.option("--repo-path", "repo_path_opt", default=None, type=click.Path(exists=True, file_okay=False), help="Target repository path.")
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
@click.option("--allow-fallback", is_flag=True, default=False, help="Allow falling back to mock provider if Gemini initialization fails.")
@click.option("--cache-dir", default=None, help="Custom cache directory.")
def search(
    query: str,
    repo_path: str | None,
    repo_path_opt: str | None,
    mode: str,
    provider_type: str,
    top_k: int,
    budget: int,
    allow_fallback: bool,
    cache_dir: str | None,
):
    """Search indexed codebase using hybrid, dense, or sparse retrieval."""
    target_repo = Path(repo_path_opt or repo_path or ".").resolve()
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
            provider._get_client()
        except Exception as e:
            if allow_fallback:
                print(f"\n[Warning] {e}")
                print("Falling back to MockEmbeddingProvider for offline execution.")
                provider = MockEmbeddingProvider()
            else:
                print(f"\n[Error] {e}", file=sys.stderr)
                print("Failed to initialize Gemini embedding provider. Set GEMINI_API_KEY or specify --provider mock or --allow-fallback.", file=sys.stderr)
                sys.exit(1)

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


@main.command(name="ask")
@click.argument("question")
@click.argument("repo_path", required=False, default=None, type=click.Path(exists=True, file_okay=False))
@click.option("--repo-path", "repo_path_opt", default=None, type=click.Path(exists=True, file_okay=False), help="Target repository path.")
@click.option(
    "--mode",
    default="hybrid",
    type=click.Choice(["hybrid", "dense_only", "sparse_only"], case_sensitive=False),
    help="Retrieval mode.",
)
@click.option(
    "--provider",
    "provider_type",
    default="gemini",
    type=click.Choice(["gemini", "mock"], case_sensitive=False),
    help="Provider for LLM reasoning and embeddings.",
)
@click.option("--top-k", default=5, type=int, help="Number of code chunks to retrieve.")
@click.option("--temperature", default=0.2, type=float, help="Sampling temperature.")
@click.option("--budget", default=8192, type=int, help="Token budget limit for context assembly.")
@click.option("--stream/--no-stream", default=False, help="Stream the answer as it is generated.")
@click.option("--allow-fallback", is_flag=True, default=False, help="Allow falling back to mock provider if Gemini initialization fails.")
@click.option("--cache-dir", default=None, help="Custom cache directory.")
def ask(
    question: str,
    repo_path: str | None,
    repo_path_opt: str | None,
    mode: str,
    provider_type: str,
    top_k: int,
    temperature: float,
    budget: int,
    stream: bool,
    allow_fallback: bool,
    cache_dir: str | None,
):
    """Answer developer questions grounded in the indexed repository."""
    target_repo = Path(repo_path_opt or repo_path or ".").resolve()
    cache = Path(cache_dir).resolve() if cache_dir else target_repo / get_settings().cache_dir

    indexer = IncrementalIndexer(target_repo, cache_dir=cache)
    chunks = indexer.load_chunks()

    if not chunks:
        print(f"Error: No indexed chunks found in {cache}. Please run 'codebase-assistant index' first.")
        return

    # Initialize Embedding Provider
    if provider_type.lower() == "mock":
        emb_provider = MockEmbeddingProvider()
        llm_client = MockGeminiClient()
    else:
        try:
            emb_provider = GeminiEmbeddingProvider()
            llm_client = GeminiClient()
            emb_provider._get_client()
            llm_client._get_client()
        except Exception as e:
            if allow_fallback:
                print(f"[Warning] {e}")
                print("Falling back to Mock providers for offline execution.")
                emb_provider = MockEmbeddingProvider()
                llm_client = MockGeminiClient()
            else:
                print(f"\n[Error] {e}", file=sys.stderr)
                print("Failed to initialize Gemini client/provider. Set GEMINI_API_KEY or specify --provider mock or --allow-fallback.", file=sys.stderr)
                sys.exit(1)

    # Load Vector Store
    manager = EmbeddingManager(target_repo, cache_dir=cache, provider=emb_provider)
    vector_store = manager.load_vector_store()

    if mode in ("hybrid", "dense_only") and vector_store.count() == 0:
        print("Notice: Vector store is empty. Generating embeddings first...")
        manager.sync()
        vector_store = manager.load_vector_store()

    try:
        retriever = HybridRetriever(
            chunks=chunks,
            vector_store=vector_store,
            embedding_provider=emb_provider,
        )
    except ValueError as e:
        print(f"Error: {e}")
        return

    qa_engine = CodebaseQAEngine(
        retriever=retriever,
        llm_client=llm_client,
        default_budget=budget,
    )

    print_info("Model", llm_client.model_name)
    print_info("Retrieval Mode", mode.upper())
    print(f"\nQuestion: {question}\n")

    if stream:
        print("Answer: ", end="", flush=True)
        stream_res = qa_engine.ask_stream(
            query=question,
            top_k=top_k,
            mode=mode.lower(),
            temperature=temperature,
            budget=budget,
        )
        for chunk in stream_res:
            print(chunk, end="", flush=True)
        print("\n")

        resp = stream_res.get_response()
        if resp and resp.citations:
            print("--- Source Citations ---")
            for c in resp.citations:
                print(f"  • {c.format_citation()}")

        if resp:
            print(
                f"\n[Retrieved: {resp.retrieved_chunks_count} chunks, "
                f"Assembled: {resp.assembled_chunks_count} chunks, "
                f"Prompt Tokens: ~{resp.context_tokens}]"
            )
    else:
        resp = qa_engine.ask(
            query=question,
            top_k=top_k,
            mode=mode.lower(),
            temperature=temperature,
            budget=budget,
        )

        print("--- Answer ---")
        print(resp.answer)

        if resp.citations:
            print("\n--- Source Citations ---")
            for c in resp.citations:
                print(f"  • {c.format_citation()}")

        print(
            f"\n[Retrieved: {resp.retrieved_chunks_count} chunks, "
            f"Assembled: {resp.assembled_chunks_count} chunks, "
            f"Prompt Tokens: ~{resp.context_tokens}]"
        )


@main.command(name="evaluate")
@click.argument("repo_path", required=False, default=None, type=click.Path(exists=True, file_okay=False))
@click.option("--repo-path", "repo_path_opt", default=None, type=click.Path(exists=True, file_okay=False), help="Target repository path.")
@click.option("--dataset", "dataset_path", default="data/golden_dataset.json", help="Path to golden evaluation dataset JSON.")
@click.option(
    "--mode",
    default="hybrid",
    type=click.Choice(["hybrid", "dense_only", "sparse_only", "all"], case_sensitive=False),
    help="Retrieval mode to evaluate (or 'all' to compare modes).",
)
@click.option("--top-k", default=5, type=int, help="Number of chunks to retrieve per question.")
@click.option("--budget", default=8192, type=int, help="Token budget limit for context assembly.")
@click.option("--live", is_flag=True, default=False, help="Explicitly enable live Gemini evaluation (requires GEMINI_API_KEY).")
@click.option("--fail-under", is_flag=True, default=False, help="Enforce quality gates and exit with code 1 if gates fail.")
@click.option("--output-json", default="reports/eval_report.json", help="File path to save JSON evaluation report (default: reports/eval_report.json).")
@click.option("--output-markdown", default="reports/eval_report.md", help="File path to save human-readable Markdown summary report (default: reports/eval_report.md).")
@click.option("--no-report", is_flag=True, default=False, help="Disable saving evaluation report files to disk.")
@click.option("--cache-dir", default=None, help="Custom cache directory.")
def evaluate(
    repo_path: str | None,
    repo_path_opt: str | None,
    dataset_path: str,
    mode: str,
    top_k: int,
    budget: int,
    live: bool,
    fail_under: bool,
    output_json: str | None,
    output_markdown: str | None,
    no_report: bool,
    cache_dir: str | None,
):
    """Run reproducible RAG evaluation benchmark and quality gate checks."""
    target_repo = Path(repo_path_opt or repo_path or ".").resolve()

    # Read-only protection: ensure evaluation cache is never inside target repository
    if cache_dir:
        resolved_cache = Path(cache_dir).resolve()
        try:
            resolved_cache.relative_to(target_repo)
            print(
                f"\n[Error] Read-only invariant violation: Cache directory '{resolved_cache}' "
                f"is located inside the target repository '{target_repo}'. Evaluation must not write into the target repository.",
                file=sys.stderr,
            )
            sys.exit(1)
        except ValueError:
            cache = resolved_cache
    else:
        default_cache = Path(".codebase_assistant/eval_cache").resolve()
        try:
            default_cache.relative_to(target_repo)
            # CWD is inside target_repo! Safely relocate to system temp directory
            cache = Path(tempfile.gettempdir()).resolve() / "codebase_assistant" / "eval_cache"
        except ValueError:
            cache = default_cache

    # Read-only protection: ensure report output paths are not inside target repository
    final_output_json = None
    final_output_markdown = None
    if not no_report:
        if output_json:
            p_json = Path(output_json).resolve()
            try:
                p_json.relative_to(target_repo)
                p_json = Path(tempfile.gettempdir()).resolve() / "codebase_assistant" / "reports" / Path(output_json).name
            except ValueError:
                pass
            final_output_json = str(p_json)

        if output_markdown:
            p_md = Path(output_markdown).resolve()
            try:
                p_md.relative_to(target_repo)
                p_md = Path(tempfile.gettempdir()).resolve() / "codebase_assistant" / "reports" / Path(output_markdown).name
            except ValueError:
                pass
            final_output_markdown = str(p_md)

    print_info("Target Repository", str(target_repo))
    print_info("Evaluation Mode", "LIVE GEMINI" if live else "OFFLINE MOCK")
    print_info("Dataset", dataset_path)

    # 1. Load Dataset
    d_path = Path(dataset_path).resolve()
    if not d_path.is_file():
        # Check relative to target_repo or current dir
        alt_path = target_repo / dataset_path
        if alt_path.is_file():
            d_path = alt_path
        else:
            print(f"Error: Golden dataset not found at {d_path}", file=sys.stderr)
            sys.exit(1)

    try:
        dataset = GoldenDataset.load_from_file(d_path)
    except Exception as e:
        print(f"Error: Failed to load golden dataset: {e}", file=sys.stderr)
        sys.exit(1)

    print_info("Loaded Cases", f"{len(dataset.cases)} (Version: {dataset.version})")

    # 2. Initialize Runner
    try:
        runner = EvaluationRunner(
            repository_path=target_repo,
            cache_dir=cache,
            is_live=live,
        )
    except Exception as e:
        print(f"\n[Error] {e}", file=sys.stderr)
        sys.exit(1)

    # 3. Execute Evaluation (Single mode or 'all' comparison)
    modes_to_run = ["hybrid", "dense_only", "sparse_only"] if mode.lower() == "all" else [mode.lower()]
    reports = []

    for m in modes_to_run:
        print_info(f"\nRunning Evaluation ({m.upper()}, Top-{top_k})...")
        report = runner.run(
            dataset=dataset,
            mode=m,  # type: ignore
            top_k=top_k,
            budget=budget,
        )
        reports.append(report)

        # Print Gate Summary
        status_str = "[bold green]PASSED[/bold green]" if report.quality_gate.passed else "[bold red]FAILED[/bold red]"
        print(f"\nQuality Gate Status: {status_str}")

        if rich_available and console:
            # Gate details table
            gate_table = Table(title=f"Quality Gates ({m.upper()})")
            gate_table.add_column("Gate Metric", style="cyan")
            gate_table.add_column("Type", style="dim")
            gate_table.add_column("Threshold", style="yellow")
            gate_table.add_column("Actual", style="green")
            gate_table.add_column("Result", style="bold")

            for g in report.quality_gate.gate_details:
                res_color = "green" if g.passed else "red"
                res_txt = "PASS" if g.passed else "FAIL"
                thresh = f"<= {g.threshold:.2f}" if "unnecessary" in g.gate_name else f">= {g.threshold:.2f}"
                gate_table.add_row(
                    g.gate_name,
                    "Hard" if g.is_hard_gate else "Provisional",
                    thresh,
                    f"{g.actual_value:.4f}",
                    f"[{res_color}]{res_txt}[/{res_color}]",
                )
            console.print(gate_table)

            # Category table
            cat_table = Table(title=f"Category Breakdown ({m.upper()})")
            cat_table.add_column("Category", style="cyan")
            cat_table.add_column("Cases", style="dim")
            cat_table.add_column("Recall@K", style="green")
            cat_table.add_column("MRR", style="yellow")
            cat_table.add_column("Hit Rate", style="magenta")
            cat_table.add_column("Abstention Acc", style="blue")

            for cat_name, cat in report.category_summaries.items():
                rec_str = f"{cat.avg_file_recall:.4f}" if cat.avg_file_recall is not None else "N/A"
                cat_table.add_row(
                    cat_name,
                    f"{cat.evaluated_cases}/{cat.total_cases}",
                    rec_str,
                    f"{cat.avg_file_mrr:.4f}",
                    f"{cat.avg_file_hit_rate:.4f}",
                    f"{cat.abstention_accuracy:.4f}",
                )
            console.print(cat_table)
        else:
            print("\n--- Gate Details ---")
            for g in report.quality_gate.gate_details:
                res_txt = "PASS" if g.passed else "FAIL"
                print(f"  [{res_txt:4}] {g.gate_name:32}: actual={g.actual_value:.4f} (thresh={g.threshold:.2f})")

        # Save Report artifacts for the primary mode
        if final_output_json or final_output_markdown:
            EvaluationReporter.save_reports(
                report=report,
                json_path=final_output_json,
                markdown_path=final_output_markdown,
            )
            if final_output_json:
                print_info("JSON Report saved to:", str(final_output_json))
            if final_output_markdown:
                print_info("Markdown Report saved to:", str(final_output_markdown))

    # Quality Gate exit condition
    if fail_under:
        any_failed = any(not r.quality_gate.passed for r in reports)
        if any_failed:
            print("\n[Error] Quality gates failed and --fail-under was specified.", file=sys.stderr)
            sys.exit(1)


def _resolve_proposal_cache(target_repo: Path, cache_dir: str | None) -> Path:
    """Resolve and isolate the cache directory for proposal generation.

    Enforces strict read-only safety:
    1. If user-supplied --cache-dir resolves inside target_repo, rejects and exits.
    2. If no cache_dir is provided, creates an isolated cache in system temp.
    3. Copies existing indexed chunks/manifest from target repository or CWD to the isolated cache
       so that subsequent embedding generation/sync never writes into the target repository.
    """
    if cache_dir:
        resolved_cache = Path(cache_dir).resolve()
        try:
            resolved_cache.relative_to(target_repo)
            print(
                f"\n[Error] Read-only invariant violation: Cache directory '{resolved_cache}' "
                f"is located inside the target repository '{target_repo}'. Proposals must not write into the target repository.",
                file=sys.stderr,
            )
            sys.exit(1)
        except ValueError:
            return resolved_cache

    # Safe external cache in temp directory keyed by target repository path
    repo_hash = hashlib.sha256(str(target_repo).encode("utf-8")).hexdigest()[:12]
    external_cache = Path(tempfile.gettempdir()).resolve() / "codebase_assistant" / "proposal_cache" / repo_hash
    external_cache.mkdir(parents=True, exist_ok=True)

    settings = get_settings()
    repo_cache = target_repo / settings.cache_dir
    candidate_dirs = [repo_cache]

    # Also consider CWD .codebase_assistant if outside target repo
    cwd_cache = Path(".codebase_assistant").resolve()
    try:
        cwd_cache.relative_to(target_repo)
    except ValueError:
        candidate_dirs.append(cwd_cache)

    for cand_dir in candidate_dirs:
        if cand_dir.is_dir():
            for filename in ("chunks.json", "manifest.json", "vector_index.json"):
                src_file = cand_dir / filename
                dst_file = external_cache / filename
                if src_file.is_file():
                    # Copy if destination missing or source is strictly newer
                    if not dst_file.exists() or src_file.stat().st_mtime > dst_file.stat().st_mtime:
                        shutil.copy2(src_file, dst_file)
            if (external_cache / "chunks.json").is_file():
                break

    return external_cache


def _display_and_save_proposal(
    result: ProposalResult,
    target_repo: Path,
    output_patch: str | None = None,
) -> None:
    """Helper to display proposal results, AST status, diffs, and save patch if requested."""
    print_info("Task", result.task_description)
    print_info("Provider", f"{result.provider_name} ({result.model_name})")
    print_info("Retrieved Chunks", str(result.retrieved_chunks_count))
    print_info("Files Affected", str(result.total_files_affected))

    if not result.proposals:
        print("\nNo code proposals generated.")
        if not result.is_valid:
            print(f"[Error] {result.verification_summary}", file=sys.stderr)
            sys.exit(1)
        return

    for idx, prop in enumerate(result.proposals, start=1):
        action_label = prop.proposal_type.value.upper()
        print(f"\n{'='*20} Proposal #{idx}: [{action_label}] {prop.target_file} {'='*20}")

        print("\n--- Architectural Reasoning ---")
        print(prop.architectural_reasoning)

        print("\n--- AST Syntax Verification ---")
        if prop.ast_valid:
            if rich_available and console:
                console.print("[bold green]✓ AST Syntax: Valid Python[/bold green]")
            else:
                print("✓ AST Syntax: Valid Python")
        else:
            err_msg = (
                f"✗ AST Syntax Error: {'; '.join(prop.syntax_errors) if prop.syntax_errors else 'Parsing failed'}"
            )
            if rich_available and console:
                console.print(f"[bold red]{err_msg}[/bold red]")
            else:
                print(err_msg)

        print("\n--- Proposed Unified Diff ---")
        if prop.unified_diff:
            if rich_available and console:
                try:
                    from rich.syntax import Syntax

                    syntax = Syntax(prop.unified_diff, "diff", background_color="default")
                    console.print(syntax)
                except Exception:
                    print(prop.unified_diff)
            else:
                print(prop.unified_diff)
        else:
            print("[No diff generated]")

    if output_patch:
        p_patch = Path(output_patch).resolve()
        # Read-only guard: ensure output patch is outside target repo
        try:
            p_patch.relative_to(target_repo)
            print(
                f"\n[Error] Read-only invariant violation: Output patch '{p_patch}' cannot be written inside the target repository '{target_repo}'.",
                file=sys.stderr,
            )
            sys.exit(1)
        except ValueError:
            pass

        p_patch.parent.mkdir(parents=True, exist_ok=True)
        diff_payload = "\n\n".join(prop.unified_diff for prop in result.proposals if prop.unified_diff)
        p_patch.write_text(diff_payload, encoding="utf-8")
        print_success("\nSaved unified diff patch to:", str(p_patch))

    print(f"\nVerification: {result.verification_summary}")
    if not result.is_valid:
        print("[Error] One or more proposals failed deterministic verification.", file=sys.stderr)
        sys.exit(1)


@main.command(name="propose-code")
@click.argument("task")
@click.argument("repo_path", required=False, default=None, type=click.Path(exists=True, file_okay=False))
@click.option("--repo-path", "repo_path_opt", default=None, type=click.Path(exists=True, file_okay=False), help="Target repository path.")
@click.option("--file", "target_file", default=None, help="Target file relative path to modify or create.")
@click.option(
    "--mode",
    default="hybrid",
    type=click.Choice(["hybrid", "dense_only", "sparse_only"], case_sensitive=False),
    help="Retrieval mode.",
)
@click.option(
    "--provider",
    "provider_type",
    default="gemini",
    type=click.Choice(["gemini", "mock"], case_sensitive=False),
    help="Provider for LLM reasoning and embeddings.",
)
@click.option("--top-k", default=5, type=int, help="Number of code chunks to retrieve.")
@click.option("--budget", default=8192, type=int, help="Token budget limit for context assembly.")
@click.option("--allow-fallback", is_flag=True, default=False, help="Allow falling back to mock provider if Gemini initialization fails.")
@click.option("--cache-dir", default=None, help="Custom cache directory.")
@click.option("--output-patch", default=None, help="Optional external file path to save the generated unified diff patch.")
def propose_code(
    task: str,
    repo_path: str | None,
    repo_path_opt: str | None,
    target_file: str | None,
    mode: str,
    provider_type: str,
    top_k: int,
    budget: int,
    allow_fallback: bool,
    cache_dir: str | None,
    output_patch: str | None,
):
    """Generate a context-aware unified diff proposal for a developer code task."""
    target_repo = Path(repo_path_opt or repo_path or ".").resolve()

    if output_patch:
        p_patch = Path(output_patch).resolve()
        try:
            p_patch.relative_to(target_repo)
            print(
                f"\n[Error] Read-only invariant violation: Output patch '{p_patch}' cannot be written inside the target repository '{target_repo}'.",
                file=sys.stderr,
            )
            sys.exit(1)
        except ValueError:
            pass

    cache = _resolve_proposal_cache(target_repo, cache_dir)

    indexer = IncrementalIndexer(target_repo, cache_dir=cache)
    chunks = indexer.load_chunks()

    if not chunks:
        # If chunks not found in isolated cache, attempt indexing directly into isolated cache
        indexer.index()
        chunks = indexer.load_chunks()

    if not chunks:
        print(f"Error: No indexed chunks found in {cache}. Please run 'codebase-assistant index' first.", file=sys.stderr)
        sys.exit(1)

    if provider_type.lower() == "mock":
        emb_provider = MockEmbeddingProvider()
        llm_client = MockGeminiClient()
    else:
        try:
            emb_provider = GeminiEmbeddingProvider()
            llm_client = GeminiClient()
            emb_provider._get_client()
            llm_client._get_client()
        except Exception as e:
            if allow_fallback:
                print(f"[Warning] {e}")
                print("Falling back to Mock providers for offline execution.")
                emb_provider = MockEmbeddingProvider()
                llm_client = MockGeminiClient()
            else:
                print(f"\n[Error] {e}", file=sys.stderr)
                print("Failed to initialize Gemini client/provider. Set GEMINI_API_KEY or specify --provider mock or --allow-fallback.", file=sys.stderr)
                sys.exit(1)

    manager = EmbeddingManager(target_repo, cache_dir=cache, provider=emb_provider)
    vector_store = manager.load_vector_store()

    if mode in ("hybrid", "dense_only") and vector_store.count() == 0:
        print("Notice: Vector store is empty. Generating embeddings first...")
        manager.sync()
        vector_store = manager.load_vector_store()

    try:
        retriever = HybridRetriever(
            chunks=chunks,
            vector_store=vector_store,
            embedding_provider=emb_provider,
        )
    except ValueError as e:
        print(f"Error initializing retriever: {e}", file=sys.stderr)
        sys.exit(1)

    generator = CodeProposalGenerator(
        repository_path=target_repo,
        retriever=retriever,
        llm_client=llm_client,
    )

    result = generator.propose(
        task=task,
        target_file=target_file,
        top_k=top_k,
        budget=budget,
    )

    _display_and_save_proposal(
        result=result,
        target_repo=target_repo,
        output_patch=output_patch,
    )


@main.command(name="propose-tests")
@click.argument("target_file")
@click.argument("repo_path", required=False, default=None, type=click.Path(exists=True, file_okay=False))
@click.option("--repo-path", "repo_path_opt", default=None, type=click.Path(exists=True, file_okay=False), help="Target repository path.")
@click.option(
    "--framework",
    default=None,
    type=click.Choice(["pytest", "unittest"], case_sensitive=False),
    help="Testing framework override (default: auto-detect).",
)
@click.option(
    "--mode",
    default="hybrid",
    type=click.Choice(["hybrid", "dense_only", "sparse_only"], case_sensitive=False),
    help="Retrieval mode.",
)
@click.option(
    "--provider",
    "provider_type",
    default="gemini",
    type=click.Choice(["gemini", "mock"], case_sensitive=False),
    help="Provider for LLM reasoning and embeddings.",
)
@click.option("--top-k", default=5, type=int, help="Number of code chunks to retrieve.")
@click.option("--budget", default=8192, type=int, help="Token budget limit for context assembly.")
@click.option("--allow-fallback", is_flag=True, default=False, help="Allow falling back to mock provider if Gemini initialization fails.")
@click.option("--cache-dir", default=None, help="Custom cache directory.")
@click.option("--output-patch", default=None, help="Optional external file path to save the generated unified diff patch.")
def propose_tests(
    target_file: str,
    repo_path: str | None,
    repo_path_opt: str | None,
    framework: str | None,
    mode: str,
    provider_type: str,
    top_k: int,
    budget: int,
    allow_fallback: bool,
    cache_dir: str | None,
    output_patch: str | None,
):
    """Generate a targeted unit test proposal in unified diff format for a source file."""
    target_repo = Path(repo_path_opt or repo_path or ".").resolve()

    if output_patch:
        p_patch = Path(output_patch).resolve()
        try:
            p_patch.relative_to(target_repo)
            print(
                f"\n[Error] Read-only invariant violation: Output patch '{p_patch}' cannot be written inside the target repository '{target_repo}'.",
                file=sys.stderr,
            )
            sys.exit(1)
        except ValueError:
            pass

    cache = _resolve_proposal_cache(target_repo, cache_dir)

    indexer = IncrementalIndexer(target_repo, cache_dir=cache)
    chunks = indexer.load_chunks()

    if not chunks:
        # If chunks not found in isolated cache, attempt indexing directly into isolated cache
        indexer.index()
        chunks = indexer.load_chunks()

    if not chunks:
        print(f"Error: No indexed chunks found in {cache}. Please run 'codebase-assistant index' first.", file=sys.stderr)
        sys.exit(1)

    if provider_type.lower() == "mock":
        emb_provider = MockEmbeddingProvider()
        llm_client = MockGeminiClient()
    else:
        try:
            emb_provider = GeminiEmbeddingProvider()
            llm_client = GeminiClient()
            emb_provider._get_client()
            llm_client._get_client()
        except Exception as e:
            if allow_fallback:
                print(f"[Warning] {e}")
                print("Falling back to Mock providers for offline execution.")
                emb_provider = MockEmbeddingProvider()
                llm_client = MockGeminiClient()
            else:
                print(f"\n[Error] {e}", file=sys.stderr)
                print("Failed to initialize Gemini client/provider. Set GEMINI_API_KEY or specify --provider mock or --allow-fallback.", file=sys.stderr)
                sys.exit(1)

    manager = EmbeddingManager(target_repo, cache_dir=cache, provider=emb_provider)
    vector_store = manager.load_vector_store()

    if mode in ("hybrid", "dense_only") and vector_store.count() == 0:
        print("Notice: Vector store is empty. Generating embeddings first...")
        manager.sync()
        vector_store = manager.load_vector_store()

    try:
        retriever = HybridRetriever(
            chunks=chunks,
            vector_store=vector_store,
            embedding_provider=emb_provider,
        )
    except ValueError as e:
        print(f"Error initializing retriever: {e}", file=sys.stderr)
        sys.exit(1)

    generator = TestProposalGenerator(
        repository_path=target_repo,
        retriever=retriever,
        llm_client=llm_client,
    )

    result = generator.propose_tests(
        source_file=target_file,
        framework=framework,
        top_k=top_k,
        budget=budget,
    )

    _display_and_save_proposal(
        result=result,
        target_repo=target_repo,
        output_patch=output_patch,
    )


@main.command(name="agent")
@click.argument("task")
@click.argument("repo_path", required=False, default=None, type=click.Path(exists=True, file_okay=False))
@click.option("--repo-path", "repo_path_opt", default=None, type=click.Path(exists=True, file_okay=False), help="Target repository path.")
@click.option(
    "--provider",
    "provider_type",
    default="gemini",
    type=click.Choice(["gemini", "mock"], case_sensitive=False),
    help="Provider for agent reasoning and embeddings.",
)
@click.option("--allow-fallback", is_flag=True, default=False, help="Allow falling back to mock provider if Gemini initialization fails.")
@click.option("--auto-approve", is_flag=True, default=False, help="Auto-approve consequential tool actions (e.g. running tests, applying patches).")
@click.option("--max-iterations", default=5, type=int, help="Maximum tool executions per turn (default: 5).")
@click.option("--max-steps", default=10, type=int, help="Maximum reasoning steps (default: 10).")
@click.option("--timeout", default=30.0, type=float, help="Command execution timeout in seconds (default: 30.0s).")
@click.option("--budget", default=8192, type=int, help="Token context budget limit (default: 8192).")
@click.option("--cache-dir", default=None, help="Custom cache directory.")
def agent(
    task: str,
    repo_path: str | None,
    repo_path_opt: str | None,
    provider_type: str,
    allow_fallback: bool,
    auto_approve: bool,
    max_iterations: int,
    max_steps: int,
    timeout: float,
    budget: int,
    cache_dir: str | None,
):
    """Execute an autonomous coding agent loop with controlled tools and safety sandbox."""
    target_repo = Path(repo_path_opt or repo_path or ".").resolve()
    print_info("Target Repository", str(target_repo))
    print_info("Agent Task", task)
    print_info("Consequential Mode", "Auto-Approve" if auto_approve else "Interactive Approval")

    cache = _resolve_proposal_cache(target_repo, cache_dir)

    indexer = IncrementalIndexer(target_repo, cache_dir=cache)
    chunks = indexer.load_chunks()
    if not chunks:
        indexer.index()
        chunks = indexer.load_chunks()

    if not chunks:
        print(f"Error: No indexed chunks found in {cache}. Please run 'codebase-assistant index' first.", file=sys.stderr)
        sys.exit(1)

    if provider_type.lower() == "mock":
        emb_provider = MockEmbeddingProvider()
        llm_client = MockGeminiClient()
    else:
        try:
            emb_provider = GeminiEmbeddingProvider()
            llm_client = GeminiClient()
            emb_provider._get_client()
            llm_client._get_client()
        except Exception as e:
            if allow_fallback:
                print(f"[Warning] {e}")
                print("Falling back to Mock providers for offline execution.")
                emb_provider = MockEmbeddingProvider()
                llm_client = MockGeminiClient()
            else:
                print(f"\n[Error] {e}", file=sys.stderr)
                print("Failed to initialize Gemini client/provider. Set GEMINI_API_KEY or specify --provider mock or --allow-fallback.", file=sys.stderr)
                sys.exit(1)

    manager = EmbeddingManager(target_repo, cache_dir=cache, provider=emb_provider)
    vector_store = manager.load_vector_store()
    if vector_store.count() == 0:
        print("Notice: Vector store is empty. Generating embeddings first...")
        manager.sync()
        vector_store = manager.load_vector_store()

    try:
        retriever = HybridRetriever(
            chunks=chunks,
            vector_store=vector_store,
            embedding_provider=emb_provider,
        )
    except ValueError as e:
        print(f"Error initializing retriever: {e}", file=sys.stderr)
        sys.exit(1)

    code_gen = CodeProposalGenerator(
        repository_path=target_repo,
        retriever=retriever,
        llm_client=llm_client,
    )
    test_gen = TestProposalGenerator(
        repository_path=target_repo,
        retriever=retriever,
        llm_client=llm_client,
    )

    policy_engine = PolicyEngine(target_repo)
    perm_engine = PermissionEngine(auto_approve_consequential=auto_approve)

    tool_registry = ToolRegistry(
        repository_path=target_repo,
        retriever=retriever,
        code_generator=code_gen,
        test_generator=test_gen,
        policy_engine=policy_engine,
        permission_engine=perm_engine,
        command_timeout_seconds=timeout,
    )

    config = AgentConfig(
        max_iterations=max_iterations,
        max_reasoning_steps=max_steps,
        command_timeout_seconds=timeout,
        token_budget=budget,
        auto_approve_consequential=auto_approve,
    )

    coordinator = AgentCoordinator(
        tool_registry=tool_registry,
        llm_client=llm_client,
        config=config,
    )

    print("\n--- Executing Agent Loop ---")
    response = coordinator.run(task)

    # Display steps
    for step in response.steps:
        print(f"\n[Step {step.step_number}]")
        if step.thought:
            print(f"Thought: {step.thought}")
        if step.tool_call:
            action_desc = f"{step.tool_call.tool_name}({json.dumps(step.tool_call.arguments)})"
            if rich_available and console:
                console.print(f"[bold cyan]Action:[/bold cyan] {action_desc}")
            else:
                print(f"Action: {action_desc}")
        if step.tool_result:
            res = step.tool_result
            if res.success:
                print(f"Observation:\n{res.output}")
            else:
                err_text = f"ERROR: {res.error}"
                if rich_available and console:
                    console.print(f"[bold red]Observation Error:[/bold red] {err_text}")
                else:
                    print(f"Observation Error: {err_text}")

    print("\n" + "=" * 60)
    print_success("Stopping Reason:", response.stopping_reason.value.upper())
    print_info("Total Steps:", str(response.total_steps))
    print_info("Total Tool Calls:", str(response.total_tool_calls))
    print_info("Duration:", f"{response.total_duration_seconds:.2f}s")

    print("\n--- Final Answer ---")
    print(response.final_answer)


@main.command()
@click.argument("task", required=False)
@click.option("--task", "-t", "task_opt", help="Task description or bug report to repair.")
@click.option("--repo", "-r", "repo_path_opt", help="Target repository directory.")
@click.option("--target-file", "-f", help="Target source file to modify.")
@click.option("--test-command", "-c", help="Test command to run for verification (e.g. 'pytest tests/test_calc.py').")
@click.option("--max-retries", "-m", default=3, type=int, help="Maximum bounded repair attempts (default: 3).")
@click.option("--provider", "-p", "provider_type", default="gemini", type=click.Choice(["gemini", "mock"], case_sensitive=False), help="LLM and embedding provider.")
@click.option("--allow-fallback/--no-fallback", default=True, help="Fall back to mock if Gemini API fails.")
@click.option("--auto-approve", is_flag=True, default=False, help="Auto-approve consequential actions (applying patch, running tests).")
@click.option("--timeout", default=30.0, type=float, help="Command execution timeout in seconds (default: 30.0s).")
@click.option("--cache-dir", default=None, help="Custom cache directory.")
@click.option("--report-file", type=click.Path(dir_okay=False), help="Path to write the markdown repair report.")
@click.option("--json", "json_output", is_flag=True, default=False, help="Output machine-readable JSON report.")
def repair(
    task: str | None,
    task_opt: str | None,
    repo_path_opt: str | None,
    target_file: str | None,
    test_command: str | None,
    max_retries: int,
    provider_type: str,
    allow_fallback: bool,
    auto_approve: bool,
    timeout: float,
    cache_dir: str | None,
    report_file: str | None,
    json_output: bool,
):
    """Execute an agentic repair workflow: plan, retrieve, propose, apply, test, diagnose, and self-repair."""
    task_str = task_opt or task
    if not task_str:
        print("Error: Task description is required. Provide it as an argument or with --task.", file=sys.stderr)
        sys.exit(1)

    target_repo = Path(repo_path_opt or ".").resolve()
    if not json_output:
        print_info("Target Repository", str(target_repo))
        print_info("Repair Task", task_str)
        print_info("Max Retries", str(max_retries))
        print_info("Consequential Mode", "Auto-Approve" if auto_approve else "Interactive Approval")

    cache = _resolve_proposal_cache(target_repo, cache_dir)

    indexer = IncrementalIndexer(target_repo, cache_dir=cache)
    chunks = indexer.load_chunks()
    if not chunks:
        indexer.index()
        chunks = indexer.load_chunks()

    if not chunks:
        print(f"Error: No indexed chunks found in {cache}. Please run 'codebase-assistant index' first.", file=sys.stderr)
        sys.exit(1)

    if provider_type.lower() == "mock":
        emb_provider = MockEmbeddingProvider()
        llm_client = MockGeminiClient()
    else:
        try:
            emb_provider = GeminiEmbeddingProvider()
            llm_client = GeminiClient()
            emb_provider._get_client()
            llm_client._get_client()
        except Exception as e:
            if allow_fallback:
                if not json_output:
                    print(f"[Warning] {e}")
                    print("Falling back to Mock providers for offline execution.")
                emb_provider = MockEmbeddingProvider()
                llm_client = MockGeminiClient()
            else:
                print(f"\n[Error] {e}", file=sys.stderr)
                print("Failed to initialize Gemini client/provider. Set GEMINI_API_KEY or specify --provider mock or --allow-fallback.", file=sys.stderr)
                sys.exit(1)

    manager = EmbeddingManager(target_repo, cache_dir=cache, provider=emb_provider)
    vector_store = manager.load_vector_store()
    if vector_store.count() == 0:
        if not json_output:
            print("Notice: Vector store is empty. Generating embeddings first...")
        manager.sync()
        vector_store = manager.load_vector_store()

    try:
        retriever = HybridRetriever(
            chunks=chunks,
            vector_store=vector_store,
            embedding_provider=emb_provider,
        )
    except ValueError as e:
        print(f"Error initializing retriever: {e}", file=sys.stderr)
        sys.exit(1)

    code_gen = CodeProposalGenerator(
        repository_path=target_repo,
        retriever=retriever,
        llm_client=llm_client,
    )
    test_gen = TestProposalGenerator(
        repository_path=target_repo,
        retriever=retriever,
        llm_client=llm_client,
    )

    policy_engine = PolicyEngine(target_repo)
    perm_engine = PermissionEngine(auto_approve_consequential=auto_approve)

    tool_registry = ToolRegistry(
        repository_path=target_repo,
        retriever=retriever,
        code_generator=code_gen,
        test_generator=test_gen,
        policy_engine=policy_engine,
        permission_engine=perm_engine,
        command_timeout_seconds=timeout,
    )

    coordinator = RepairWorkflowCoordinator(
        repository_path=target_repo,
        retriever=retriever,
        code_generator=code_gen,
        test_generator=test_gen,
        gemini_client=llm_client,
        policy_engine=policy_engine,
        permission_engine=perm_engine,
        tool_registry=tool_registry,
        max_retries=max_retries,
        command_timeout_seconds=timeout,
    )

    if not json_output:
        print("\n--- Initiating Agentic Repair Workflow ---")

    report = coordinator.run(
        task=task_str,
        target_file=target_file,
        test_command=test_command,
    )

    if json_output:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print("\n" + "=" * 60)
        status_label = report.status.value.upper()
        if report.status == RepairWorkflowStatus.SUCCESS:
            print_success("Repair Outcome:", f"SUCCESS ({report.total_attempts} attempt(s))")
        else:
            print_warning("Repair Outcome:", f"{status_label} ({report.total_attempts} attempt(s))")

        print_info("Workflow ID:", report.workflow_id)
        print_info("Duration:", f"{report.total_duration_seconds:.2f}s")
        print("\n--- Summary ---")
        print(report.summary)

        if report.final_diff:
            print("\n--- Final Verified Patch ---")
            print(report.final_diff)

    if report_file:
        try:
            report_path = Path(report_file).resolve()
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(report.to_markdown(), encoding="utf-8")
            if not json_output:
                print_success("\nReport Saved:", str(report_path))
        except Exception as e:
            print(f"Warning: Failed to save report to '{report_file}': {e}", file=sys.stderr)

    if report.status != RepairWorkflowStatus.SUCCESS:
        sys.exit(1)


if __name__ == "__main__":
    main()
