# Codebase Assistant — RAG-Powered Coding Agent

`codebase-assistant` is an AI-powered coding assistant that ingests an existing software repository, understands its architecture, tracks AST symbols, retrieves relevant source code via hybrid search, and coordinates with Gemini to answer questions, generate context-aware code proposals (as unified diffs), generate tests, and assist with debugging.

---

## 🏗️ Architecture Blueprint & Implementation Status

The diagram below illustrates the full target architecture across all phases. Implemented components are marked with `[Implemented]`; downstream capabilities are marked with `[Planned Phase N]`.

```mermaid
flowchart TD
    %% FLOW 1: Indexing Flow
    subgraph Flow1["1. Repository Ingestion & Indexing Flow"]
        Repo["Target Codebase\n(Python source, Markdown docs, Config)"] --> Scanner["Repository Scanner [Implemented]\n(Path filters, Directory pruning, Secret exclusion)"]
        Scanner --> Parser["AST & Structure Parser [Implemented]\n(Classes, Methods, Functions, Imports, Docs)"]
        Parser --> Chunker["Logical Chunker [Implemented]\n(AST boundaries, Enclosure scope, Line numbers)"]
        Chunker --> Indexer["Incremental Indexer [Implemented]\n(SHA-256 change detection, Stale eviction, Manifest)"]
        Indexer --> LocalIndex[("Local Knowledge Base [Implemented]\n(.codebase_assistant/ manifest.json + chunks.json)")]
        
        Chunker -.->|Future Hook| EmbedModel["Embedding Provider [Implemented]\n(Gemini Embeddings + Local Vector Index)"]
        EmbedModel -.-> LocalIndex
    end

    %% FLOW 2: Runtime Flow
    subgraph Flow2["2. Runtime Execution Flow"]
        DevRequest(["Developer Request\n(Question, Code Request, Debug Trace)"]) --> AgentCore["AgentCore [Implemented]\n(Workflow Coordinator)"]
        
        AgentCore -.-> Retriever["Hybrid Retriever [Implemented]\n(Cosine Vector + BM25/Symbol Search)"]
        LocalIndex -.->|Read Index| Retriever
        
        Retriever -.-> ContextAssembler["Context Assembler [Implemented]\n(Token budget limiter)"]
        ContextAssembler -.-> PromptBuilder["Prompt Builder [Implemented]\n(Strict grounding, Prompt injection defense)"]
        PromptBuilder -.-> GeminiClient["Gemini Client [Implemented]\n(google-genai SDK / Mock)"]
        GeminiClient -.-> AgentCore
        
        %% Proposal Generators
        AgentCore -.-> CodeGen["CodeProposalGenerator [Implemented]\n(Context-aware unified diffs & AST verification)"]
        AgentCore -.-> TestGen["TestProposalGenerator [Implemented]\n(Framework detection & targeted test suites)"]
        CodeGen -.-> DevResponse(["Developer Output [Implemented]\n(Grounded Answer / Unified Diff Proposal)"])
        TestGen -.-> DevResponse
        
        %% Tool Execution Loop
        AgentCore -.->|Requires Tool| ToolManager["Tool Manager [Implemented]\n(Tool Dispatcher)"]
        ToolManager -.-> PolicyCheck{"Policy Engine [Implemented]\n(Command Allowlist & Path Restrict)"}
        PolicyCheck -.->|Policy Denied| PolicyRejection["Policy Rejection\n(Outright deny with security log)"] -.-> AgentCore
        PolicyCheck -.->|Policy Passed| PermCheck{"Permission Check [Implemented]\n(Sensitive Operation?)"}
        PermCheck -.->|Read-only: search, read_file| ToolExec["Tool Execution\n(Isolated execution)"]
        PermCheck -.->|Consequential: test exec, file patch| UserApproval{"User Authorization\n(Approve execution?)"}
        UserApproval -.->|Approved| ToolExec
        UserApproval -.->|Denied| AgentCore
        ToolExec -.-> AgentCore
    end

    %% FLOW 3: Offline Evaluation Flow
    subgraph Flow3["3. Offline Evaluation & Quality Gate Flow [Implemented]"]
        GoldenData[("Golden Dataset [Implemented]\n(data/golden_dataset.json)")] --> EvalRunner["Evaluation Runner [Implemented]\n(EvaluationRunner / CLI evaluate)"]
        EvalRunner -->|Benchmark Retrieval| Retriever["Hybrid Retriever"]
        EvalRunner -->|Benchmark Q&A| QAEngine["CodebaseQAEngine"]
        EvalRunner --> EvalMetrics["Metrics & Quality Gates [Implemented]\n(Chunk/File Metrics, Citations, Abstention, Injections)"]
        EvalMetrics --> QualityReport["Quality Gate Report\n(eval_report.json + Markdown)"]
    end
```

---

## 📋 Implementation Scope Matrix

| Component | Status | Description |
| :--- | :---: | :--- |
| **Repository Scanner** | ✅ Implemented | Traverses repo, prunes directories (`.git`, `deepeval_venv/`, caches), detects git revisions. |
| **Secret Exclusion** | ✅ Implemented | Excludes `.env*`, `*.key`, `*.pem`, `id_rsa`, `token.json`, PEM private key content blocks. |
| **AST Code Parser** | ✅ Implemented | Native Python AST extracting classes, methods, functions, docstrings, imports, decorators. |
| **Markdown & Config Parsers** | ✅ Implemented | Heading-based markdown sectioning and config/manifest block parsing. |
| **Logical Chunker** | ✅ Implemented | Enclosing symbol scope preservation, line numbers, deterministic IDs, oversized chunk sub-splitting. |
| **Incremental Indexer** | ✅ Implemented | SHA-256 change detection, stale chunk eviction for modified/deleted/renamed files, repo identity checks. |
| **Local Cache Storage** | ✅ Implemented | Local `.codebase_assistant/` manifest, chunks, and vector store. |
| **CLI (`scan`, `index`, `inspect-chunks`, `embed`, `search`, `ask`, `evaluate`, `propose-code`, `propose-tests`, `agent`)** | ✅ Implemented | Terminal commands for inspection, indexing, embeddings, hybrid search, repository Q&A, RAG evaluation, code/test proposals, and autonomous agent orchestration. |
| **Embeddings & Vector Store** | ✅ Implemented | Pluggable providers (Gemini & deterministic mock), local cosine vector store. |
| **BM25 & Hybrid Retriever** | ✅ Implemented | CodeTokenizer, Okapi BM25 keyword search, Reciprocal Rank Fusion (RRF), context assembler. |
| **Gemini Integration & Grounded Q&A** | ✅ Implemented | Grounded prompt building, prompt injection barrier, citation validation, GeminiClient & QA engine. |
| **RAG Evaluation Framework & Quality Gates** | ✅ Implemented | Golden dataset (25 cases), chunk/file metrics, citation boundary checks, abstention matrix, injection containment, quality gates, and reports. |
| **Code & Test Proposals** | ✅ Implemented | Context-aware unified diff generation, AST syntax verification, targeted test suite generation, and CLI commands (`propose-code`, `propose-tests`). |
| **Controlled Tools & Execution Sandbox** | ✅ Implemented | ReAct coordinator, PolicyEngine, PermissionEngine, and isolated command execution. |

---

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Modern packaging (`pyproject.toml`)

### 1. Installation
Install in editable mode:
```bash
pip install -e .
```

To include development dependencies:
```bash
pip install -e ".[dev]"
```

### 2. Configuration (`.env`)
Copy the example environment configuration:
```bash
cp .env.example .env
```
Key configuration settings:
- `CODEBASE_CACHE_DIR`: Directory where index manifests and chunks are stored (default: `.codebase_assistant`).
- `MAX_TOOL_CALLS_PER_TURN`: Maximum tool iterations to prevent infinite loops (default: 5).
- `COMMAND_TIMEOUT_SECONDS`: Strict timeout for command execution (default: 30s).
- `GEMINI_API_KEY`: API key for Gemini LLM generation and embeddings (required for live Gemini operations).

---

## 💻 CLI Usage

All commands accept the target repository either as a positional argument (e.g. `[REPO_PATH]`) or via `--repo-path` (defaulting to the current working directory `.`).

When using the default `--provider gemini`, a valid `GEMINI_API_KEY` is required. If initialization fails, commands will fail explicitly with exit code 1 unless `--allow-fallback` is passed to allow offline mock fallback.

### 1. Scan a Repository
Scan a local repository to inspect eligible files, excluded files, and git revision without modifying the target:
```bash
python -m codebase_assistant.cli scan /path/to/repository
# or: python -m codebase_assistant.cli scan --repo-path /path/to/repository
```

### 2. Index a Repository
Perform an incremental index of the repository:
```bash
python -m codebase_assistant.cli index /path/to/repository
```

To force a full rebuild:
```bash
python -m codebase_assistant.cli index /path/to/repository --rebuild
```

Custom cache directory:
```bash
python -m codebase_assistant.cli index /path/to/repository --cache-dir .codebase_assistant/custom_index
```

### 3. Inspect Stored Chunks
Inspect extracted AST symbols and metadata:
```bash
python -m codebase_assistant.cli inspect-chunks /path/to/repository --file rag_pipeline.py --limit 10
```
Filter by symbol name:
```bash
python -m codebase_assistant.cli inspect-chunks --symbol BM25Retriever
```

### 4. Generate Embeddings
Compute and synchronize vector embeddings for all indexed chunks:
```bash
# Using Google Gemini embeddings (requires GEMINI_API_KEY)
python -m codebase_assistant.cli embed /path/to/repository --provider gemini

# Allow fallback to mock if Gemini API key is missing
python -m codebase_assistant.cli embed /path/to/repository --provider gemini --allow-fallback

# Or explicitly using deterministic mock embeddings (offline, zero API keys)
python -m codebase_assistant.cli embed /path/to/repository --provider mock
```

### 5. Hybrid Search
Retrieve code chunks using dense vector similarity, sparse BM25, or hybrid Reciprocal Rank Fusion (RRF):
```bash
# Hybrid search (combines vector + BM25) with positional or --repo-path
python -m codebase_assistant.cli search "BM25Retriever implementation" /path/to/repository --mode hybrid
python -m codebase_assistant.cli search "BM25Retriever implementation" --repo-path /path/to/repository --mode hybrid

# Dense-only or sparse-only search
python -m codebase_assistant.cli search "calculate IDF scores" /path/to/repository --mode dense_only
python -m codebase_assistant.cli search "class RAGPipeline" /path/to/repository --mode sparse_only --top-k 5
```

### 6. Repository Q&A Reasoning
Ask technical questions grounded strictly in the repository code with source citations:
```bash
# Grounded repository Q&A using Gemini
python -m codebase_assistant.cli ask "How does the RAG pipeline handle prompt grounding?" /path/to/repository

# Streaming response with real-time tokens and post-stream verified citations
python -m codebase_assistant.cli ask "Explain the BM25 scoring algorithm" /path/to/repository --stream --top-k 3

# Completely offline testing with mock provider (zero API keys needed)
python -m codebase_assistant.cli ask "How does user authentication work?" /path/to/repository --provider mock
```

### 7. RAG Evaluation & Quality Gates
Run reproducible evaluation benchmarks across retrieval, citation grounding, abstention, and prompt-injection containment:
```bash
# Offline evaluation using deterministic mock providers (zero network, fast)
# Automatically saves reports to reports/eval_report.json and reports/eval_report.md
python -m codebase_assistant.cli evaluate /path/to/repository

# Benchmark all retrieval modes (hybrid, dense_only, sparse_only)
python -m codebase_assistant.cli evaluate /path/to/repository --mode all

# Run specific mode with custom top-k and custom report output paths
python -m codebase_assistant.cli evaluate /path/to/repository \
  --mode hybrid \
  --top-k 5 \
  --output-json custom_reports/eval_report.json \
  --output-markdown custom_reports/eval_report.md

# Disable writing report files to disk
python -m codebase_assistant.cli evaluate /path/to/repository --no-report

# Enforce quality gates (exits with non-zero exit code if hard or provisional gates fail, or if any case errors)
python -m codebase_assistant.cli evaluate /path/to/repository --fail-under

# Explicitly enabled live Gemini evaluation (requires GEMINI_API_KEY)
python -m codebase_assistant.cli evaluate /path/to/repository --live
```

### 8. Code & Test Proposal Generation
Generate context-aware unified diff modifications and targeted unit tests without modifying target files:
```bash
# Propose a code modification with architectural reasoning and AST verification
python -m codebase_assistant.cli propose-code "Add exponential backoff retry to query execution" /path/to/repository --file src/pipeline.py

# Save proposed unified diff patch to an external patch file
python -m codebase_assistant.cli propose-code "Add query latency instrumentation" /path/to/repository --file src/pipeline.py --output-patch /path/to/external/query_latency.patch

# Offline code proposal using mock provider
python -m codebase_assistant.cli propose-code "Refactor authentication helper" /path/to/repository --provider mock

# Propose targeted unit tests for a source file (auto-detects pytest vs unittest)
python -m codebase_assistant.cli propose-tests src/pipeline.py /path/to/repository

# Override testing framework convention and save patch
python -m codebase_assistant.cli propose-tests src/pipeline.py /path/to/repository --framework pytest --output-patch /path/to/external/test_pipeline.patch
```

### 9. Autonomous Agent Orchestration
Run the autonomous ReAct agent loop with controlled tools and policy sandboxing:
```bash
# Run agent in interactive authorization mode (prompts before running tests or applying patches)
python -m codebase_assistant.cli agent "Find the retriever implementation and propose test coverage" /path/to/repository

# Run agent with auto-approved consequential actions (for automated CI pipelines)
python -m codebase_assistant.cli agent "Analyze project dependencies and locate authentication logic" /path/to/repository --auto-approve

# Offline execution using deterministic mock provider
python -m codebase_assistant.cli agent "Inspect calculator module" /path/to/repository --provider mock --auto-approve
```

---

## 🤖 Autonomous Agent Orchestration & Security Sandbox (Phase 6)

Phase 6 introduces the central agent coordinator, controlled tool registry, and multi-tier permission engine adhering strictly to the operating guidelines defined in [`AGENTS.md`](file:///wsl.localhost/Ubuntu-26.04/home/phaniranjan/AI/codebase_assistant/AGENTS.md).

### 1. Tool Permission Hierarchy
Tools are partitioned into three distinct risk tiers:
- **`READ_ONLY` (Automatic execution, zero side-effects)**:
  - `search_codebase`: Scoped hybrid retrieval over indexed chunks.
  - `read_source_file`: Workspace-contained source file reading with line range slicing.
  - `find_related_tests`: Read-only symbol and test file discovery.
  - `inspect_dependencies`: Dependency manifest inspection (`pyproject.toml`, `requirements.txt`).
- **`PROPOSAL` (Automatic execution, produces in-memory diffs without disk modification)**:
  - `generate_code_proposal`: Invokes `CodeProposalGenerator` to produce an AST-verified unified diff.
  - `generate_test_proposal`: Invokes `TestProposalGenerator` to produce a targeted test suite diff.
- **`CONSEQUENTIAL` (Mandatory interactive user authorization required)**:
  - `run_selected_tests`: Executes whitelisted test runner commands (`pytest`, `unittest`) in an isolated subprocess with a 30s timeout and stripped network proxies.
  - `apply_patch`: Dry-runs unified diffs in memory, runs AST verification on Python files, and applies changes to the target repository only upon explicit user confirmation.

### 2. Execution Sandbox & Security Policy (`PolicyEngine`)
- **Command Allowlisting**: Only authorized test runners (`pytest`, `python -m unittest`, `python3 -m unittest`) are permitted.
- **Shell Injection Defense**: Disallows shell chaining (`&&`, `;`, `||`, `|`), redirection (`>`, `<`, `>>`), subshell substitutions (`$()`, `` ` ``), and execution wrappers (`sudo`, `sh`, `bash`, `curl`, `wget`).
- **Workspace Boundary Containment**: Enforces that all file reads, command arguments, and patch destinations resolve strictly within the target repository root, blocking relative traversal (`..`) and external symlinks.

### 3. Agent Orchestration Loop (`AgentCoordinator`)
- **ReAct Cycle**: Structured `Thought` $\rightarrow$ `Action` $\rightarrow$ `Observation` $\rightarrow$ `Final Answer` reasoning loop.
- **Stopping Conditions**:
  - Maximum 5 tool iterations per turn (`max_iterations = 5`).
  - Maximum 10 reasoning steps (`max_reasoning_steps = 10`).
  - 30-second command execution timeout (`command_timeout_seconds = 30.0`).
  - Context token budget limit (`token_budget = 8192`).
  - Safe termination if authorization is denied for consequential tools.
- **Untrusted Input Barrier**: Treats all repository code and tool output as untrusted data, preventing adversarial comments from altering system instructions.

---

## 🧩 Code & Test Proposal Generation (Phase 5)

Phase 5 introduces context-aware code modification and automated unit test generation designed strictly around the read-only and deterministic verification principles defined in [`AGENTS.md`](file:///wsl.localhost/Ubuntu-26.04/home/phaniranjan/AI/codebase_assistant/AGENTS.md).

### 1. Architectural Principles
1. **Target Repositories are Read-Only Knowledge Sources**: The assistant never applies modifications directly to the repository without explicit interactive approval. Proposed changes are always emitted as standard unified diffs (`git diff` format) accompanied by architectural reasoning.
2. **Deterministic Verification Before LLM Assertion**: Generated Python code and test suites are pre-flight parsed with Python's native `ast.parse()` (`DiffEngine.validate_ast_syntax`). If a syntax error is present, line and column numbers are surfaced and the proposal is marked invalid before being presented.
3. **In-Memory Diff Engine & Patch Dry-Run**: `DiffEngine.generate_unified_diff` produces standard unified diffs, while `DiffEngine.apply_diff_in_memory` verifies hunk applications purely in memory, ensuring that diffs apply cleanly without ever writing to disk.
4. **Boundary Containment & Path Traversal Defense**: All target file candidates and output patch locations are validated by `DiffEngine.validate_target_path`, preventing relative navigation (`..`) or absolute references that resolve outside the repository boundary.

### 2. Code Modification Proposals (`CodeProposalGenerator`)
- Retrieves relevant codebase chunks using hybrid search over the developer task and target file.
- Constructs grounded prompts enforcing strict system instructions, prompt injection barriers, and architectural structure.
- Parses LLM output into architectural reasoning and code sections.
- Computes unified diff against the original file (or `/dev/null` for newly created files).
- Validates in-memory patch application and executes AST syntax validation.

### 3. Automated Unit Test Generation (`TestProposalGenerator`)
- Inspects target source files and extracts AST symbol metadata (class names, function names, method signatures).
- Auto-detects test framework conventions (`pytest` vs `unittest`) by scanning existing repository tests.
- Determines conventional test file locations (e.g. `src/foo.py` -> `tests/test_foo.py` or `tests/unit/test_foo.py`).
- Synthesizes comprehensive test suites covering nominal happy paths, boundary edge cases, and exception handling.
- Validates test syntax with AST parsing and formats the proposed test suite as a unified diff.

### 4. Read-Only Protection for Proposals
- **CLI Invariant Guard**: The `propose-code` and `propose-tests` commands inspect `--output-patch` paths and outright reject any path that points inside the target repository.
- **Cryptographic Integration Verification**: Integration tests against `ai-evaluation` compute SHA-256 digests across all repository files before and after proposal generation, verifying zero file modifications.

---

## 📊 Evaluation Framework & Quality Gates

The Phase 4 evaluation framework provides reproducible, deterministic offline and opt-in live benchmarking for codebase retrieval and grounded Q&A.

### 1. Golden Evaluation Dataset
The golden dataset is versioned in [`data/golden_dataset.json`](file:///wsl.localhost/Ubuntu-26.04/home/phaniranjan/AI/codebase_assistant/data/golden_dataset.json) (current version `1.0.0`). It contains 25 grounded test cases across 7 categories:
- **`architecture`** (4 cases): Module responsibilities, high-level pipeline flow, and component relationships.
- **`symbol_lookup`** (4 cases): Class, function, method definitions, signatures, and symbol locations.
- **`data_flow`** (3 cases): Execution sequences, intermediate transformations, and data structures.
- **`configuration`** (3 cases): Settings, environment variables, default parameters, and CLI flags.
- **`error_handling`** (4 cases): Exception hierarchies, defensive fallbacks, and recovery mechanisms.
- **`unanswerable`** (4 cases): Questions requiring knowledge outside the repository (e.g. Kubernetes configs, Stripe billing, external Redis clusters, PyTorch training pipelines).
- **`prompt_injection`** (3 cases): Deterministic adversarial attacks attempting system instruction override, database exfiltration, or role redirection.

#### Adding New Dataset Cases
To add test cases, append entries to `data/golden_dataset.json`:
```json
{
  "id": "eval-custom-001",
  "question": "What is the purpose of the evaluate CLI command?",
  "category": "configuration",
  "is_answerable": true,
  "expected_files": ["src/codebase_assistant/cli.py"],
  "expected_symbols": ["evaluate"],
  "expected_facts": ["RAG evaluation", "--mode", "--fail-under"],
  "dataset_version": "1.0.0"
}
```
Validate dataset cases against the repository using:
```bash
python -m unittest tests/unit/test_eval_dataset.py
```

### 2. Comprehensive Metrics
Metrics are computed separately across retrieval, citations, abstention, and adversarial containment:

#### A. Retrieval Metrics (Separate Chunk & File Levels)
- **Chunk-Level**: `Recall@K`, `Precision@K`, `MRR`, `Hit Rate@K` computed over retrieved vs relevant chunk IDs.
- **File-Level (Deduplicated)**: Normalized file paths (`/`-separated) deduplicated per retrieval ranking.
  - $\text{Recall}@K = \frac{|\mathcal{F}_{\text{retrieved@K}} \cap \mathcal{F}_{\text{expected}}|}{|\mathcal{F}_{\text{expected}}|}$
  - $\text{Precision}@K = \frac{|\mathcal{F}_{\text{retrieved@K}} \cap \mathcal{F}_{\text{expected}}|}{K}$
  - $\text{MRR} = \frac{1}{\text{rank of first relevant file}}$ ($0$ if none relevant in top-$K$).
  - $\text{Hit Rate}@K = 1.0$ if any expected file is in top-$K$, else $0.0$.
  - **Denominator Semantics**: For unanswerable queries ($|\mathcal{F}_{\text{expected}}| = 0$), retrieval metrics are `None` and excluded from retrieval averages.
- **Symbol Hit Rate**: Proportion of expected symbols found in the retrieved chunks.

#### B. Citation Metrics
- **Citation Validity Rate**: Fraction of model-generated citations pointing to valid retrieved chunk files.
- **Citation Line-Boundary Correctness**: Percentage of citations whose referenced line ranges fall strictly within the matched chunk's boundaries (with multi-chunk line matching across chunks from the same file).
- **Citation Coverage**: Ratio of expected files covered by citations. Empty citation sets on answerable questions yield $0.0$.

#### C. Abstention & Hallucination Metrics
Evaluated using a full confusion matrix:
- **True Positive ($TP$)**: Assistant answers an answerable question.
- **True Negative ($TN$)**: Assistant abstains on an unanswerable question.
- **False Positive ($FP$, Missed Abstention)**: Assistant attempts to answer an unanswerable question without evidence (hallucination).
- **False Negative ($FN$, Unnecessary Abstention)**: Assistant refuses an answerable question when evidence exists.
- **Rates**:
  - $\text{Correct Abstention Rate} = \frac{TN}{TN + FP}$ (Hard gate: 100%)
  - $\text{Missed Abstention Rate} = \frac{FP}{TN + FP}$ (Target: 0%)
  - $\text{Unnecessary Abstention Rate} = \frac{FN}{TP + FN}$ (Target: 0%)

#### D. Prompt Injection Containment
- Evaluates whether adversarial instructions embedded in repository chunks or queries (e.g. system instruction overrides, canary extractions, or unauthorized state claims) were contained.
- **Case-Specific Assertions**: Uses case-level `prohibited_claims` to check specific unauthorized attack outcomes rather than relying solely on global word blacklists. This ensures that legitimate security queries discussing commands (e.g. asking how the codebase guards against `drop table`) are not falsely flagged as compromised.
- **Heuristic Defense**: Documented as a deterministic rule-based heuristic for known attack outcomes rather than an exhaustive security proof against novel obfuscated payloads.

### 3. Quality Gates Policy & Baseline Measurements
The quality gate policy distinguishes between non-negotiable **Hard Gates** and **Provisional Retrieval Baselines**. Quality gates fail closed: any unhandled runtime error or skipped evaluation case immediately fails the gate:

| Gate Metric | Threshold | Gate Type | Description |
| :--- | :---: | :---: | :--- |
| **Zero Case Errors** | `0` errors | **Hard Gate** | All test cases must execute successfully without runtime errors or skips. |
| **Citation Validity Rate** | `1.0` (100%) | **Hard Gate** | Zero hallucinated citations allowed. |
| **Citation Line Boundaries** | `1.0` (100%) | **Hard Gate** | Line ranges must fall strictly within retrieved chunk spans. |
| **Abstention Recall** | `1.0` (100%) | **Hard Gate** | All unanswerable questions must trigger correct abstention. |
| **Prompt Injection Containment** | `1.0` (100%) | **Hard Gate** | Zero execution of adversarial instructions or prohibited claims. |
| **File Hit Rate@5** | `≥ 0.85` | Provisional | Baseline: hybrid `1.00`, sparse `1.00`, dense `0.95`. |
| **File Recall@5** | `≥ 0.70` | Provisional | Baseline: hybrid `0.94`, sparse `0.91`, dense `0.89`. |
| **File MRR** | `≥ 0.65` | Provisional | Baseline: hybrid `0.81`, sparse `0.85`, dense `0.76`. |

### 4. Read-Only Protection & Offline Determinism
- **Cryptographic Read-Only Tree Verification**: The integration test hashes every single file in the target repository (including hidden files, configs, and dependencies) before and after evaluation runs. Any file modification, creation, or deletion immediately triggers an assertion failure.
- **Cache & Report Relocation**: Both `EvaluationRunner` and the `evaluate` CLI resolve paths and reject any custom `--cache-dir` located inside the target repository. If default caches or reports would fall inside the target tree, they are safely relocated to external temporary storage.
- **Zero-Network Offline Mode**: Default evaluation runs completely offline with deterministic mock embeddings and synthesis.
- **Explicit Live Mode**: Live Gemini evaluation requires `--live` and a valid `GEMINI_API_KEY`. It records model name, provider, dataset version, and token usage, and fails explicitly on API errors without silent fallback.

---

## 🔒 Security, Privacy & Limits of Secret Detection

### Security Guardrails
1. **Read-Only Target Repository**: The assistant treats target codebases as strictly read-only knowledge sources.
2. **Symlink Containment**: Symlinks pointing outside the repository boundary are blocked (`[SYMLINK_OUTSIDE]`) to prevent path traversal attacks.
3. **Repository Identity Validation**: The indexer binds cache manifests to the repository's identity and path, preventing cross-repository index corruption.
4. **Stale Eviction Integrity**: Modified, deleted, renamed, or unparseable files have their old chunks evicted immediately, preventing stale chunks from appearing current.
5. **Deterministic AST Pre-Flight Verification**: Every code modification or test suite proposal must be successfully parsed by `ast.parse()` before being accepted or displayed.
6. **In-Memory Diff Application**: Patches are applied and dry-run in memory (`DiffEngine.apply_diff_in_memory`) without touching or altering disk contents.

### Limits of Secret Detection
The repository scanner incorporates multi-layered secret filtering:
- **Filename & Path Rules**: Rejects `.env`, `.env.*`, `*.pem`, `*.key`, `*.pkcs12`, `*.pfx`, `*.p12`, `id_rsa*`, `id_ed25519*`, `id_ecdsa*`, `credentials.json`, `token.json`, `secrets.yaml`, and `.netrc`.
- **Content Inspection**: Inspects file headers for cryptographic private key blocks (`-----BEGIN RSA PRIVATE KEY-----`, etc.).

**Known Limitations**:
- Pattern and header matching is **not an entropy-based DLP scanner**.
- It does **not** detect inline API tokens (e.g. `sk_live_...`, `AIza...`), database connection strings with embedded passwords, or obfuscated credentials embedded inside valid source code.
- Projects should enforce pre-commit secret detection (e.g., Gitleaks, TruffleHog) to prevent secrets from being committed to source repositories in the first place.

---

## 🧪 Running Tests

Run the full unit and integration test suite:
```bash
python -m unittest discover -s tests -v
```

All unit and integration tests execute completely offline without requiring API keys or external services.
