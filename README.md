# Codebase Assistant — RAG-Powered Coding Agent

`codebase-assistant` is an autonomous, RAG-powered coding assistant and agentic repair system designed to inspect software repositories, index AST symbols and documentation, perform hybrid semantic/keyword search, answer technical architecture questions with strict citations, generate AST-verified code and test proposals (as unified diffs), and orchestrate autonomous, bounded repair workflows with sandboxed test execution.

---

## 🏗️ Architecture Blueprint

The diagram below illustrates the end-to-end architecture across all seven development phases, highlighting the clear separation between ingestion, grounded retrieval, autonomous tool orchestration, sandboxed repair, and offline quality evaluation:

```mermaid
flowchart TD
    %% FLOW 1: Indexing Flow
    subgraph Flow1["1. Repository Ingestion & Indexing (Phases 1 & 2)"]
        Repo["Target Codebase\n(Read-Only Knowledge Source)"] --> Scanner["Repository Scanner\n(Path filters, pruning, secret exclusion)"]
        Scanner --> Parser["AST & Structure Parser\n(Classes, methods, functions, docs)"]
        Parser --> Chunker["Logical Chunker\n(Scope preservation, line bounds, chunk IDs)"]
        Chunker --> Indexer["Incremental Indexer\n(SHA-256 change detection, stale eviction)"]
        Indexer --> LocalIndex[("Local Knowledge Base\n(.codebase_assistant/ manifest + chunks)")]
        Chunker -.-> EmbedModel["Embedding Provider\n(Gemini text-embedding-004 / Mock)"]
        EmbedModel -.-> VectorIndex[("Vector Store\n(.codebase_assistant/ cosine index)")]
    end

    %% FLOW 2: Runtime Retrieval & Q&A
    subgraph Flow2["2. Grounded Retrieval & Q&A (Phase 3)"]
        UserQ(["Developer Query / Task"]) --> Retriever["Hybrid Retriever\n(Cosine Vector Similarity + Okapi BM25 / RRF)"]
        LocalIndex --> Retriever
        VectorIndex --> Retriever
        Retriever --> ContextAssembler["Context Assembler\n(Token budget limiter & deduplication)"]
        ContextAssembler --> PromptBuilder["Prompt Builder\n(Strict grounding, injection barrier)"]
        PromptBuilder --> LLMClient["Gemini Client\n(gemini-2.5-pro / gemini-2.5-flash / Mock)"]
        LLMClient --> Answer(["Grounded Answer with Citations"])
    end

    %% FLOW 3: Proposals & Patch Verification
    subgraph Flow3["3. Proposal Generation & Diff Engine (Phase 5)"]
        PromptBuilder --> CodeGen["CodeProposalGenerator\n(Context-aware unified diff generation)"]
        PromptBuilder --> TestGen["TestProposalGenerator\n(Framework auto-detection, targeted test suite)"]
        CodeGen --> DiffEngine["Deterministic Diff Engine\n(In-memory patch dry-run, AST syntax check)"]
        TestGen --> DiffEngine
        DiffEngine --> VerifiedProposal(["Verified Unified Diff Proposal"])
    end

    %% FLOW 4: Agent Orchestration & Sandboxing
    subgraph Flow4["4. Agent Orchestration & Security Policy (Phase 6)"]
        AgentCore["AgentCoordinator (ReAct Loop)\n(Step, token budget & iteration bounds)"]
        AgentCore --> ToolReg["Tool Registry\n(READ_ONLY, PROPOSAL, CONSEQUENTIAL)"]
        ToolReg --> PolicyCheck{"PolicyEngine\n(Command whitelist & path boundary)"}
        PolicyCheck -->|Policy Denied| Abort["Immediate Rejection"]
        PolicyCheck -->|Policy Passed| PermCheck{"PermissionEngine\n(Consequential Action?)"}
        PermCheck -->|Read-only / Proposal| AutoExec["Automatic Execution"]
        PermCheck -->|Consequential| AuthCheck{"User Authorization\n(Interactive prompt / --auto-approve)"}
        AuthCheck -->|Denied| StopAgent["Clean Termination"]
        AuthCheck -->|Approved| ExecTool["Execute Tool"]
    end

    %% FLOW 5: Agentic Repair Workflows
    subgraph Flow5["5. Agentic Repair & Sandboxed Verification (Phase 7)"]
        RepairCoord["RepairWorkflowCoordinator\n(Cumulative budget, timeout & retry tracking)"]
        RepairCoord --> PlanStep["Task Planning & Context Retrieval"]
        PlanStep --> ProposeStep["Diff Proposal & AST Verification"]
        ProposeStep --> SandboxExec["Disposable Sandbox Workspace\n(tempfile workspace, proxy blocking, 30s timeout)"]
        SandboxExec --> TestRun{"Sandboxed Test Execution\n(pytest / python -m unittest)"}
        TestRun -->|Tests Pass| SuccessState["SUCCESS\n(Report generated, changes verified)"]
        TestRun -->|Tests Fail| DiagEngine["DiagnosticEngine\n(Stack trace parsing, prompt sanitization)"]
        DiagEngine --> RetryCheck{"Retries < Max?\n(Bounded self-repair)"}
        RetryCheck -->|Yes| ProposeStep
        RetryCheck -->|Exhausted| Rollback["Atomic Rollback\n(Restore backups, unlink new files)"]
        Rollback --> FailState["MAX_RETRIES_EXCEEDED / ROLLBACK_FAILED"]
    end

    %% FLOW 6: Offline Evaluation Flow
    subgraph Flow6["6. Offline Benchmark & Quality Gates (Phase 4)"]
        GoldenData[("Golden Dataset\n(data/golden_dataset.json - 25 cases)")] --> EvalRunner["EvaluationRunner\n(Deterministic offline benchmark)"]
        EvalRunner --> QualityGates{"Quality Gate Policy\n(Hard gates & provisional baselines)"}
        QualityGates --> EvalReport["Reports & Artifacts\n(reports/eval_report.json + Markdown)"]
    end
```

---

## 📋 Features by Phase

| Phase | Capability | Core Modules |
| :--- | :--- | :--- |
| **Phase 1** | **Repository Ingestion & AST Parsing** | `RepositoryScanner`, `CodeParser`, `LogicalChunker`, `IncrementalIndexer` |
| **Phase 2** | **Incremental Indexing & Local Caching** | SHA-256 change detection, repo identity check, stale chunk eviction |
| **Phase 3** | **Embeddings, Hybrid Retrieval & Grounded Q&A** | `GeminiEmbeddingProvider`, `VectorStore`, `CodeTokenizer`, `BM25Retriever`, `HybridRetriever`, `CodebaseQAEngine` |
| **Phase 4** | **Golden Dataset & Quality Gate Evaluation** | 25-case golden benchmark, chunk/file metrics, citation line checking, abstention matrix, injection defense |
| **Phase 5** | **Code & Test Proposals with Diff Engine** | `DiffEngine` (in-memory dry-run, AST syntax check, unified diff splitting), `CodeProposalGenerator`, `TestProposalGenerator` |
| **Phase 6** | **Controlled Tools & Execution Sandbox** | `AgentCoordinator` (ReAct loop), `PolicyEngine`, `PermissionEngine`, `ToolRegistry` |
| **Phase 7** | **Agentic Repair Workflows** | `RepairWorkflowCoordinator`, disposable sandbox copy, `DiagnosticEngine`, multi-file backup/atomic rollback |
| **Phase 8** | **Release Hardening, CI/CD & CLI Baseline** | Matrix CI workflow, Black/isort/mypy compliance, read-only cryptographic guarantees, packaging |
| **Phase 9** | **Antigravity Conversational IDE Integration (MCP)** | `MCPBackend`, `FastMCP` STDIO server, 11 semantic engineering tools, conversational skill (`.agents/skills/codebase-assistant/SKILL.md`), cryptographic token gating, fail-closed provider safety |

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Git

### 1. Editable Installation
Clone the repository and install with all optional dependencies:
```bash
git clone https://github.com/phaniranjan/codebase_assistant.git
cd codebase_assistant

# Install with AI (Gemini/FAISS), dev (pytest/mypy/black), and eval dependencies:
pip install -e ".[ai,dev,eval]"
```

Verify that the CLI is installed and responsive:
```bash
codebase-assistant --help
```

### 2. Environment Configuration
Copy the example `.env` file:
```bash
cp .env.example .env
```

Key environment configuration variables:
```ini
# Gemini API Key (required for live Gemini operations; not required for mock/offline mode)
GEMINI_API_KEY=your_gemini_api_key_here

# Model Selection
GEMINI_MODEL=gemini-2.5-flash
GEMINI_EMBEDDING_MODEL=text-embedding-004

# Cache & Storage
CODEBASE_CACHE_DIR=.codebase_assistant

# Agent Bounds & Safety
MAX_TOOL_CALLS_PER_TURN=5
MAX_REASONING_STEPS=10
COMMAND_TIMEOUT_SECONDS=30.0
CONTEXT_TOKEN_BUDGET=8192
```

---

## 💻 Complete CLI Reference

All CLI commands accept the target repository either as a positional argument (`[REPO_PATH]`) or via `--repo-path` (defaulting to the current working directory `.`).

Commands fail fast by default: when live Gemini operations fail due to quota or missing keys, they exit with code 1 instead of silently falling back to mock mode, unless `--allow-fallback` is explicitly provided.

### 1. `scan`
Scan a repository and display detected files, revisions, and excluded secrets:
```bash
codebase-assistant scan /path/to/repository
```

### 2. `index`
Build or incrementally update the local AST symbol index:
```bash
codebase-assistant index /path/to/repository

# Force full rebuild of the index:
codebase-assistant index /path/to/repository --rebuild

# Custom cache directory:
codebase-assistant index /path/to/repository --cache-dir /external/cache/dir
```

### 3. `inspect-chunks`
Inspect extracted chunks, AST symbol names, line boundaries, and metadata:
```bash
# Inspect chunks for a specific source file:
codebase-assistant inspect-chunks /path/to/repository --file src/pipeline.py --limit 10

# Filter by symbol name:
codebase-assistant inspect-chunks /path/to/repository --symbol BM25Retriever
```

### 4. `embed`
Generate and synchronize vector embeddings for all indexed chunks:
```bash
# Using live Gemini embeddings (requires GEMINI_API_KEY):
codebase-assistant embed /path/to/repository --provider gemini

# Offline execution using deterministic mock embeddings (zero API keys):
codebase-assistant embed /path/to/repository --provider mock
```

### 5. `search`
Retrieve relevant code chunks using dense vector similarity, sparse BM25, or hybrid Reciprocal Rank Fusion:
```bash
# Hybrid search (default):
codebase-assistant search "BM25 score calculation" /path/to/repository --mode hybrid --top-k 5

# Dense-only or sparse-only search:
codebase-assistant search "class CodebaseQAEngine" /path/to/repository --mode sparse_only
```

### 6. `ask`
Ask questions grounded strictly in the target repository code with verified citations:
```bash
# Live grounded Q&A with Gemini:
codebase-assistant ask "How does the indexer detect modified files?" /path/to/repository

# Streaming response with real-time token output:
codebase-assistant ask "Explain the hybrid retrieval fusion formula" /path/to/repository --stream

# Completely offline using mock provider:
codebase-assistant ask "What is the token budget limit?" /path/to/repository --provider mock
```

### 7. `evaluate`
Execute reproducible offline or live evaluation benchmarks against the golden dataset:
```bash
# Offline benchmark with quality gates enforced (exits code 1 on failure):
codebase-assistant evaluate /path/to/repository --dataset data/golden_dataset.json --fail-under

# Compare all retrieval modes (hybrid, dense_only, sparse_only):
codebase-assistant evaluate /path/to/repository --mode all

# Explicit live evaluation with Gemini:
codebase-assistant evaluate /path/to/repository --live
```

### 8. `propose-code`
Generate context-aware unified diff modifications with pre-flight AST validation:
```bash
# Propose a code modification:
codebase-assistant propose-code "Add exponential backoff retry" /path/to/repository --file src/pipeline.py

# Save proposed patch to an external patch file (never inside the target repository):
codebase-assistant propose-code "Add logging" /path/to/repository --file src/pipeline.py --output-patch /tmp/patch.diff
```

### 9. `propose-tests`
Generate targeted unit tests with automatic framework detection (`pytest` vs `unittest`):
```bash
codebase-assistant propose-tests src/pipeline.py /path/to/repository
```

### 10. `agent`
Execute autonomous ReAct agent loop with controlled tools and interactive authorization:
```bash
# Interactive mode (prompts before running tests or modifying files):
codebase-assistant agent "Locate calculator implementation and verify tests" /path/to/repository

# Automated mode with pre-approved consequential actions (for disposable CI):
codebase-assistant agent "Analyze dependency vulnerabilities" /path/to/repository --auto-approve

# Offline agent execution:
codebase-assistant agent "Inspect scanner module" /path/to/repository --provider mock --auto-approve
```

### 11. `repair`
Orchestrate an end-to-end autonomous repair workflow (plan, propose, patch in memory, run sandboxed tests, diagnose failures, bounded self-repair, atomic rollback):
```bash
# Autonomous repair with automated verification:
codebase-assistant repair --task "Fix ZeroDivisionError in calculate_average" --target-file src/calc.py --test-command "pytest tests/test_calc.py" /path/to/repository --auto-approve

# Output JSON report:
codebase-assistant repair --task "Refactor cache sync" /path/to/repository --json

# Save Markdown audit report to external file:
codebase-assistant repair --task "Fix timeout bug" /path/to/repository --report-file reports/repair_audit.md
```

### 12. `mcp`
Launch the Model Context Protocol (MCP) server over standard input/output (STDIO) for Google Antigravity conversational IDE integration:
```bash
# Launch MCP server with live Gemini providers (fails fast if GEMINI_API_KEY is missing or invalid):
codebase-assistant mcp

# Force offline deterministic mock mode (zero network and zero API keys required):
codebase-assistant mcp --mock

# Launch MCP server allowing fallback to mock mode if live Gemini initialization fails:
codebase-assistant mcp --allow-fallback
```

You can also launch the server directly using the registered console script:
```bash
codebase-assistant-mcp
```

---

## 🔒 Security Model & Sandbox Architecture

The security model is defined strictly around the principle that **target codebases are untrusted and read-only knowledge sources**:

### 1. Tool Permission Hierarchy & Authorization Tiers
- **`READ_ONLY`** (`search_codebase`, `read_source_file`, `find_related_tests`, `inspect_dependencies`, `repository_status`, `ask_codebase`): Scoped strictly within the repository workspace. External symlinks and path traversals (`..`) are outright rejected.
- **`PROPOSAL`** (`generate_code_proposal`, `generate_test_proposal`, `propose_code`, `propose_tests`): Synthesizes changes entirely in memory as standard unified diffs. Pre-flight parses code with `ast.parse()`. No disk writes.
- **`CONSEQUENTIAL`** (`run_selected_tests`, `apply_patch`, `apply_approved_patch`): Requires explicit human authorization. In CLI mode, interactive prompts or `--auto-approve` (disposable CI only) are enforced. Over conversational MCP, model-generated confirmations alone cannot bypass authorization.

### 2. Cryptographic Authorization & Model Confirmation Immunity
- **Model Confirmation Immunity**: LLMs can easily generate strings like `confirmation="CONFIRM_APPLY"`. In conversational MCP mode, passing confirmation parameters alone **never modifies files on disk**.
- **Safe Pre-Flight Fallback**: When `apply_approved_patch` is invoked without a valid server authorization token, it performs complete in-memory hunk dry-runs and Python AST syntax validation for all target files, saves the pre-flight validated patch to an external `.diff` file outside the repository, and returns the exact command for manual application:
  ```bash
  cd /path/to/workspace && git apply /external/cache/patches/validated_patch.diff
  ```
- **Cryptographic Token Gating**: Direct programmatic disk modifications are permitted only when presenting a valid, unexpired server-issued one-time token (`auth_token`) generated via `issue_authorization_token()`.

### 3. Workspace Boundary & Protected Repository Invariant
- **Protected Repositories**: The target repository (e.g., `ai-evaluation` / `ai_evaluation`) is permanently hardcoded as a protected read-only knowledge source. Attempting to register it as writable or modify it raises an immediate `PermissionError` and is denied.
- **Exact Registered Workspaces**: Modifications can only target explicitly registered writable workspaces (`register_writable_workspace` or `set_active_repository(..., writable=True)`). Paths are canonicalized via `os.path.realpath()`; symlink escapes, parent traversals (`..`), and absolute path escapes are blocked. An active repository's writable flag never authorizes an unrelated path.

### 4. Multi-File Patch Atomicity, Permissions & Exact Rollback
- **Deterministic Splitting**: Multi-file unified diffs are deterministically split into per-file patches via `DiffEngine.split_unified_diff`.
- **Pre-Flight Validation (Zero Partial Writes)**: Every affected file undergoes an in-memory dry-run (`DiffEngine.apply_diff_in_memory`) and deterministic AST syntax check (`DiffEngine.validate_ast_syntax`) before any disk write. If *any* file fails, the entire operation is rejected with zero disk modifications.
- **Atomic Rollback & Permission Preservation**: During authorized writes, original file contents and filesystem permissions (`st_mode`) are tracked in a `written_stack`. If any write fails midway:
  - All previously modified files are restored in reverse order to their exact original contents and `st_mode` permissions.
  - Any newly created files are unlinked.
  - If rollback itself encounters an error, `CRITICAL: ROLLBACK_FAILED` is explicitly reported with the full list of affected files.

### 5. Disposable Execution Workspace, Credential Scrubbing & Network Isolation
When running test suites during repair workflows:
1. **Disposable Working Copy**: Tests are executed strictly within an isolated temporary directory (`tempfile.TemporaryDirectory(prefix="repair_sandbox_")`). The target repository is never directly touched.
2. **Subprocess Credential Scrubbing**: In `ToolRegistry._handle_run_selected_tests`, all sensitive environment variables matching `api_key`, `secret`, `token`, `password`, `auth`, `gemini`, `groq`, `aws`, etc., are stripped from the test execution subprocess.
3. **Loopback Proxy Redirection**: Network isolation is enforced by setting `HTTP_PROXY`, `HTTPS_PROXY`, and `ALL_PROXY` to `http://127.0.0.1:0` with `NO_PROXY=""`, redirecting all external socket connections to a dead loopback port.
4. **Command Whitelist & Timeout**: `PolicyEngine` strictly permits only `pytest` and `python -m unittest` / `python3 -m unittest`. Shell metacharacters (`&&`, `;`, `|`, `>`, `$()`) are blocked. Commands enforce a hard 30-second timeout per execution.

### 6. Untrusted Repository & Diagnostic Defanging
Source code, docstrings, commit messages, and test failure logs may contain prompt injections or adversarial instructions (e.g. `[SYSTEM INSTRUCTIONS]`). `sanitize_untrusted_text()` scrubs delimiters and bounds diagnostics before sending context to the LLM.

---

## 🔧 Troubleshooting & FAQ

### Gemini API Quotas & Credentials
- **Issue**: `google.genai.errors.APIError: 429 Quota Exceeded` or missing key.
- **Resolution**: Verify that `GEMINI_API_KEY` is set in your environment or `.env` file. If running in CI or offline testing, use `--provider mock` or evaluate without `--live` for deterministic, zero-network execution.

### Offline & Mock Execution
- Every core command (`embed`, `search`, `ask`, `propose-code`, `propose-tests`, `agent`, `repair`, `evaluate`) supports `--provider mock` (or offline default in `evaluate`). Mock execution produces reproducible results without external API access.

### Cache Management & Stale Eviction
- If repository files have been reorganized or if you want to regenerate the index from scratch, run:
  ```bash
  codebase-assistant index /path/to/repository --rebuild
  ```
- To specify an external cache directory outside the target repository:
  ```bash
  codebase-assistant index /path/to/repository --cache-dir /tmp/my_cache
  ```

### Windows & WSL Path Handling
- Paths are normalized across Windows (`\`) and Unix (`/`) using `normalize_rel_path()`.
- When referencing UNC or WSL paths (e.g., `\\wsl.localhost\...`), forward slashes or standard UNC format can be used interchangeably.
- Test paths and target files are resolved and verified against the repository boundary to prevent path traversal attacks.

---

## 🧪 Testing & CI/CD

### Running Tests Locally
```bash
# Run full unit test suite (247 tests):
pytest tests/unit -v --tb=short

# Run MCP server integration unit suite (26 tests):
pytest tests/unit/test_mcp_server.py -v

# Run adversarial security test suite (14 tests):
pytest tests/unit/test_repair_adversarial.py -v

# Run format and lint checks:
black --check src tests
isort --check-only src tests
mypy src

# Run golden dataset quality evaluation:
codebase-assistant evaluate ../ai-evaluation --dataset data/golden_dataset.json --fail-under
```

### Test Coverage & Quality Baselines
- **Unit Test Suite**: 247 unit tests across ingestion, vector storage, hybrid retrieval, context assembly, Q&A engine, proposal generation, ReAct coordinator, and MCP server.
- **Adversarial Security Suite**: 14 adversarial tests verifying prompt injection defanging, symlink escape rejection, path traversal blocking, process timeout termination, and atomic rollback failure handling.
- **Antigravity MCP Integration Suite**: 26 integration tests covering tool discovery, fail-closed provider safety, workspace authorization, model confirmation immunity, and multi-file atomic rollback.
- **RAG Evaluation Quality Gate**: 9 out of 9 hard and provisional quality gates passing against the 25-case golden benchmark (`data/golden_dataset.json`).

### GitHub Actions CI
The CI workflow (`.github/workflows/ci.yml`) executes on every push and pull request across Python **3.10, 3.11, 3.12, and 3.13**, verifying formatting, typing, unit tests, adversarial security tests, and RAG quality gates on every commit.

---

## 🔌 Using codebase-assistant from Google Antigravity

`codebase-assistant` v0.2.0 features native integration with Google Antigravity IDE using the **Model Context Protocol (MCP)**. This enables natural-language conversational pair programming directly inside Antigravity without needing to remember CLI commands, flags, or internal tool names.

### 1. Integration Architecture

```mermaid
flowchart LR
    User["Developer (Antigravity Chat)"] <--> AGY["Google Antigravity IDE"]
    AGY <-->|"STDIO (JSON-RPC)"| MCP["codebase-assistant MCP Server"]
    MCP --> Retrieval["HybridRetriever (FAISS + BM25)"]
    MCP --> LLM["Grounded Gemini Q&A Engine"]
    MCP --> Proposal["DiffEngine & Proposal Generators"]
    MCP --> Repair["RepairWorkflowCoordinator (Disposable Sandbox)"]
    MCP -.->|"Read-Only Access"| Repo["Target Codebase"]
```

- **Conversational Interface**: Antigravity acts as the conversational client, parsing developer intent and selecting appropriate tools.
- **Intelligence & Execution Backend**: `codebase-assistant` runs as a local subprocess over standard input/output (STDIO), executing indexing, hybrid search, AST validation, and sandboxed test-driven repair.
- **Clean Transport Isolation**: All JSON-RPC protocol messages flow over `stdin`/`stdout`, while application diagnostic logs route strictly to `stderr`.

---

### 2. Antigravity MCP Configuration (`mcp_config.json`)

To register `codebase-assistant` with Antigravity, add it to your global configuration file (`~/.gemini/config/mcp_config.json`) or workspace configuration:

#### Windows Configuration
```json
{
  "mcpServers": {
    "codebase-assistant": {
      "command": "python",
      "args": [
        "-m",
        "codebase_assistant.mcp_server"
      ],
      "env": {
        "GEMINI_API_KEY": "${GEMINI_API_KEY}"
      }
    }
  }
}
```

#### Linux / WSL Configuration
```json
{
  "mcpServers": {
    "codebase-assistant": {
      "command": "/home/username/codebase_assistant_venv/bin/python",
      "args": [
        "-m",
        "codebase_assistant.mcp_server"
      ],
      "env": {
        "GEMINI_API_KEY": "${GEMINI_API_KEY}"
      }
    }
  }
}
```

> [!TIP]
> Alternatively, you can run `codebase-assistant mcp` or `codebase-assistant-mcp` as the executable command.

---

### 3. Antigravity Skill Installation

The assistant bundles an Antigravity skill located at:
```
.agents/skills/codebase-assistant/SKILL.md
```
Antigravity automatically discovers skills placed in `.agents/skills/` within your workspace, allowing the conversational agent to understand when to invoke search, Q&A, proposal generation, or self-repair based on your prompts.

---

### 4. Registered Semantic Engineering Tools (11 Tools)

The server exposes 11 domain-specific tools categorised into three safety tiers:

| Tool Name | Tier | Parameters | Safety & Boundary Policy |
| :--- | :--- | :--- | :--- |
| `set_active_repository` | Workspace | `repository_path`, `cache_dir`, `writable` | Canonicalizes path via `realpath`; blocks symlink escapes; rejects protected repos from being marked writable; initializes incremental indexing and vector store. |
| `repository_status` | Read-only | *None* | Returns active repository path, chunk count, vector count, model name, execution mode, and capability list. |
| `search_codebase` | Read-only | `query`, `top_k=5`, `mode='hybrid'` | Hybrid vector + BM25 search via Reciprocal Rank Fusion; returns relevant code snippets and line numbers. |
| `read_source_file` | Read-only | `file_path`, `start_line`, `end_line` | Reads line-numbered source code; strictly validates repository boundary and blocks path traversal (`..`). |
| `ask_codebase` | Read-only | `question`, `top_k=5`, `mode='hybrid'`, `budget=8192` | Grounded Q&A backed by Gemini; returns verified line-level source citations and enforces prompt injection boundaries. |
| `propose_code` | Proposal | `task`, `target_file`, `top_k=5`, `budget=8192` | Generates context-aware unified diff with architectural reasoning; deterministic pre-flight AST syntax check; zero disk writes. |
| `propose_tests` | Proposal | `task`, `target_file`, `top_k=5`, `budget=8192` | Targeted unit test proposal with framework auto-detection (`pytest` vs `unittest`) and deterministic AST validation. |
| `inspect_dependencies` | Read-only | `manifest_file=None` | Inspects dependencies (`pyproject.toml`, `requirements.txt`, `package.json`); validates path containment. |
| `investigate_task` | Autonomous | `task`, `max_iterations=5`, `max_steps=10`, `budget=8192` | Autonomous multi-turn ReAct investigation loop with cumulative token and step budgets. |
| `repair_code` | Autonomous | `task`, `test_command`, `max_retries=3`, `max_steps=10`, `budget=8192`, `workflow_timeout=180.0` | End-to-end self-repair executed strictly inside an isolated temporary directory (`tempfile.TemporaryDirectory`); target repository is never touched; proxy blocking and credential stripping enforced. |
| `apply_approved_patch` | Consequential | `patch_content`, `target_workspace=None`, `auth_token=None`, `confirmation=None` | Multi-file diff parsing with all-or-nothing in-memory AST pre-flight dry-runs. Model-generated confirmation strings alone cannot authorize disk writes; unauthenticated requests return pre-flight validated `.diff` files for manual `git apply`. Programmatic disk modification requires a valid host authorization token (`auth_token`). |

---

### 5. Natural-Language Conversational Examples

Once connected, you can talk to the agent naturally:

| User Request | What Antigravity Executes |
| :--- | :--- |
| *"Select the `/path/to/my-repo` repository for analysis."* | Calls `set_active_repository`, indexes files, and synchronizes embeddings. |
| *"What components make up the authentication architecture?"* | Calls `ask_codebase` to provide a grounded response with verified file citations. |
| *"Where is the retry backoff algorithm implemented?"* | Calls `search_codebase` using hybrid vector + BM25 ranking and presents code snippets. |
| *"Add exponential backoff with jitter to the API client."* | Calls `propose_code`, generates an AST-validated unified diff, and presents the reasoning. |
| *"Generate unit tests for `src/auth.py` covering token expiration."* | Calls `propose_tests` with automatic framework detection (`pytest` vs `unittest`). |
| *"Investigate why users get a 403 when updating their profiles."* | Calls `investigate_task` to run an autonomous multi-step ReAct investigation loop. |
| *"Run the test suite, diagnose any failures, and fix the code."* | Calls `repair_code` in an isolated disposable sandbox with network proxy isolation. |
| *"Apply the proposed patch to my development checkout."* | Calls `apply_approved_patch`; returns pre-flight validated patch and manual `git apply` instructions unless a valid host token is presented. |

---

### 6. Security, Authorization & Human Approval Model

- **Read-Only Invariant**: Target codebases (such as `ai-evaluation`) are treated as strictly immutable knowledge sources. Attempts to register them as writable are rejected with `PermissionError`.
- **Model Confirmation Immunity**: In conversational MCP mode, model-generated confirmation strings (e.g., `confirmation="CONFIRM_APPLY"`) can never authorize direct disk modifications. Unauthenticated requests are pre-flight validated and written to external `.diff` files for safe manual application via `git apply`.
- **Cryptographic Authorization Tokens**: Direct programmatic disk modifications are permitted only when presenting a valid, unexpired server-issued one-time token (`auth_token`) via `issue_authorization_token()`.
- **Exact Registered Workspaces**: All target paths are canonicalized via `os.path.realpath()`; symlink escapes, parent traversals (`..`), and absolute path escapes are blocked. Only explicitly registered writable workspaces can be modified.
- **Multi-File Atomic Patching & Rollback**: Multi-file diffs are parsed with `DiffEngine.split_unified_diff`. All files are pre-flight dry-run in memory with AST syntax validation before any disk write. If any write fails midway, all modified files are restored in reverse order to their exact contents and filesystem permissions (`st_mode`), and new files are unlinked. Rollback errors surface explicitly with status `CRITICAL: ROLLBACK_FAILED`.
- **Disposable Sandbox Repair Isolation**: Autonomous repair workflows execute strictly in isolated temporary directories (`tempfile.TemporaryDirectory(prefix="repair_sandbox_")`). Subprocess environments strip all sensitive credentials and API keys (`GEMINI_API_KEY`, tokens, secrets), and network access is blocked via loopback proxy redirection (`HTTP_PROXY="http://127.0.0.1:0"`).
- **Untrusted Diagnostic Defanging**: Repository code, logs, and stack traces are treated as untrusted data and defanged with `sanitize_untrusted_text()`.

---

### 7. Troubleshooting Antigravity MCP Integration

- **Fail-Closed Provider Safety**:
  - By default, if live Gemini mode is enabled but provider initialization fails (missing or invalid API key), `codebase-assistant` fails closed with a `RuntimeError` rather than silently switching to mock mode.
  - To allow fallback to mock mode, launch with `--allow-fallback`.
  - To explicitly run offline without network access, launch with `--mock`.
- **MCP Protocol Errors / JSON Parsing Failures**:
  - Verify that standard logging goes to `stderr`. Never use `print()` statements on `stdout` in server code or imported utilities.
- **Tool Timeout on First Run**:
  - Initial repository indexing can take several seconds for large repositories. In your MCP client configuration, set timeouts to `120s` or higher.
- **Path Resolution across Windows / WSL**:
  - Always pass absolute or workspace-relative paths. Windows UNC paths (`\\wsl.localhost\Ubuntu\...`) and Unix paths (`/home/...`) are automatically normalized.
