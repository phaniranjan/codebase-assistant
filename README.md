# Codebase Assistant — RAG-Powered Coding Agent

`codebase-assistant` is an AI-powered coding assistant that ingests an existing software repository, understands its architecture, tracks AST symbols, retrieves relevant source code via hybrid search, and coordinates with Gemini to answer questions, generate context-aware code proposals (as unified diffs), generate tests, and assist with debugging.

---

## 🏗️ Architecture Blueprint & Implementation Status

The diagram below illustrates the full target architecture across all phases. Components implemented in **Milestone 1** are marked with `[Implemented]`; downstream capabilities are marked with `[Planned Phase N]`.

```mermaid
flowchart TD
    %% FLOW 1: Indexing Flow
    subgraph Flow1["1. Repository Ingestion & Indexing Flow"]
        Repo["Target Codebase\n(Python source, Markdown docs, Config)"] --> Scanner["Repository Scanner [Implemented]\n(Path filters, Directory pruning, Secret exclusion)"]
        Scanner --> Parser["AST & Structure Parser [Implemented]\n(Classes, Methods, Functions, Imports, Docs)"]
        Parser --> Chunker["Logical Chunker [Implemented]\n(AST boundaries, Enclosure scope, Line numbers)"]
        Chunker --> Indexer["Incremental Indexer [Implemented]\n(SHA-256 change detection, Stale eviction, Manifest)"]
        Indexer --> LocalIndex[("Local Knowledge Base [Implemented]\n(.codebase_assistant/ manifest.json + chunks.json)")]
        
        Chunker -.->|Future Hook| EmbedModel["Embedding Provider [Planned Phase 3]\n(Gemini Embeddings + Local Vector Index)"]
        EmbedModel -.-> LocalIndex
    end

    %% FLOW 2: Runtime Flow
    subgraph Flow2["2. Runtime Execution Flow [Planned Phases 4-6]"]
        DevRequest(["Developer Request\n(Question, Code Request, Debug Trace)"]) --> AgentCore["AgentCore [Planned Phase 4]\n(Workflow Coordinator)"]
        
        AgentCore -.-> Retriever["Hybrid Retriever [Planned Phase 3]\n(FAISS Dense Vector + BM25/Symbol Search)"]
        LocalIndex -.->|Read Index| Retriever
        
        Retriever -.-> ContextAssembler["Context Assembler [Planned Phase 3]\n(Neighbor expansion, Token budget)"]
        ContextAssembler -.-> PromptBuilder["Prompt Builder [Planned Phase 4]\n(Strict grounding, Prompt injection defense)"]
        PromptBuilder -.-> GeminiClient["Gemini Client [Planned Phase 4]\n(google-genai SDK)"]
        GeminiClient -.-> AgentCore
        
        %% Tool Execution Loop
        AgentCore -.->|Requires Tool| ToolManager["Tool Manager [Planned Phase 6]\n(Tool Dispatcher)"]
        ToolManager -.-> PolicyCheck{"Policy Engine [Planned Phase 6]\n(Command Allowlist & Path Restrict)"}
        PolicyCheck -.->|Policy Denied| PolicyRejection["Policy Rejection\n(Outright deny with security log)"] -.-> AgentCore
        PolicyCheck -.->|Policy Passed| PermCheck{"Permission Check [Planned Phase 6]\n(Sensitive Operation?)"}
        PermCheck -.->|Read-only: search, read_file| ToolExec["Tool Execution\n(Isolated execution)"]
        PermCheck -.->|Consequential: test exec, file patch| UserApproval{"User Authorization\n(Approve execution?)"}
        UserApproval -.->|Approved| ToolExec
        UserApproval -.->|Denied| AgentCore
        ToolExec -.-> AgentCore
        
        AgentCore -.-> DevResponse(["Developer Output [Planned Phase 4/5]\n(Grounded Answer / Proposed Diff)"])
    end

    %% FLOW 3: Offline Evaluation Flow
    subgraph Flow3["3. Offline Evaluation Loop [Planned Phase 7]"]
        GoldenData[("Golden Dataset\n(evals/golden_questions.json)")] -.-> EvalRunner["Evaluation Runner [Planned Phase 7]\n(Pytest / CLI / CI harness)"]
        EvalRunner -.->|Benchmark Retrieval| Retriever
        EvalRunner -.->|Benchmark End-to-End| AgentCore
        EvalRunner -.-> EvalMetrics["Quality Metrics [Planned Phase 7]\n(Precision@k, Recall@k, Faithfulness, Code Validity)"]
        EvalMetrics -.-> QualityReport["Quality Report & Regression Check"]
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
| **Local Cache Storage** | ✅ Implemented | Local `.codebase_assistant/` manifest and chunks storage. |
| **CLI (`scan`, `index`, `inspect-chunks`)** | ✅ Implemented | Terminal commands for repository inspection and indexing. |
| **Embeddings & FAISS Vector Store** | ⏳ Planned (Phase 3) | Dense embedding generation and local vector similarity search. |
| **BM25 & Hybrid Retriever** | ⏳ Planned (Phase 3) | Exact identifier lookup and Reciprocal Rank Fusion (RRF). |
| **Gemini Integration & Q&A Agent** | ⏳ Planned (Phase 4) | Grounded prompt building and citation-backed repository question answering. |
| **Code & Test Proposals** | ⏳ Planned (Phase 5) | Unified git diff generation and pytest test generation. |
| **Controlled Tools & Execution Sandbox** | ⏳ Planned (Phase 6) | Policy engine, permissions, and isolated command execution. |
| **Evaluation Suite & Benchmarks** | ⏳ Planned (Phase 7) | Golden dataset benchmarks for retrieval and groundedness. |

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
- `GEMINI_API_KEY`: API key for Gemini LLM generation and embeddings (required in Phase 3+).

---

## 💻 CLI Usage

### 1. Scan a Repository
Scan a local repository to inspect eligible files, excluded files, and git revision without modifying the target:
```bash
python -m codebase_assistant.cli scan /path/to/repository
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
python -m codebase_assistant.cli inspect-chunks --file rag_pipeline.py --limit 10
```
Filter by symbol name:
```bash
python -m codebase_assistant.cli inspect-chunks --symbol BM25Retriever
```

---

## 🔒 Security, Privacy & Limits of Secret Detection

### Security Guardrails
1. **Read-Only Target Repository**: The assistant treats target codebases as strictly read-only knowledge sources.
2. **Symlink Containment**: Symlinks pointing outside the repository boundary are blocked (`[SYMLINK_OUTSIDE]`) to prevent path traversal attacks.
3. **Repository Identity Validation**: The indexer binds cache manifests to the repository's identity and path, preventing cross-repository index corruption.
4. **Stale Eviction Integrity**: Modified, deleted, renamed, or unparseable files have their old chunks evicted immediately, preventing stale chunks from appearing current.

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
