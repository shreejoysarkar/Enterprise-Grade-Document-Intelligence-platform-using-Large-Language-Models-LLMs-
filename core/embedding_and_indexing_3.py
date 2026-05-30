"""
Embedding & Hybrid-Search Indexing Module (Phase 3)

This module bridges the gap between chunked documents and retrieval-ready vectors.
It performs three key operations:

1. **Dense Embeddings** — Uses a HuggingFace sentence-transformer model to convert
   each text chunk into a dense vector representation, capturing semantic meaning.

2. **Sparse Vectors (BM25)** — Uses the pinecone-text BM25 encoder to produce
   sparse vector representations, capturing lexical/keyword signals.

3. **Pinecone Hybrid Index** — Upserts both dense and sparse vectors into a
   Pinecone Serverless index configured with `dotproduct` metric, enabling
   hybrid search that combines the strengths of semantic and keyword retrieval.

Input:  JSONL chunk files from  Data/chunks/
Output: Vectors indexed in Pinecone, ready for hybrid retrieval.
"""

import json
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from sentence_transformers import SentenceTransformer

from utils.config import get_settings
from utils.logger import get_logger, setup_logging

# ── Bootstrap ──────────────────────────────────────────────────────────────────
load_dotenv()
setup_logging()
logger = get_logger(__name__)


# ── Helper: load JSONL chunks ──────────────────────────────────────────────────
def _load_chunks_from_jsonl(file_path: Path) -> list[dict[str, Any]]:
    """Load all chunk dicts from a JSONL file.

    Each line is a JSON object with at least `id` and `text` keys.
    """
    chunks: list[dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Skipping malformed JSON on line %d of %s: %s",
                    line_number, file_path.name, exc,
                )
    return chunks


class HybridSearchIndexer:
    """Generates dense + sparse vectors and upserts them to a Pinecone hybrid index.

    Workflow
    -------
    1. Initialize HuggingFace dense encoder and BM25 sparse encoder.
    2. Create (or connect to) a Pinecone Serverless index.
    3. Load JSONL chunks → encode → batch-upsert.
    4. Expose a `hybrid_search` method for downstream retrieval.
    """

    def __init__(
        self,
        chunks_dir: Path = Path("Data") / "chunks",
        embedding_model_name: str | None = None,
        pinecone_index_name: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        """Initialize the indexer with models and Pinecone client.

        Args:
            chunks_dir: Directory containing JSONL chunk files.
            embedding_model_name: Override for the dense embedding model name.
            pinecone_index_name: Override for the target Pinecone index.
            batch_size: Override for upsert batch size.
        """
        self.settings = get_settings()
        self.chunks_dir = Path(chunks_dir)

        # ── Dense encoder (HuggingFace) ────────────────────────────────────
        self.embedding_model_name = (
            embedding_model_name or self.settings.embedding_model
        )
        logger.info("Loading dense encoder: %s", self.embedding_model_name)
        self.dense_encoder = SentenceTransformer(self.embedding_model_name)
        self.embedding_dim = self.dense_encoder.get_sentence_embedding_dimension()
        logger.info(
            "Dense encoder ready — dimension=%d", self.embedding_dim
        )

        # ── Sparse encoder (BM25) ─────────────────────────────────────────
        logger.info("Fitting BM25 sparse encoder on chunk corpus…")
        self.sparse_encoder = self._fit_bm25()
        logger.info("BM25 encoder fitted.")

        # ── Pinecone client & index ────────────────────────────────────────
        self.index_name = pinecone_index_name or self.settings.pinecone_index_name
        self.batch_size = batch_size or self.settings.embedding_batch_size
        self.namespace = self.settings.pinecone_namespace

        self.pc = Pinecone(api_key=self.settings.PINECONE_API_KEY)
        self.index = self._get_or_create_index()
        logger.info("Connected to Pinecone index: %s", self.index_name)

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _collect_all_texts(self) -> list[str]:
        """Collect all chunk texts from every JSONL file in chunks_dir."""
        texts: list[str] = []
        for jsonl_file in sorted(self.chunks_dir.glob("*.jsonl")):
            for chunk in _load_chunks_from_jsonl(jsonl_file):
                text = chunk.get("text", "")
                if text.strip():
                    texts.append(text)
        logger.info("Collected %d texts for BM25 fitting.", len(texts))
        return texts

    def _fit_bm25(self) -> BM25Encoder:
        """Fit a BM25 encoder on the full chunk corpus."""
        corpus = self._collect_all_texts()
        if not corpus:
            logger.warning(
                "No texts found for BM25 fitting — encoder will be default-initialised."
            )
            return BM25Encoder.default()

        encoder = BM25Encoder()
        encoder.fit(corpus)
        return encoder

    def _get_or_create_index(self):
        """Return a Pinecone Index handle, creating the index if necessary."""
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]

        if self.index_name not in existing_indexes:
            logger.info(
                "Creating Pinecone index '%s' (dim=%d, metric=%s)…",
                self.index_name, self.embedding_dim, self.settings.pinecone_metric,
            )
            self.pc.create_index(
                name=self.index_name,
                dimension=self.embedding_dim,
                metric=self.settings.pinecone_metric,
                spec=ServerlessSpec(
                    cloud=self.settings.pinecone_cloud,
                    region=self.settings.pinecone_region,
                ),
            )
            # Wait until the index is ready
            self._wait_for_index_ready()
        else:
            logger.info("Pinecone index '%s' already exists.", self.index_name)

        return self.pc.Index(self.index_name)

    def _wait_for_index_ready(self, timeout: int = 120) -> None:
        """Block until the Pinecone index reports as ready."""
        start = time.time()
        while time.time() - start < timeout:
            desc = self.pc.describe_index(self.index_name)
            if desc.status.get("ready", False):
                logger.info("Index '%s' is ready.", self.index_name)
                return
            logger.info("Waiting for index '%s' to become ready…", self.index_name)
            time.sleep(5)
        raise TimeoutError(
            f"Pinecone index '{self.index_name}' was not ready after {timeout}s."
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Encoding helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _encode_dense(self, texts: list[str]) -> list[list[float]]:
        """Generate dense embeddings for a list of texts."""
        embeddings = self.dense_encoder.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  # Required for dotproduct metric
        )
        return embeddings.tolist()

    def _encode_sparse(self, texts: list[str]) -> list[dict]:
        """Generate sparse BM25 vectors for a list of texts.

        Returns a list of dicts with `indices` and `values` keys
        matching Pinecone's sparse vector format.
        """
        sparse_vectors = self.sparse_encoder.encode_documents(texts)
        return sparse_vectors

    # ──────────────────────────────────────────────────────────────────────────
    # Metadata builder
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
        """Flatten chunk data into Pinecone-compatible metadata.

        Pinecone metadata values must be strings, numbers, booleans, or
        lists of strings. We store the raw text for retrieval and carry
        forward any existing metadata.
        """
        meta: dict[str, Any] = {
            "text": chunk.get("text", ""),
            "source_file": "",
            "chunk_type": "",
            "token_count": chunk.get("token_count", 0),
            "start_index": chunk.get("start_index", 0),
            "end_index": chunk.get("end_index", 0),
        }

        chunk_meta = chunk.get("metadata", {}) or {}
        meta["source_file"] = chunk_meta.get("source_file", chunk_meta.get("filename", ""))
        meta["chunk_type"] = chunk_meta.get("type", "text")

        return meta

    # ──────────────────────────────────────────────────────────────────────────
    # Core upsert logic
    # ──────────────────────────────────────────────────────────────────────────

    def index_file(self, file_path: Path) -> int:
        """Process a single JSONL chunk file: encode and upsert to Pinecone.

        Args:
            file_path: Path to a JSONL file.

        Returns:
            Number of vectors upserted.
        """
        chunks = _load_chunks_from_jsonl(file_path)
        if not chunks:
            logger.warning("No chunks found in %s — skipping.", file_path.name)
            return 0

        logger.info(
            "Indexing %d chunks from %s…", len(chunks), file_path.name,
        )

        # Extract texts and IDs
        ids = [c["id"] for c in chunks]
        texts = [c.get("text", "") for c in chunks]

        # Encode
        logger.info("Generating dense embeddings…")
        dense_vectors = self._encode_dense(texts)

        logger.info("Generating sparse (BM25) vectors…")
        sparse_vectors = self._encode_sparse(texts)

        # Build metadata
        metadata_list = [self._build_metadata(c) for c in chunks]

        # Batch upsert
        total_upserted = 0
        for i in range(0, len(ids), self.batch_size):
            batch_end = min(i + self.batch_size, len(ids))

            upsert_batch = []
            for j in range(i, batch_end):
                vector_record = {
                    "id": ids[j],
                    "values": dense_vectors[j],
                    "sparse_values": sparse_vectors[j],
                    "metadata": metadata_list[j],
                }
                upsert_batch.append(vector_record)

            self.index.upsert(
                vectors=upsert_batch,
                namespace=self.namespace,
            )
            total_upserted += len(upsert_batch)
            logger.info(
                "  Upserted batch %d–%d (%d/%d)",
                i, batch_end - 1, total_upserted, len(ids),
            )

        logger.info(
            "✅ Finished indexing %s — %d vectors upserted.",
            file_path.name, total_upserted,
        )
        return total_upserted

    def index_all(self) -> None:
        """Index all JSONL files in the chunks directory."""
        if not self.chunks_dir.exists():
            logger.error("Chunks directory not found: %s", self.chunks_dir)
            return

        jsonl_files = sorted(self.chunks_dir.glob("*.jsonl"))
        if not jsonl_files:
            logger.warning("No JSONL files found in %s", self.chunks_dir)
            return

        logger.info(
            "Starting hybrid indexing of %d file(s) from %s",
            len(jsonl_files), self.chunks_dir,
        )

        total_vectors = 0
        successful = 0
        failed = 0

        for jsonl_file in jsonl_files:
            try:
                count = self.index_file(jsonl_file)
                total_vectors += count
                successful += 1
            except Exception as exc:
                logger.error(
                    "❌ Error indexing %s: %s", jsonl_file.name, exc, exc_info=True,
                )
                failed += 1

        # Print index stats
        try:
            stats = self.index.describe_index_stats()
            logger.info("📊 Index stats: %s", stats)
        except Exception:
            pass

        logger.info("🎉 Hybrid indexing complete!")
        logger.info(
            "✅ Successful: %d | ❌ Failed: %d | 📦 Total vectors: %d",
            successful, failed, total_vectors,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Hybrid search (for downstream retrieval)
    # ──────────────────────────────────────────────────────────────────────────

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.7,
        filter_dict: dict | None = None,
    ) -> list[dict[str, Any]]:
        """Perform a hybrid (dense + sparse) search on the Pinecone index.

        The `alpha` parameter controls the weighting between dense and sparse:
            alpha = 1.0 → pure semantic (dense only)
            alpha = 0.0 → pure keyword  (sparse only)
            alpha = 0.7 → 70% semantic + 30% keyword  (default, recommended)

        Args:
            query: The search query string.
            top_k: Number of top results to return.
            alpha: Weight for dense vs sparse blending.
            filter_dict: Optional Pinecone metadata filter.

        Returns:
            List of result dicts with `id`, `score`, and `metadata`.
        """
        # Dense query vector
        dense_query = self.dense_encoder.encode(
            [query], normalize_embeddings=True
        ).tolist()[0]

        # Sparse query vector
        sparse_query = self.sparse_encoder.encode_queries([query])[0]

        # Scale vectors by alpha for hybrid blending
        scaled_dense = [v * alpha for v in dense_query]
        scaled_sparse = {
            "indices": sparse_query["indices"],
            "values": [v * (1 - alpha) for v in sparse_query["values"]],
        }

        # Execute hybrid query
        query_params = {
            "namespace": self.namespace,
            "top_k": top_k,
            "vector": scaled_dense,
            "sparse_vector": scaled_sparse,
            "include_metadata": True,
        }
        if filter_dict:
            query_params["filter"] = filter_dict

        results = self.index.query(**query_params)

        # Format results
        formatted: list[dict[str, Any]] = []
        for match in results.get("matches", []):
            formatted.append({
                "id": match["id"],
                "score": match["score"],
                "metadata": match.get("metadata", {}),
            })

        logger.info(
            "Hybrid search for '%s' returned %d results (alpha=%.2f).",
            query[:60], len(formatted), alpha,
        )
        return formatted



def main():
    indexer = HybridSearchIndexer()
    indexer.index_all()


if __name__ == "__main__":
    main()
