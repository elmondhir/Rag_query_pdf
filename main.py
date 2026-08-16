import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from dotenv import load_dotenv
import uuid
from data_loader import load_and_chunk_pdf
from openrouter import OPENROUTER_EMBED_DIM, chat_completion, embed_texts
from vector_db import QdrantStorage
from custom_types import (
    RAGSearchResult,
    RAGUpserResult,
    RAGChunkAndSrc,
)

load_dotenv()

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer(),
)


@inngest_client.create_function(
    fn_id="RAG: Ingest PDF", trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)
async def rag_ingest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpserResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        vecs = embed_texts(chunks, input_type="passage")
        ids = [
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}"))
            for i in range(len(chunks))
        ]
        payloads = [
            {"source": source_id, "text": chunks[i]} for i in range(len(chunks))
        ]
        QdrantStorage(dim=OPENROUTER_EMBED_DIM).upsert(ids, vecs, payloads)

        return RAGUpserResult(ingested=len(chunks))

    chunks_and_src = await ctx.step.run(
        "load-and-chunk", lambda: _load(ctx), output_type=RAGChunkAndSrc
    )
    ingested = await ctx.step.run(
        "embed-and-upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpserResult
    )

    return ingested.model_dump()


@inngest_client.create_function(
    fn_id="RAG: Query PDF", trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int = 5):
        query_vector = embed_texts([question], input_type="query")[0]
        store = QdrantStorage(dim=OPENROUTER_EMBED_DIM)
        found = store.search(query_vector, top_k)

        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    found = await ctx.step.run(
        "embed-and-search",
        lambda: _search(question, top_k),
        output_type=RAGSearchResult,
    )
    context_block = "\n\n".join(f"- {c}" for c in found.contexts)

    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )

    answer = await ctx.step.run(
        "llm-answer",
        lambda: chat_completion(
            [
                {
                    "role": "system",
                    "content": "You answer questions using only the provided context.",
                },
                {"role": "user", "content": user_content},
            ]
        ),
    )

    return {
        "answer": answer,
        "sources": found.sources,
        "num_contexts": len(found.contexts),
    }


app = FastAPI()


inngest.fast_api.serve(
    app, inngest_client, functions=[rag_ingest_pdf, rag_query_pdf_ai]
)
