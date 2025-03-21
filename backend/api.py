import os
import asyncio
import zipfile
from io import BytesIO
from pathlib import Path
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, HTTPException, status, BackgroundTasks, Query
from fastapi.responses import FileResponse, StreamingResponse
import logging

from llm_manager import LLMManager
from rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)
from pydantic import BaseModel, Field
from typing import Literal, Optional

from .redis_manager import (
    send_to_redis_stream, 
    receive_llm_response
)
from pipelines import (
    store_uploaded_pdf,
    get_pdf_content,
    standardize_docling,
    standardize_markitdown,
    html_to_md_docling,
    get_job_name,
    pdf_to_md_docling,
    clean_temp_files,
    pdf_to_md_enterprise,
    html_to_md_enterprise,
)

load_dotenv()
app = FastAPI()

# LLM Service instance
llm_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global llm_service
    from llm_service import LLMService

    llm_service = LLMService()
    asyncio.create_task(llm_service.start())
    logger.info("LLM Service started")

    yield

    # Shutdown
    logger.info("Shutting down LLM Service")


app = FastAPI(lifespan=lifespan)


class URLRequest(BaseModel):
    url: str


class PDFSelection(BaseModel):
    pdf_id: str = Field(..., min_length=8, max_length=24)


class PDFUploadResponse(BaseModel):
    pdf_id: str
    status: str
    message: Optional[str] = None


class SummaryRequest(BaseModel):
    pdf_id: str = Field(..., min_length=8, max_length=100)
    summary_length: int = Field(200, gt=50, lt=1000)
    max_tokens: int = Field(500, gt=100, lt=2000)
    model: str = Field(
        "gemini/gemini-2.0-flash", description="LLM model to use for summary generation"
    )


class QuestionRequest(BaseModel):
    pdf_id: str
    question: str = Field(..., min_length=10)
    max_tokens: int = Field(500, gt=100, lt=2000)
    model: str = Field(
        "gemini/gemini-2.0-flash", description="LLM model to use for question answering"
    )


active_rag_pipelines = {}

@app.post("/processurl/", status_code=status.HTTP_200_OK)
async def process_url(
    background_tasks: BackgroundTasks,
    request: URLRequest,
    include_markdown: bool = Query(False),
    include_images: bool = Query(False),
    include_tables: bool = Query(False),
):
    if not any([include_markdown, include_images, include_tables]):
        raise HTTPException(
            status_code=400, detail="At least one output type must be selected"
        )
    try:
        url = request.url
        job_name = get_job_name()
        result = html_to_md_docling(url, job_name)
        background_tasks.add_task(my_background_task)

        if include_images or include_tables:  # images or tables are requested
            flag, zip_buffer, messages = create_zip_archive(
                result, include_markdown, include_images, include_tables
            )
            if flag:
                return StreamingResponse(
                    zip_buffer,
                    media_type="application/zip",
                    headers={
                        "Content-Disposition": f"attachment; filename={job_name}.zip"
                    },
                )
            else:
                raise HTTPException(status_code=500, detail=messages)
        else:
            if not result["markdown"]:
                raise HTTPException(
                    status_code=500,
                    detail="Markdown couldn't be generated. Maybe webpage has no data.",
                )
            return FileResponse(
                result["markdown"],
                media_type="application/octet-stream",
                headers={"Content-Disposition": f"attachment; filename={job_name}.md"},
                filename=f"{job_name}.md",
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/processpdf/", status_code=status.HTTP_200_OK)
async def process_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile,
    include_markdown: bool = Query(False),
    include_images: bool = Query(False),
    include_tables: bool = Query(False),
):
    if not any([include_markdown, include_images, include_tables]):
        raise HTTPException(
            status_code=400, detail="At least one output type must be selected"
        )

    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File must be a PDF")

    try:
        background_tasks.add_task(my_background_task)
        contents = await file.read()
        output = Path("./temp_processing/output/pdf")
        os.makedirs(output, exist_ok=True)
        job_name = get_job_name()

        file_path = output / f"{job_name}.pdf"
        with open(file_path, "wb") as f:
            f.write(contents)
            await file.close()

        result = pdf_to_md_docling(file_path, job_name)

        if include_images or include_tables:  # images or tables are requested
            flag, zip_buffer, messages = create_zip_archive(
                result, include_markdown, include_images, include_tables
            )
            if flag:
                return StreamingResponse(
                    zip_buffer,
                    media_type="application/zip",
                    headers={
                        "Content-Disposition": f"attachment; filename={job_name}.zip"
                    },
                )
            else:
                raise HTTPException(status_code=500, detail=messages)

        else:
            if not result["markdown"] or not os.path.exists(result["markdown"]):
                raise HTTPException(
                    status_code=500,
                    detail="Markdown couldn't be generated. Maybe webpage has no data.",
                )
            return FileResponse(
                result["markdown"],
                media_type="application/octet-stream",
                filename=f"{job_name}.md",
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await file.close()


@app.post("/standardizedoclingpdf/", status_code=status.HTTP_200_OK)
async def standardizedoclingpdf(file: UploadFile, background_tasks: BackgroundTasks):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File must be a PDF")

    background_tasks.add_task(my_background_task)
    contents = await file.read()
    output = Path("./temp_processing/output/pdf")
    os.makedirs(output, exist_ok=True)
    job_name = get_job_name()
    try:
        file_path = output / f"{job_name}.pdf"
        with open(file_path, "wb") as f:
            f.write(contents)
            await file.close()
        standardized_output = standardize_docling(str(file_path), job_name)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return FileResponse(
        standardized_output,
        media_type="application/octet-stream",
        filename=f"{file.filename}.md",
    )


@app.post("/standardizedoclingurl/", status_code=status.HTTP_200_OK)
async def standardizedoclingurl(request: URLRequest, background_tasks: BackgroundTasks):
    try:
        url = request.url
        job_name = get_job_name()
        background_tasks.add_task(my_background_task)

        standardized_output = standardize_docling(url, job_name)

        if standardized_output == -1:
            raise HTTPException(
                status_code=500,
                detail="Markdown couldn't be generated. Maybe webpage has no data.",
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return FileResponse(
        standardized_output,
        media_type="application/octet-stream",
        filename=f"{job_name}.md",
    )


@app.post("/standardizemarkitdownpdf/", status_code=status.HTTP_200_OK)
async def standardizemarkitdownpdf(file: UploadFile, background_tasks: BackgroundTasks):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File must be a PDF")

    background_tasks.add_task(my_background_task)
    contents = await file.read()
    output = Path("./temp_processing/output/pdf")
    os.makedirs(output, exist_ok=True)
    job_name = get_job_name()
    try:
        file_path = output / f"{job_name}.pdf"
        with open(file_path, "wb") as f:
            f.write(contents)
            await file.close()
        standardized_output = standardize_markitdown(str(file_path), job_name)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return FileResponse(
        standardized_output,
        media_type="application/octet-stream",
        filename=f"{file.filename}.md",
    )


@app.post("/standardizemarkitdownurl/", status_code=status.HTTP_200_OK)
async def standardizemarkitdownurl(
    request: URLRequest, background_tasks: BackgroundTasks
):
    try:
        url = request.url
        job_name = get_job_name()
        background_tasks.add_task(my_background_task)

        standardized_output = standardize_markitdown(url, job_name)

        if standardized_output == -1:
            raise HTTPException(
                status_code=500,
                detail="Markdown couldn't be generated. Maybe webpage has no data.",
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return FileResponse(
        standardized_output,
        media_type="application/octet-stream",
        filename=f"{job_name}.md",
    )


@app.post("/processpdfenterprise/", status_code=status.HTTP_200_OK)
async def process_pdf_enterprise(
    background_tasks: BackgroundTasks,
    file: UploadFile,
    include_markdown: bool = Query(False),
    include_images: bool = Query(False),
    include_tables: bool = Query(False),
):
    if not any([include_markdown, include_images, include_tables]):
        raise HTTPException(
            status_code=400, detail="At least one output type must be selected"
        )

    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File must be a PDF")
    try:
        background_tasks.add_task(my_background_task)
        contents = await file.read()
        output = Path("./temp_processing/output/pdf")
        os.makedirs(output, exist_ok=True)
        job_name = get_job_name()

        file_path = output / f"{job_name}.pdf"
        with open(file_path, "wb") as f:
            f.write(contents)
            await file.close()

        result = pdf_to_md_enterprise(file_path, job_name)

        if include_images or include_tables:  # images or tables are requested
            flag, zip_buffer, messages = create_zip_archive(
                result, include_markdown, include_images, include_tables
            )
            if flag:
                return StreamingResponse(
                    zip_buffer,
                    media_type="application/zip",
                    headers={
                        "Content-Disposition": f"attachment; filename={job_name}.zip"
                    },
                )
            else:
                raise HTTPException(status_code=500, detail=messages)
        else:
            if not result["markdown"] or not os.path.exists(result["markdown"]):
                raise HTTPException(
                    status_code=500,
                    detail="Markdown couldn't be generated. Maybe webpage has no data.",
                )
            return FileResponse(
                result["markdown"],
                media_type="application/octet-stream",
                filename=f"{job_name}.md",
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await file.close()


@app.post("/processurlenterprise/", status_code=status.HTTP_200_OK)
async def process_url_enterprise(
    background_tasks: BackgroundTasks,
    request: URLRequest,
    include_markdown: bool = Query(False),
    include_images: bool = Query(False),
    include_tables: bool = Query(False),
):
    if not any([include_markdown, include_images, include_tables]):
        raise HTTPException(
            status_code=400, detail="At least one output type must be selected"
        )
    try:
        url = request.url
        job_name = get_job_name()
        result = html_to_md_enterprise(url, job_name)
        background_tasks.add_task(my_background_task)

        if include_images or include_tables:  # images or tables are requested
            flag, zip_buffer, messages = create_zip_archive(
                result, include_markdown, include_images, include_tables
            )
            if flag:
                return StreamingResponse(
                    zip_buffer,
                    media_type="application/zip",
                    headers={
                        "Content-Disposition": f"attachment; filename={job_name}.zip"
                    },
                )
            else:
                raise HTTPException(status_code=500, detail=messages)
        else:
            if not result["markdown"]:
                raise HTTPException(
                    status_code=500,
                    detail="Markdown couldn't be generated. Maybe webpage has no data.",
                )
            return FileResponse(
                result["markdown"],
                media_type="application/octet-stream",
                headers={"Content-Disposition": f"attachment; filename={job_name}.md"},
                filename=f"{job_name}.md",
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/select_pdfcontent", status_code=status.HTTP_200_OK, tags=["Assignment 4"])
async def select_pdf_content(request: PDFSelection):
    try:

        content = get_pdf_content(request.pdf_id)
        return {
            "pdf_id": request.pdf_id,
            "content": content[:5000],  # Return first 5k chars for preview
            "metadata": {"pages": len(content.split("\n\f"))},
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload_pdf", status_code=status.HTTP_201_CREATED, tags=["Assignment 4"])
async def upload_pdf(
    file: UploadFile,
    parser: Literal["docling", "mistral"],
    chunking_strategy: Literal["recursive", "semantic", "fixed"],
    vector_store: Literal["chroma", "pinecone", "naive"],
):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type")

    try:
        contents = await file.read()
        pdf_id = store_uploaded_pdf(contents, parser)
        contents = get_pdf_content(pdf_id)
        rag_pipeline = RAGPipeline(
            pdf_id=pdf_id,
            text=contents,
            chunking_strategy=chunking_strategy,
            vector_store=vector_store,
        )
        rag_pipeline.process()
        logger.info(f"PDF {pdf_id} processed")
        active_rag_pipelines[pdf_id] = rag_pipeline

        # Send text content to stream for processing
        # try:

        # await send_to_redis_stream(
        #     "pdf_content",
        #     {
        #         "pdf_id": pdf_id,
        #         "content": contents,
        #     },
        # )
        #     logger.info(f"PDF {pdf_id} content sent to stream")
        # except Exception as e:
        #     logger.error(f"Failed to process PDF: {str(e)}")
        #     raise
        return PDFUploadResponse(
            pdf_id=pdf_id, status="success", message=f"PDF stored with ID: {pdf_id}"
        )
    except Exception as e:
        logger.error(f"Failed to process PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await file.close()


@app.post("/summarize/", response_model=dict, tags=["Assignment 4"])
async def generate_summary(request: SummaryRequest):
    # logger = logging.getLogger(__name__)

    try:

        # Send request to Redis stream and wait for response
        try:
            content = get_pdf_content(request.pdf_id)
            logger.info(
                f"Sending PDF content to Redis stream for {request.pdf_id} content: {content}"
            )
            await send_to_redis_stream(
                "pdf_content",
                {
                    "pdf_id": request.pdf_id,
                    "content": content,
                },
            )
            logger.info(f"PDF content sent to Redis stream for {request.pdf_id}")
            await send_to_redis_stream(
                "llm_requests",
                {
                    "type": "summary",
                    "pdf_id": request.pdf_id,
                    "max_tokens": request.max_tokens,
                    "model": request.model,
                },
            )
            logger.info(f"Summary request sent to Redis stream for {request.pdf_id}")
            response = await receive_llm_response()
            logger.info(
                f"Summary response received from Redis stream for {request.pdf_id}"
            )
            if not response:
                raise HTTPException(status_code=408, detail="LLM response timeout")

            # Extract usage metrics from response
            usage_metrics = response.get("usage", {})
            if not usage_metrics:
                logger.warning("No usage metrics found in response")

            return {
                "summary": response.get("content", ""),
                "usage_metrics": {
                    "input_tokens": usage_metrics.get("input_tokens", 0),
                    "output_tokens": usage_metrics.get("output_tokens", 0),
                    "total_tokens": usage_metrics.get("total_tokens", 0),
                    "cost": usage_metrics.get("cost", 0.0),
                },
            }

        except Exception as e:
            logger.error(f"LLM processing failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Summary generation failed")

    except FileNotFoundError as e:
        logger.error(f"PDF content lookup failed: {str(e)}")
        raise HTTPException(status_code=404, detail="PDF content not found")


@app.post("/ask_question", status_code=status.HTTP_200_OK, tags=["Assignment 4"])
async def answer_pdf_question(request: QuestionRequest):
    # logger = logging.getLogger(__name__)
    if request.pdf_id not in active_rag_pipelines:
        raise HTTPException(status_code=404, detail="PDF not found")
    try:
        rag_pipeline: RAGPipeline = active_rag_pipelines[request.pdf_id]
        relevant_chunks = rag_pipeline.get_relevant_chunks(request.question, 5)
        logger.info(f"PDF {request.pdf_id} retrieved {len(relevant_chunks)} chunks")
        context = "\n\n".join(relevant_chunks)
        logger.info(f"PDF {request.pdf_id} context: {context}")
        # await send_to_redis_stream(
        #     "pdf_content",
        #     {
        #         "pdf_id": request.pdf_id,
        #         "content": context,
        #     },
        # )
        # logger.info(f"PDF {request.pdf_id} context sent to stream")
        # await send_to_redis_stream(
        #     "llm_requests",
        #     {
        #         "type": "question",
        #         "pdf_id": request.pdf_id,
        #         "question": request.question,
        #         "max_tokens": request.max_tokens,
        #     },
        # )
        # logger.info(f"Question {request.question} sent to stream")
        # response = await receive_llm_response()
        system_message = """You are a helpful assistant that provides accurate information based on the given context. 
                            If the context doesn't contain relevant information to answer the question, acknowledge that and provide general information if possible.
                            Always cite your sources by referring to the source numbers provided in brackets. Do not make up information."""

        # Define the user message with query and context
        user_message = f"""Question: {request.question}
        
        Context information:
        {context}
        
        Please answer the question based on the context information provided."""
        # prompt = (
        #     "Context:\n{context}\n\nQuestion: {question}\n\n"
        #     "Requirements:"
        #     "- Answer must be factual based on context"
        #     "- Maximum {max_tokens} tokens"
        #     '- If unsure, state "I cannot determine from the provided content'
        # ).format(
        #     context=context, question=request.question, max_tokens=request.max_tokens
        # )
        prompt = {
            "system_message": system_message,
            "user_message": user_message,
        }

        llm_manager = LLMManager()
        content, usage_metrics = await llm_manager.get_llm_response(
            prompt, request.model
        )
        # if not response:
        #     raise HTTPException(status_code=408, detail="LLM response timeout")

        # Extract content from response
        # content = response.get("content")
        # if not content:
        #     logger.error(f"Missing content in response: {response}")
        #     raise HTTPException(
        #         status_code=500, detail="Invalid response format from LLM service"
        #     )

        # Check response status
        # if response.get("status") != "success":
        #     logger.error(f"Failed response status: {response}")
        #     raise HTTPException(status_code=500, detail="LLM service processing failed")

        # Extract usage metrics from response
        # usage_metrics = response.get("usage", {})
        if not usage_metrics:
            logger.warning("No usage metrics found in response")

        return {
            "question": request.question,
            "answer": content,
            "source_pdf": request.pdf_id,
            "usage_metrics": {
                "input_tokens": usage_metrics.get("input_tokens", 0),
                "output_tokens": usage_metrics.get("output_tokens", 0),
                "total_tokens": usage_metrics.get("total_tokens", 0),
                "cost": usage_metrics.get("cost", 0.0),
            },
            "status": "success",
        }
    except Exception as e:
        logger.error(f"LLM processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Question answering failed")

@app.post("/ask_nvidia", status_code=status.HTTP_200_OK, tags=["Assignment 4"])
async def ask_nvidia(request: QuestionRequest, year: Optional[str] = None, quarter: Optional[str] = None):
    try:
        pipeline = RAGPipeline(
            pdf_id="0000",
            text="",
            vector_store="nvidia",
            chunking_strategy="recursive",
            year=year,
            quarter=quarter
        )
        pipeline.process()
        
        relevant_chunks = pipeline.get_relevant_chunks(
            query=request.question,
            k=5
        )
        
        response = answer_question_rag(
            query=request.question,
            model_name=request.model,
            text='\n'.join(relevant_chunks)
        )
        return response
    except Exception as e:
        logger.error(f"Error in NVIDIA RAG query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
def create_zip_archive(result, include_markdown, include_images, include_tables):
    flag = False
    messages = []
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # Markdown
        if include_markdown:
            if not result["markdown"]:
                messages.append(
                    "Markdown couldn't be generated. Maybe webpage has blockers."
                )
            else:
                zip_file.write(result["markdown"], arcname="document.md")
                flag = flag or True

        # Images
        if include_images:
            if not result["images"]:
                messages.append("No images found in the input webpage.")
            else:
                for img in result["images"].iterdir():
                    zip_file.write(img, arcname=f"images/{img.name}")
                flag = flag or True

        # Tables
        if include_tables:
            if not result["tables"]:
                messages.append("No tables found in the input webpage.")
            else:
                for table in result["tables"].iterdir():
                    zip_file.write(table, arcname=f"tables/{table.name}")
                flag = flag or True

    zip_buffer.seek(0)
    return flag, zip_buffer, messages


def my_background_task():
    logger.info("Running background maintenance tasks")
    clean_temp_files()
    print("Performed cleanup of temp files.")
