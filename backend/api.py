import os
import asyncio
import zipfile
from io import BytesIO
from pathlib import Path
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import (
    FastAPI,
    UploadFile,
    HTTPException,
    status,
    BackgroundTasks,
    Query
)
from fastapi.responses import FileResponse, StreamingResponse
import logging

logger = logging.getLogger(__name__)
from pydantic import BaseModel, Field
from typing import Optional

from .redis_manager import (
    send_to_redis_stream, 
    get_pdf_content,
    receive_llm_response
)
from .pipelines import (
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
    from .llm_service import LLMService
    
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
    model: str = Field("gemini/gemini-2.0-flash", description="LLM model to use for summary generation")
    
class QuestionRequest(BaseModel):
    pdf_id: str
    question: str = Field(..., min_length=10)
    max_tokens: int = Field(500, gt=100, lt=2000)
    model: str = Field("gemini/gemini-2.0-flash", description="LLM model to use for question answering")


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
async def upload_pdf(file: UploadFile):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type")

    try:
        contents = await file.read()
        pdf_id = store_uploaded_pdf(contents, "mistral")
        contents = get_pdf_content(pdf_id)
        
        # Send text content to stream for processing
        try:
            await send_to_redis_stream('pdf_content', {
                'pdf_id': pdf_id,
                'content': contents,
            })
            logger.info(f"PDF {pdf_id} content sent to stream")
        except Exception as e:
            logger.error(f"Failed to process PDF: {str(e)}")
            raise
        return PDFUploadResponse(
            pdf_id=pdf_id, status="success", message=f"PDF stored with ID: {pdf_id}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await file.close()

@app.post("/summarize/", response_model=dict, tags=["Assignment 4"])
async def generate_summary(request: SummaryRequest):
    logger = logging.getLogger(__name__)

    try:

        # Send request to Redis stream and wait for response
        try:
            await send_to_redis_stream('llm_requests', {
                'type': 'summary',
                'pdf_id': request.pdf_id,
                'max_tokens': request.max_tokens,
                'model': request.model
            })
            
            response = await receive_llm_response()
            if not response:
                raise HTTPException(status_code=408, detail="LLM response timeout")

            # Extract usage metrics from response
            usage_metrics = response.get('usage', {})
            if not usage_metrics:
                logger.warning("No usage metrics found in response")

            return {
                "summary": response.get('content', ''), 
                "usage_metrics": {
                "input_tokens": usage_metrics.get('input_tokens', 0),
                "output_tokens": usage_metrics.get('output_tokens', 0),
                "total_tokens": usage_metrics.get('total_tokens', 0),
                "cost": usage_metrics.get('cost', 0.0)
                }   
            }
            
        except Exception as e:
            logger.error(f"LLM processing failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Summary generation failed")
            
    except FileNotFoundError as e:
        logger.error(f"PDF content lookup failed: {str(e)}")
        raise HTTPException(status_code=404, detail="PDF content not found")

@app.post("/ask_question", status_code=status.HTTP_200_OK, tags=["Assignment 4"])
async def answer_pdf_question(request: QuestionRequest):
    logger = logging.getLogger(__name__)
    try:
        await send_to_redis_stream('llm_requests', {
            'type': 'question',
            'pdf_id': request.pdf_id,
            'question': request.question,
            'max_tokens': request.max_tokens,
            'model': request.model
        })
            
        response = await receive_llm_response()
        if not response:
            raise HTTPException(status_code=408, detail="LLM response timeout")
            
        # Extract content from response
        content = response.get('content')
        if not content:
            logger.error(f"Missing content in response: {response}")
            raise HTTPException(status_code=500, detail="Invalid response format from LLM service")
        
        # Check response status
        if response.get('status') != 'success':
            logger.error(f"Failed response status: {response}")
            raise HTTPException(status_code=500, detail="LLM service processing failed")

        # Extract usage metrics from response
        usage_metrics = response.get('usage', {})
        if not usage_metrics:
            logger.warning("No usage metrics found in response")

        return {
            "question": request.question,
            "answer": content,
            "source_pdf": request.pdf_id,
            "usage_metrics": {
                "input_tokens": usage_metrics.get('input_tokens', 0),
                "output_tokens": usage_metrics.get('output_tokens', 0),
                "total_tokens": usage_metrics.get('total_tokens', 0),
                "cost": usage_metrics.get('cost', 0.0)
            },
            "status": "success"
        }
    except Exception as e:
        logger.error(f"LLM processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Question answering failed")

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
