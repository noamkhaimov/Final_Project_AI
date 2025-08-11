"""
Flask blueprint with endpoints for:
  • POST /papers  – upload one or more PDFs and push them to ChromaDB
  • POST /query   – retrieve context from ChromaDB and ask the LLM
  • GET /pdf/files – list all uploaded PDF files
  • GET /pdf/files/<file_id> – get details of a specific PDF file
  • GET /logs/application – get application logs

Environment variables expected
------------------------------
CHROMA_HOST   (default: chroma)          – host of the Chromadb server
CHROMA_PORT   (default: 8000)            – port of the Chromadb server
OLLAMA_MODEL  (default: phi3:mini)       – model tag served by ollama
MONGO_URI     (default: mongodb://mongo:27017/)
MONGO_DB      (default: pdf_upload_db)

"""

from __future__ import annotations

# Standard library imports
import os
import io
import uuid
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Third-party imports
import ollama
import requests
from flask import Blueprint, current_app, jsonify, request
from pymongo import MongoClient
from werkzeug.utils import secure_filename
from bson import ObjectId
from bson.errors import InvalidId
from pydantic import ValidationError
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate

# Local application imports
from ..utils import app_logger, PDFProcessor
from ..models import QueryInput, QueryOutput
from ..rag import RAGOutputParser, QUERY_PROMPT, load_pdf, split_documents
from ..database import mongo_client, chroma, add_documents
from ..database.chroma_client import get_llm_model_name
from ..config import settings

# ─── LangChain / Chroma ------------------------------------------------------------------
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate

# ─── Blueprint ---------------------------------------------------------------------------
api_bp = Blueprint("api", __name__)

# ─── Constants & one-time inits -----------------------------------------------------------
UPLOAD_DIR = Path("/app/storage/pdfs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Get MongoDB collections
pdf_files = mongo_client['pdf_files']
logs = mongo_client['logs']

prompt = ChatPromptTemplate.from_template(
    "Answer the question using ONLY the context below.\n\n{context}\n\nQuestion: {question}"
)
llm = ChatOllama(model=get_llm_model_name(),
    base_url=os.getenv("OLLAMA_API_BASE", "http://ollama:11434"))


def _allowed(filename: str) -> bool:
    return PDFProcessor.allowed_file(filename)


def handle_error(error: Exception, endpoint: str, operation: str) -> Dict[str, Any]:
    """
    Centralized error handler that provides detailed error information
    """
    error_info = {
        "error": str(error),
        "error_type": type(error).__name__,
        "endpoint": endpoint,
        "operation": operation,
        "traceback": traceback.format_exc()
    }
    app_logger.log_request(endpoint, operation, 500, f'Error: {str(error)}')
    current_app.logger.error(f"Error in {endpoint} ({operation}): {str(error)}\n{traceback.format_exc()}")
    return error_info


# ──────────────────────────────────────────────────────────────────────────────
# POST /papers
# ──────────────────────────────────────────────────────────────────────────────
@api_bp.post("/papers")
def upload_papers():
    """Accept one or more PDF files and index them in ChromaDB."""
    try:
        files = request.files.getlist("files")
        if not files:
            app_logger.log_request('/papers', 'POST', 400, 'No files provided')
            return jsonify({"error": "field 'files' missing"}), 400

        doc_ids: List[str] = []
        for f in files:
            if not _allowed(f.filename):
                continue

            try:
                # Read file content first
                file_content = f.read()
                if not file_content:
                    app_logger.log_request('/papers', 'POST', 400, f'Empty file: {f.filename}')
                    return jsonify({"error": f"File is empty: {f.filename}"}), 400

                # Save file to disk
                fname = secure_filename(f.filename)
                path = UPLOAD_DIR / fname
                with open(path, 'wb') as file:
                    file.write(file_content)

                # Extract text from PDF using the file content
                try:
                    text_content = PDFProcessor.extract_text_from_pdf(io.BytesIO(file_content))
                except Exception as pdf_error:
                    error_info = handle_error(pdf_error, '/papers', 'PDF text extraction')
                    return jsonify(error_info), 500

                # Save metadata to MongoDB
                try:
                    pdf_data = {
                        'filename': fname,
                        'original_filename': f.filename,
                        'file_size': len(file_content),
                        'upload_time': datetime.utcnow(),
                        'content_text': text_content,
                        'content_length': len(text_content),
                        'mime_type': f.content_type or 'application/pdf'
                    }
                    record_id = pdf_files.insert_one(pdf_data).inserted_id
                except Exception as mongo_error:
                    error_info = handle_error(mongo_error, '/papers', 'MongoDB insertion')
                    return jsonify(error_info), 500

                # Process PDF for ChromaDB
                try:
                    documents = load_pdf(str(path))
                    pages = split_documents(documents)
                    for chunk in pages:
                        chunk.metadata.update({
                            "document_id": str(record_id),
                            "filename": fname,
                            "original_filename": f.filename
                        })
                    add_documents(pages, str(record_id))
                except Exception as chroma_error:
                    error_info = handle_error(chroma_error, '/papers', 'ChromaDB indexing')
                    return jsonify(error_info), 500

                current_app.logger.info("Indexed %s (%d chunks)", fname, len(pages))
                doc_ids.append(str(record_id))

                app_logger.log_request('/papers', 'POST', 200, 'PDF uploaded successfully',
                                   {'record_id': str(record_id), 'filename': fname})

            except Exception as file_error:
                error_info = handle_error(file_error, '/papers', f'Processing file: {f.filename}')
                return jsonify(error_info), 500

        return jsonify({"document_ids": doc_ids})

    except Exception as e:
        error_info = handle_error(e, '/papers', 'Main endpoint')
        return jsonify(error_info), 500


# ──────────────────────────────────────────────────────────────────────────────
# POST /query
# ──────────────────────────────────────────────────────────────────────────────
@api_bp.post("/query")
def query():
    """Run a RAG query over the indexed papers."""
    try:
        current_app.logger.info("Starting query processing")
        
        # Get question from form parameter (easier than JSON body)
        question = request.form.get('question', '').strip()
        
        # Initialize log entry with basic information
        log_entry = {
            "timestamp": datetime.utcnow(),
            "question": question,
            "success": False,  # Default to False, will update if successful
            "filepdfname": "unknown",
            "model": get_llm_model_name()
        }
        
        if not question:
            error_msg = "Question parameter is required"
            log_entry["answer"] = error_msg
            logs.insert_one(log_entry)
            return jsonify({
                "success": False,
                "filepdfname": "unknown",
                "answer": error_msg
            }), 400
        
        if len(question) < 3:
            error_msg = "Question must be at least 3 characters long"
            log_entry["answer"] = error_msg
            logs.insert_one(log_entry)
            return jsonify({
                "success": False,
                "filepdfname": "unknown",
                "answer": error_msg
            }), 400
            
        current_app.logger.info(f"Processing query: {question}")

        # Retrieve relevant context from ChromaDB
        try:
            current_app.logger.info("Querying ChromaDB for relevant documents")
            chroma_db = chroma()  # Get ChromaDB instance
            docs = chroma_db.similarity_search(
                question,
                k=1  # Get single most relevant chunk for speed
            )
            current_app.logger.info(f"Found {len(docs) if docs else 0} relevant documents")
            
            if not docs:
                error_msg = "No relevant documents found for your query."
                log_entry["answer"] = error_msg
                logs.insert_one(log_entry)
                return jsonify({
                    "success": False,
                    "filepdfname": "unknown",
                    "answer": error_msg
                }), 404
        except Exception as chroma_error:
            error_msg = f"Error searching documents: {str(chroma_error)}"
            log_entry["answer"] = error_msg
            logs.insert_one(log_entry)
            current_app.logger.error(f"ChromaDB search failed: {str(chroma_error)}", exc_info=True)
            return jsonify({
                "success": False,
                "filepdfname": "unknown",
                "answer": error_msg
            }), 500
        
        # Join context with metadata preserved and limit size
        try:
            context_parts = []
            for i, d in enumerate(docs):
                # Limit each document to 1000 characters to prevent context overflow
                content = d.page_content[:1000] + "..." if len(d.page_content) > 1000 else d.page_content
                context_parts.append(f"[Document {i+1}]\n{content}")
            
            context = "\n\n".join(context_parts)
            current_app.logger.debug(f"Built context from {len(docs)} documents (limited to 1000 chars each)")
        except Exception as context_error:
            error_msg = f"Error building context: {str(context_error)}"
            log_entry["answer"] = error_msg
            logs.insert_one(log_entry)
            current_app.logger.error(f"Failed to build context: {str(context_error)}", exc_info=True)
            return jsonify({
                "success": False,
                "filepdfname": "unknown",
                "answer": error_msg
            }), 500

        # Generate answer using LLM with our structured prompt
        try:
            current_app.logger.info("Sending query to LLM")
            prompt = QUERY_PROMPT.format(
                context=context,
                question=question
            )
            current_app.logger.debug(f"Using prompt: {prompt}")
            
            llm_response = llm.invoke(prompt).content
            log_entry["raw_llm_response"] = llm_response
            current_app.logger.info("Received response from LLM")
            current_app.logger.debug(f"Raw LLM response: {llm_response}")
        except Exception as llm_error:
            error_msg = str(llm_error)
            log_entry["answer"] = f"Error generating response: {error_msg}"
            logs.insert_one(log_entry)
            current_app.logger.error(f"LLM processing failed: {error_msg}", exc_info=True)
            
            # Handle specific Ollama decode errors
            if "decode: cannot decode batches" in error_msg or "llama_encode" in error_msg:
                return jsonify({
                    "success": False,
                    "filepdfname": docs[0].metadata.get('filename', 'unknown') if docs else "unknown",
                    "answer": f"Error generating response: {error_msg}"
                }), 500

        # Parse the response
        try:
            current_app.logger.info("Parsing LLM response")
            output_parser = RAGOutputParser()
            parsed_output = output_parser.parse_response(
                llm_output=llm_response,
                context_docs=docs,
                model_name=get_llm_model_name()
            )
            current_app.logger.info("Successfully parsed LLM response")
            
            # Update log entry with successful response
            log_entry.update({
                "success": parsed_output.success,
                "filepdfname": parsed_output.filepdfname,
                "answer": parsed_output.answer
            })
            logs.insert_one(log_entry)
            
        except Exception as parser_error:
            error_msg = f"Error parsing response: {str(parser_error)}"
            log_entry["answer"] = error_msg
            logs.insert_one(log_entry)
            current_app.logger.error(f"Response parsing failed: {str(parser_error)}", exc_info=True)
            return jsonify({
                "success": False,
                "filepdfname": docs[0].metadata.get('filename', 'unknown') if docs else "unknown",
                "answer": error_msg
            }), 500

        app_logger.log_request('/query', 'POST', 200, 'Query processed successfully')
        current_app.logger.info("Query processing completed successfully")
        
        # Return response
        return jsonify({
            "success": parsed_output.success,
            "filepdfname": parsed_output.filepdfname,
            "answer": parsed_output.answer
        })

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        log_entry = {
            "timestamp": datetime.utcnow(),
            "question": request.form.get('question', '').strip(),
            "success": False,
            "filepdfname": "unknown",
            "answer": error_msg,
            "model": get_llm_model_name()
        }
        logs.insert_one(log_entry)
        current_app.logger.error(f"Unexpected error in query endpoint: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "filepdfname": "unknown",
            "answer": error_msg
        }), 500


# ──────────────────────────────────────────────────────────────────────────────
# GET /pdf/files
# ──────────────────────────────────────────────────────────────────────────────
@api_bp.get("/pdf/files")
def list_pdf_files():
    """Get list of all uploaded PDF files."""
    try:
        # Get limit parameter from query string (default: 100)
        limit = request.args.get('limit', default=100, type=int)
        limit = min(max(1, limit), 1000)  # Ensure limit is between 1 and 1000

        # Retrieve PDF records from database
        records = []
        for record in pdf_files.find().limit(limit):
            # Convert ObjectId to string and format response
            record['_id'] = str(record['_id'])
            if 'content_text' in record:
                content = record['content_text']
                record['content_preview'] = content[:200] + '...' if len(content) > 200 else content
                del record['content_text']
            if 'upload_time' in record:
                record['upload_time'] = record['upload_time'].isoformat()
            records.append(record)

        app_logger.log_request('/pdf/files', 'GET', 200, f'Retrieved {len(records)} PDF records')
        return jsonify({
            "success": True,
            "message": f"Retrieved {len(records)} PDF file records",
            "data": records
        })

    except Exception as e:
        app_logger.log_request('/pdf/files', 'GET', 500, f'Failed to retrieve PDF files: {str(e)}')
        return jsonify({
            "success": False,
            "message": f"Failed to retrieve PDF files: {str(e)}",
            "data": None
        }), 500


# ──────────────────────────────────────────────────────────────────────────────
# GET /pdf/files/<file_id>
# ──────────────────────────────────────────────────────────────────────────────
@api_bp.get("/pdf/files/<file_id>")
def get_pdf_file(file_id):
    """Get specific PDF file record."""
    try:
        # Convert string ID to ObjectId
        object_id = ObjectId(file_id)
        record = pdf_files.find_one({'_id': object_id})

        if not record:
            app_logger.log_request(f'/pdf/files/{file_id}', 'GET', 404, 'PDF file not found')
            return jsonify({
                "success": False,
                "message": "PDF file not found",
                "data": None
            }), 404

        # Convert ObjectId to string and format response
        record['_id'] = str(record['_id'])
        if 'content_text' in record:
            content = record['content_text']
            record['content_preview'] = content[:200] + '...' if len(content) > 200 else content
            del record['content_text']
        if 'upload_time' in record:
            record['upload_time'] = record['upload_time'].isoformat()

        app_logger.log_request(f'/pdf/files/{file_id}', 'GET', 200, 'PDF file retrieved')
        return jsonify({
            "success": True,
            "message": "PDF file retrieved successfully",
            "data": record
        })

    except InvalidId:
        app_logger.log_request(f'/pdf/files/{file_id}', 'GET', 400, 'Invalid file ID format')
        return jsonify({
            "success": False,
            "message": "Invalid file ID format",
            "data": None
        }), 400
    except Exception as e:
        app_logger.log_request(f'/pdf/files/{file_id}', 'GET', 500, f'Failed to retrieve PDF file: {str(e)}')
        return jsonify({
            "success": False,
            "message": f"Failed to retrieve PDF file: {str(e)}",
            "data": None
        }), 500


# ──────────────────────────────────────────────────────────────────────────────
# GET /logs/application
# ──────────────────────────────────────────────────────────────────────────────
@api_bp.get("/logs/application")
def get_application_logs():
    """Get application logs."""
    try:
        # Get limit parameter from query string (default: 100)
        limit = request.args.get('limit', default=100, type=int)
        limit = min(max(1, limit), 1000)  # Ensure limit is between 1 and 1000

        # Retrieve logs from database
        logs_list = []
        for log in logs.find().sort('timestamp', -1).limit(limit):  # Use application_logs collection
            log['_id'] = str(log['_id'])
            if 'timestamp' in log:
                log['timestamp'] = log['timestamp'].isoformat()
            logs_list.append(log)

        app_logger.log_request('/logs/application', 'GET', 200, f'Retrieved {len(logs_list)} log records')
        return jsonify({
            "success": True,
            "message": f"Retrieved {len(logs_list)} log records",
            "logs": logs_list
        })

    except Exception as e:
        app_logger.log_request('/logs/application', 'GET', 500, f'Failed to retrieve logs: {str(e)}')
        return jsonify({
            "success": False,
            "message": f"Failed to retrieve logs: {str(e)}",
            "logs": None
        }), 500



