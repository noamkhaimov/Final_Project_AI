from flask import Blueprint, jsonify

SWAGGER_SPEC = {
    "openapi": "3.0.0",
    "info": {
        "title": "AI Agent",
        "version": "1.0.0",
        "description": "API for uploading PDFs, querying them using RAG (Retrieval-Augmented Generation), and managing application logs"
    },
    "servers": [
        {
            "url": "/",
            "description": "Local development server"
        }
    ],
    "paths": {
        "/papers": {
            "post": {
                "summary": "Upload PDF papers",
                "description": "Upload one or more PDF files to be indexed in ChromaDB",
                "requestBody": {
                    "required": True,
                    "content": {
                        "multipart/form-data": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "files": {
                                        "type": "array",
                                        "items": {
                                            "type": "string",
                                            "format": "binary"
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Files successfully uploaded and indexed",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "document_ids": {
                                            "type": "array",
                                            "items": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "400": {
                        "description": "Bad request - no files provided"
                    },
                    "500": {
                        "description": "Server error processing files"
                    }
                }
            }
        },
        "/query": {
            "post": {
                "summary": "Query the indexed papers",
                "description": "Ask the AI Agent a question about the uploaded papers using RAG",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/x-www-form-urlencoded": {
                            "schema": {
                                "type": "object",
                                "required": ["question"],
                                "properties": {
                                    "question": {
                                        "type": "string",
                                        "description": "The question to ask about the papers",
                                        "minLength": 3,
                                        "example": "summarize 'sleep_class.pdf' for me"
                                    }
                                }
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Query answered successfully",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/QueryResponse"}
                            }
                        }
                    },
                    "400": {
                        "description": "Bad request - question too short or missing"
                    },
                    "500": {
                        "description": "Server error processing query"
                    }
                }
            }
        },
        "/pdf/files": {
            "get": {
                "summary": "List all PDF files",
                "description": "Get a list of all uploaded PDF files with metadata",
                "parameters": [
                    {
                        "name": "limit",
                        "in": "query",
                        "description": "Maximum number of records to return (default: 100, max: 1000)",
                        "required": False,
                        "schema": {
                            "type": "integer",
                            "default": 100,
                            "minimum": 1,
                            "maximum": 1000
                        }
                    }
                ],
                "responses": {
                    "200": {
                        "description": "List of PDF files retrieved successfully",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/PDFListResponse"}
                            }
                        }
                    },
                    "500": {
                        "description": "Server error retrieving files"
                    }
                }
            }
        },
        "/pdf/files/{file_id}": {
            "get": {
                "summary": "Get PDF file details",
                "description": "Get detailed information about a specific PDF file",
                "parameters": [
                    {
                        "name": "file_id",
                        "in": "path",
                        "description": "ID of the PDF file to retrieve",
                        "required": True,
                        "schema": {
                            "type": "string"
                        }
                    }
                ],
                "responses": {
                    "200": {
                        "description": "PDF file details retrieved successfully",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/PDFDetailResponse"}
                            }
                        }
                    },
                    "404": {
                        "description": "PDF file not found"
                    },
                    "500": {
                        "description": "Server error retrieving file"
                    }
                }
            }
        },
        "/logs/application": {
            "get": {
                "summary": "Get application logs",
                "description": "Retrieve application logs with optional filtering",
                "parameters": [
                    {
                        "name": "limit",
                        "in": "query",
                        "description": "Maximum number of logs to return (default: 100)",
                        "required": False,
                        "schema": {
                            "type": "integer",
                            "default": 100,
                            "minimum": 1,
                            "maximum": 1000
                        }
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Logs retrieved successfully",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/LogListResponse"}
                            }
                        }
                    },
                    "500": {
                        "description": "Server error retrieving logs"
                    }
                }
            }
        },

    },
    "components": {
        "schemas": {
            "QueryRequest": {
                "type": "object",
                "required": ["question"],
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to ask about the papers",
                        "minLength": 3
                    }
                }
            },
            "QueryResponse": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The generated answer based on the papers' content"
                    }
                }
            },
            "PDFListResponse": {
                "type": "object",
                "properties": {
                    "success": {
                        "type": "boolean",
                        "description": "Operation success status"
                    },
                    "message": {
                        "type": "string",
                        "description": "Response message"
                    },
                    "data": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "_id": {
                                    "type": "string",
                                    "description": "PDF record ID"
                                },
                                "filename": {
                                    "type": "string",
                                    "description": "Name of the PDF file"
                                },
                                "file_size": {
                                    "type": "integer",
                                    "description": "Size of the file in bytes"
                                },
                                "upload_time": {
                                    "type": "string",
                                    "format": "date-time",
                                    "description": "When the file was uploaded"
                                },
                                "content_preview": {
                                    "type": "string",
                                    "description": "Preview of the PDF content"
                                }
                            }
                        }
                    }
                }
            },
            "PDFDetailResponse": {
                "type": "object",
                "properties": {
                    "success": {
                        "type": "boolean",
                        "description": "Operation success status"
                    },
                    "message": {
                        "type": "string",
                        "description": "Response message"
                    },
                    "data": {
                        "type": "object",
                        "properties": {
                            "_id": {
                                "type": "string",
                                "description": "PDF record ID"
                            },
                            "filename": {
                                "type": "string",
                                "description": "Name of the PDF file"
                            },
                            "original_filename": {
                                "type": "string",
                                "description": "Original name of the uploaded file"
                            },
                            "file_size": {
                                "type": "integer",
                                "description": "Size of the file in bytes"
                            },
                            "upload_time": {
                                "type": "string",
                                "format": "date-time",
                                "description": "When the file was uploaded"
                            },
                            "content_preview": {
                                "type": "string",
                                "description": "Preview of the PDF content"
                            },
                            "content_length": {
                                "type": "integer",
                                "description": "Length of the extracted text content"
                            }
                        }
                    }
                }
            },
            "LogListResponse": {
                "type": "object",
                "properties": {
                    "success": {
                        "type": "boolean",
                        "description": "Operation success status"
                    },
                    "message": {
                        "type": "string",
                        "description": "Response message"
                    },
                    "logs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "_id": {
                                    "type": "string",
                                    "description": "Log record ID"
                                },
                                "timestamp": {
                                    "type": "string",
                                    "format": "date-time",
                                    "description": "When the log was created"
                                },
                                "endpoint": {
                                    "type": "string",
                                    "description": "API endpoint that was called"
                                },
                                "method": {
                                    "type": "string",
                                    "description": "HTTP method used"
                                },
                                "status_code": {
                                    "type": "integer",
                                    "description": "HTTP response status code"
                                },
                                "message": {
                                    "type": "string",
                                    "description": "Log message"
                                },
                                "additional_data": {
                                    "type": "object",
                                    "description": "Additional context data"
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

swagger_bp = Blueprint("swagger", __name__)

@swagger_bp.route("/spec")
def spec():
    return jsonify(SWAGGER_SPEC)
