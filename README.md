<div align="center">

# 🗄️ ChromaDB Knowledge Server

### Production-Ready Vector Database Server for RAG Applications

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.x-green.svg)](https://flask.palletsprojects.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Latest-orange.svg)](https://www.trychroma.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[Features](#-features) • [Quick Start](#-quick-start) • [API Reference](#-api-reference) • [Deployment](#-deployment) • [Contributing](#-contributing)

</div>

---

## 📖 Overview

ChromaDB Knowledge Server is a **production-ready REST API** for vector-based document storage and semantic search. It provides everything you need to build RAG (Retrieval-Augmented Generation) applications:

- 📄 **Document Processing** - Upload PDF, DOCX, TXT files with automatic text extraction
- 🔍 **Semantic Search** - Find relevant content using AI-powered similarity search
- 🔐 **Authentication** - Role-based access control with secure sessions
- 📊 **Dashboard** - Modern web interface for management and monitoring
- 🌍 **Multilingual** - Full Arabic and English support

---

## ✨ Features

### Document Processing Pipeline
```
Upload → Extract Text → Chunk → Generate Embeddings → Store in ChromaDB
```

| Format | Features |
|--------|----------|
| **PDF** | Text extraction, tables, Arabic OCR |
| **DOCX** | Full text and table support |
| **TXT** | Direct ingestion, encoding detection |

### Security & Access Control
- 🔑 Session-based authentication
- 👥 Three roles: Admin, Manager, Viewer
- 🛡️ Rate limiting protection
- 🔒 Password hashing with Werkzeug

### Web Dashboard
- 📈 Real-time statistics
- 📁 File management
- ⬆️ Drag-and-drop upload
- 💚 Health monitoring
- 👤 User management

### API Capabilities
- RESTful design with JSON responses
- 15+ endpoints for full functionality
- Comprehensive error handling
- CORS support for frontend integration

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/Salman-Naif/chromadb-server-public.git
cd chromadb-server-public

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your settings
```

### Configuration

Create a `.env` file:

```env
# Required
SECRET_KEY=your-secret-key-here
OPENROUTER_API_KEY=sk-or-v1-xxxxx

# Optional
FLASK_ENV=production
CHROMADB_PATH=./data/chromadb
ADMIN_PASSWORD=your-admin-password
```

### Run the Server

```bash
# Development
flask run --debug

# Production
gunicorn app:app --bind 0.0.0.0:5000 --workers 1 --timeout 300
```

Visit `http://localhost:5000` and login with:
- **Username**: `admin`
- **Password**: (set in .env or default: `admin123`)

---

## 📡 API Reference

### Authentication

All API endpoints require authentication via session cookie.

### Endpoints

#### Upload Document
```http
POST /api/upload
Content-Type: multipart/form-data

file: <document file>
```

**Response:**
```json
{
  "success": true,
  "file_id": "abc123",
  "filename": "document.pdf",
  "chunks_count": 15,
  "message": "File uploaded successfully"
}
```

#### Semantic Search
```http
POST /api/search
Content-Type: application/json

{
  "query": "What are the graduation requirements?",
  "top_k": 5
}
```

**Response:**
```json
{
  "success": true,
  "results": [
    {
      "text": "Graduation requires completion of 132 credit hours...",
      "score": 0.89,
      "metadata": {
        "source": "academic_guide.pdf",
        "chunk_index": 12
      }
    }
  ]
}
```

#### List Documents
```http
GET /api/documents
```

#### Delete Document
```http
DELETE /api/documents/{file_id}
```

#### System Health
```http
GET /api/health
```

#### System Statistics
```http
GET /api/stats
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   ChromaDB Knowledge Server                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Flask      │  │   ChromaDB   │  │   SQLite     │      │
│  │   REST API   │  │   Vectors    │  │   Users/Logs │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                 │                 │               │
│  ┌──────┴─────────────────┴─────────────────┴──────┐       │
│  │              Core Services                       │       │
│  │                                                  │       │
│  │  • DocumentProcessor  - Text extraction          │       │
│  │  • EmbeddingsManager  - Vector generation        │       │
│  │  • ChromaDBManager    - Storage & search         │       │
│  │  • AuthManager        - User authentication      │       │
│  └──────────────────────────────────────────────────┘       │
│                           │                                  │
├───────────────────────────┼──────────────────────────────────┤
│  External Services:       ▼                                  │
│  ┌──────────────────────────────┐                           │
│  │  OpenRouter API (Embeddings) │                           │
│  └──────────────────────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

---

## ☁️ Deployment

### Render.com (Recommended)

1. Fork this repository
2. Create new Web Service on Render
3. Connect your GitHub repo
4. Set environment variables:
   - `SECRET_KEY`
   - `OPENROUTER_API_KEY`
5. Deploy!

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
```

```bash
docker build -t chromadb-server .
docker run -p 5000:5000 -e SECRET_KEY=xxx -e OPENROUTER_API_KEY=xxx chromadb-server
```

### Railway

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new)

---

## ⚙️ Configuration Options

| Variable | Description | Default |
|----------|-------------|---------|
| `SECRET_KEY` | Flask secret key | Required |
| `OPENROUTER_API_KEY` | API key for embeddings | Required |
| `CHROMADB_PATH` | Database storage path | `./data/chromadb` |
| `CHROMADB_COLLECTION` | Collection name | `knowledge_base` |
| `EMBEDDING_MODEL` | Model for embeddings | `qwen/qwen3-embedding-8b` |
| `EMBEDDING_DIMENSION` | Vector dimensions | `4096` |
| `MAX_CONTENT_LENGTH` | Max upload size | `52428800` (50MB) |
| `ADMIN_USERNAME` | Default admin user | `admin` |
| `ADMIN_PASSWORD` | Default admin password | `admin123` |

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Search Latency | < 500ms |
| Max Document Size | 50MB |
| Max Pages per Document | 50+ |
| Minimum RAM | 512MB |
| Concurrent Users | 100+ |

### Memory Optimization

The server is optimized for constrained environments:
- Automatic chunk limiting (max 15 per document)
- Garbage collection after processing
- Worker recycling to prevent memory leaks

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Run linter
flake8 app.py
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [ChromaDB](https://www.trychroma.com/) - Vector database
- [OpenRouter](https://openrouter.ai/) - AI API gateway
- [Flask](https://flask.palletsprojects.com/) - Web framework
- [pdfplumber](https://github.com/jsvine/pdfplumber) - PDF extraction

---

## 📧 Contact

**SALMAN NAIF ALMUHAYSIN** - [ Find me on X : @Snma7_ ](https://x.com/snma7_?s=21) 
- Or Mail me on : Snma2003@outlook.sa

Project Link: [https://github.com/Salman-Naif/chromadb-server-public.git]
---

<div align="center">

**⭐ Star this repo if you find it useful!**

Made with ❤️ for the open source community

</div>
