# SQL Query AI Agent

[English](#english) | [Bahasa Indonesia](#bahasa-indonesia)

# English

An AI Agent system capable of converting natural language questions into SQL queries, executing those queries, and providing explanations of the results in natural language.

## 🌟 Key Features

- Natural language to SQL query conversion
- PostgreSQL database integration
- Query results explanation in natural language
- Support for Local LLM (LM Studio with OpenAI API compatibility)
- Containerized with Docker

## 🔧 Technology Stack

- Python 3.9
- FastAPI
- LangChain
- SQLAlchemy
- PostgreSQL
- Docker
- Local LLM (LM Studio)

## 🏗️ Architecture and Flow

### Main Components

1. **SQLAgent (`agent.py`)**
   - Manages query conversion and database interactions
   - Uses LLM for SQL generation and result elaboration
   - Handles errors and logging

2. **FastAPI Server (`main.py`)**
   - Provides REST API endpoints
   - Handles requests and responses

3. **Database Connector (`database.py`)**
   - Manages PostgreSQL connections
   - Provides session management

### Process Flow

1. **Input Processing**
   ```
   User Input (Natural Language)
         ↓
   FastAPI Endpoint
         ↓
   SQLAgent
   ```

2. **Query Generation**
   ```
   1. Fetch Database Schema
   2. Combine with Prompt Template
   3. Send to LLM
   4. Generate SQL Query
   ```

3. **Query Execution**
   ```
   SQL Query
      ↓
   Execute in PostgreSQL
      ↓
   Convert results (handle Decimal)
   ```

4. **Result Elaboration**
   ```
   Query Results
        ↓
   Format JSON
        ↓
   LLM Elaboration
        ↓
   Natural Language Explanation
   ```

## 🚀 Getting Started

### Prerequisites

1. Docker and Docker Compose
2. LM Studio with OpenAI API compatible model
3. PostgreSQL database

### Environment Variables

Create a `.env` file with:

```env
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_HOST=your_db_host
DB_PORT=5432
DB_NAME=your_database
LOCAL_LLM_BASE_URL=http://your_llm_host:1234
```

### Running the Application

```bash
# Build and run containers
docker-compose up --build

# Test API
curl -X POST "http://localhost:8000/api/query" \
-H "Content-Type: application/json" \
-d '{"text": "total sales this month"}'
```

## 📝 Usage Example

### Request
```json
{
    "text": "total sales this month"
}
```

### Response
```json
{
    "status": "success",
    "data": "Based on the data, total sales for this month is $15,750,000. 
    This figure represents an accumulation of 23 transactions recorded in the system..."
}
```

## 🧠 AI Agent Logic

### 1. SQL Query Generation
- Uses structured system prompts
- Includes database schema for context
- Optimizes queries based on database structure

### 2. Result Elaboration
- Comprehensive analysis of query results
- Provides business insights
- Explains trends and patterns
- Easy-to-understand language format

### 3. Error Handling
- Input validation
- Database error handling
- Data type conversion (Decimal handling)
- Logging for debugging

## 🔒 Security

- Credentials stored in environment variables
- URL encoding for special characters
- Query input validation
- Safe error messages

## 📚 API Documentation

### POST /api/query

**Request Body:**
```json
{
    "text": "string"
}
```

**Response:**
```json
{
    "status": "success|error",
    "data": "string"
}
```

## 🤝 Contributing

Please create issues or pull requests for improvements or feature additions.

## 📄 License

MIT License

---

# Bahasa Indonesia

Sistem AI Agent yang dapat mengkonversi pertanyaan dalam bahasa natural (Bahasa Indonesia) menjadi query SQL, mengeksekusi query tersebut, dan memberikan penjelasan hasil dalam Bahasa Indonesia.

## 🌟 Fitur Utama

- Konversi bahasa natural ke SQL query
- Integrasi dengan PostgreSQL database
- Penjelasan hasil query dalam Bahasa Indonesia
- Mendukung Local LLM (LM Studio dengan OpenAI API compatibility)
- Containerized dengan Docker

## 🔧 Teknologi

- Python 3.9
- FastAPI
- LangChain
- SQLAlchemy
- PostgreSQL
- Docker
- Local LLM (LM Studio)

## 🏗️ Arsitektur dan Flow

### Komponen Utama

1. **SQLAgent (`agent.py`)**
   - Mengelola konversi query dan interaksi dengan database
   - Menggunakan LLM untuk generasi SQL dan elaborasi hasil
   - Menangani error dan logging

2. **FastAPI Server (`main.py`)**
   - Menyediakan REST API endpoint
   - Menangani request dan response

3. **Database Connector (`database.py`)**
   - Mengelola koneksi ke PostgreSQL
   - Menyediakan session management

### Flow Proses

1. **Input Processing**
   ```
   User Input (Bahasa Natural)
         ↓
   FastAPI Endpoint
         ↓
   SQLAgent
   ```

2. **Query Generation**
   ```
   1. Ambil Schema Database
   2. Kombinasikan dengan Prompt Template
   3. Kirim ke LLM
   4. Generate SQL Query
   ```

3. **Query Execution**
   ```
   SQL Query
      ↓
   Execute di PostgreSQL
      ↓
   Convert hasil (handle Decimal)
   ```

4. **Result Elaboration**
   ```
   Query Results
        ↓
   Format JSON
        ↓
   LLM Elaboration
        ↓
   Penjelasan dalam Bahasa Indonesia
   ```

## 🚀 Cara Menjalankan

### Prerequisites

1. Docker dan Docker Compose
2. LM Studio dengan model yang compatible dengan OpenAI API
3. PostgreSQL database

### Environment Variables

Buat file `.env` dengan isi:

```env
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_HOST=your_db_host
DB_PORT=5432
DB_NAME=your_database
LOCAL_LLM_BASE_URL=http://your_llm_host:1234
```

### Menjalankan Aplikasi

```bash
# Build dan jalankan container
docker-compose up --build

# Test API
curl -X POST "http://localhost:8000/api/query" \
-H "Content-Type: application/json" \
-d '{"text": "total penjualan bulan ini"}'
```

## 📝 Contoh Penggunaan

### Request
```json
{
    "text": "total penjualan bulan ini"
}
```

### Response
```json
{
    "status": "success",
    "data": "Berdasarkan data yang ada, total penjualan untuk bulan ini adalah Rp 15,750,000. 
    Angka ini merupakan akumulasi dari 23 transaksi yang tercatat di sistem..."
}
```

## 🧠 Logic AI Agent

### 1. SQL Query Generation
- Menggunakan sistem prompt yang terstruktur
- Menyertakan schema database untuk konteks
- Mengoptimalkan query berdasarkan struktur database

### 2. Result Elaboration
- Menganalisis hasil query secara komprehensif
- Memberikan insights bisnis
- Menjelaskan tren dan pola
- Format bahasa yang mudah dipahami

### 3. Error Handling
- Validasi input
- Penanganan error database
- Konversi tipe data (Decimal handling)
- Logging untuk debugging

## 🔒 Security

- Credentials disimpan di environment variables
- URL encoding untuk special characters
- Validasi input query
- Error messages yang aman

## 📚 API Documentation

### POST /api/query

**Request Body:**
```json
{
    "text": "string"
}
```

**Response:**
```json
{
    "status": "success|error",
    "data": "string"
}
```

## 🤝 Kontribusi

Silakan buat issue atau pull request untuk perbaikan atau penambahan fitur.

## 📄 License

MIT License
