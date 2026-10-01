import os
import time
import uuid
import sqlite3

import numpy as np
import faiss

from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from google import genai


# =========================================================
# 1. CONFIGURATION
# =========================================================

load_dotenv()

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
DATABASE = "memory.db"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found. Please add it to your .env file."
    )

# Current Gemini model
GEMINI_MODEL = "gemini-3.8-flash"

# Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

# Embedding model
print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# =========================================================
# 2. GLOBAL RAG VARIABLES
# =========================================================

faiss_index = None

document_chunks = []

uploaded_filename = None


# =========================================================
# 3. DATABASE / MEMORY
# =========================================================

def init_database():

    connection = sqlite3.connect(DATABASE)

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            message TEXT NOT NULL
        )
    """)

    connection.commit()

    connection.close()


init_database()


# =========================================================
# 4. SAVE MEMORY
# =========================================================

def save_message(session_id, role, message):

    connection = sqlite3.connect(DATABASE)

    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO conversations
        (session_id, role, message)
        VALUES (?, ?, ?)
        """,
        (
            session_id,
            role,
            message
        )
    )

    connection.commit()

    connection.close()


# =========================================================
# 5. GET MEMORY
# =========================================================

def get_memory(session_id, limit=10):

    connection = sqlite3.connect(DATABASE)

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT role, message
        FROM conversations
        WHERE session_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (
            session_id,
            limit
        )
    )

    rows = cursor.fetchall()

    connection.close()

    rows.reverse()

    return rows


# =========================================================
# 6. CLEAR MEMORY
# =========================================================

def clear_session_memory(session_id):

    connection = sqlite3.connect(DATABASE)

    cursor = connection.cursor()

    cursor.execute(
        """
        DELETE FROM conversations
        WHERE session_id = ?
        """,
        (session_id,)
    )

    connection.commit()

    connection.close()


# =========================================================
# 7. PDF TEXT EXTRACTION
# =========================================================

def extract_pdf_text(pdf_path):

    reader = PdfReader(pdf_path)

    full_text = ""

    for page_number, page in enumerate(reader.pages):

        try:

            text = page.extract_text()

            if text:
                full_text += f"\n\nPAGE {page_number + 1}\n"
                full_text += text

        except Exception as error:

            print(
                f"Could not read page {page_number + 1}: {error}"
            )

    return full_text


# =========================================================
# 8. TEXT CHUNKING
# =========================================================

def create_chunks(
    text,
    chunk_size=800,
    overlap=100
):

    text = text.replace("\r", "\n")

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(
            words[start:end]
        )

        if chunk.strip():

            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# =========================================================
# 9. CREATE FAISS VECTOR STORE
# =========================================================

def create_vector_store(chunks):

    global faiss_index
    global document_chunks

    if not chunks:

        faiss_index = None
        document_chunks = []

        return

    print("Creating embeddings...")

    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    dimension = embeddings.shape[1]

    faiss_index = faiss.IndexFlatL2(
        dimension
    )

    faiss_index.add(
        embeddings
    )

    document_chunks = chunks

    print(
        f"Vector store created with {len(chunks)} chunks."
    )


# =========================================================
# 10. SEARCH PDF
# =========================================================

def search_pdf(
    query,
    top_k=4
):

    global faiss_index
    global document_chunks

    if faiss_index is None:

        return []

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    distances, indices = faiss_index.search(
        query_embedding,
        min(top_k, len(document_chunks))
    )

    results = []

    for index in indices[0]:

        if index >= 0 and index < len(document_chunks):

            results.append(
                document_chunks[index]
            )

    return results


# =========================================================
# 11. GEMINI GENERATION WITH RETRY
# =========================================================

def generate_gemini_response(prompt):

    max_attempts = 4

    for attempt in range(max_attempts):

        try:

            print(
                f"Gemini request attempt "
                f"{attempt + 1}/{max_attempts}"
            )

            response = client.models.generate_content(

                model=GEMINI_MODEL,

                contents=prompt
            )

            if response and response.text:

                return response.text

            return "Gemini returned an empty response."

        except Exception as error:

            error_text = str(error)

            print(
                f"Gemini API error: {error_text}"
            )

            # Retry temporary 503 errors
            if (
                "503" in error_text
                or "UNAVAILABLE" in error_text
                or "high demand" in error_text.lower()
            ):

                if attempt < max_attempts - 1:

                    wait_time = 2 ** attempt

                    print(
                        f"Gemini temporarily unavailable."
                    )

                    print(
                        f"Retrying in {wait_time} seconds..."
                    )

                    time.sleep(wait_time)

                    continue

            # Other errors should not be retried
            raise error

    return "Gemini is temporarily unavailable. Please try again."


# =========================================================
# 12. HOME PAGE
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# 13. PDF UPLOAD
# =========================================================

@app.route(
    "/upload",
    methods=["POST"]
)
def upload_pdf():

    global uploaded_filename

    try:

        if "file" not in request.files:

            return jsonify({
                "success": False,
                "message": "No PDF file selected."
            }), 400

        file = request.files["file"]

        if file.filename == "":

            return jsonify({
                "success": False,
                "message": "Please select a PDF file."
            }), 400

        if not file.filename.lower().endswith(".pdf"):

            return jsonify({
                "success": False,
                "message": "Only PDF files are allowed."
            }), 400

        # Create unique filename
        unique_filename = (
            str(uuid.uuid4())
            + "_"
            + file.filename
        )

        pdf_path = os.path.join(
            UPLOAD_FOLDER,
            unique_filename
        )

        file.save(pdf_path)

        print(
            f"PDF saved: {pdf_path}"
        )

        # Extract text
        text = extract_pdf_text(
            pdf_path
        )

        if not text.strip():

            return jsonify({
                "success": False,
                "message": "Could not extract text from this PDF."
            }), 400

        # Create chunks
        chunks = create_chunks(
            text
        )

        if not chunks:

            return jsonify({
                "success": False,
                "message": "No readable content found in PDF."
            }), 400

        # Create vector store
        create_vector_store(
            chunks
        )

        uploaded_filename = file.filename

        return jsonify({

            "success": True,

            "message": "PDF uploaded successfully.",

            "filename": uploaded_filename,

            "chunks": len(chunks)

        })

    except Exception as error:

        print(
            f"PDF upload error: {error}"
        )

        return jsonify({

            "success": False,

            "message": str(error)

        }), 500


# =========================================================
# 14. CHAT
# =========================================================

@app.route(
    "/chat",
    methods=["POST"]
)
def chat():

    try:

        data = request.get_json()

        if not data:

            return jsonify({
                "success": False,
                "message": "Invalid request."
            }), 400

        user_message = data.get(
            "message",
            ""
        ).strip()

        session_id = data.get(
            "session_id"
        )

        if not session_id:

            session_id = str(
                uuid.uuid4()
            )

        if not user_message:

            return jsonify({

                "success": False,

                "message": "Please enter a question."

            }), 400

        # -----------------------------------------
        # Check PDF
        # -----------------------------------------

        if faiss_index is None:

            return jsonify({

                "success": False,

                "message":
                    "Please upload a PDF before asking questions."

            }), 400

        # -----------------------------------------
        # Get relevant PDF chunks
        # -----------------------------------------

        relevant_chunks = search_pdf(
            user_message,
            top_k=4
        )

        pdf_context = "\n\n".join(
            relevant_chunks
        )

        # -----------------------------------------
        # Get conversation memory
        # -----------------------------------------

        memory = get_memory(
            session_id,
            limit=10
        )

        memory_text = ""

        for role, message in memory:

            memory_text += (
                f"{role.upper()}: {message}\n"
            )

        # -----------------------------------------
        # Save current user message
        # -----------------------------------------

        save_message(
            session_id,
            "user",
            user_message
        )

        # -----------------------------------------
        # Build RAG + Memory prompt
        # -----------------------------------------

        prompt = f"""
You are a Memory + RAG assistant.

You must follow these rules carefully.

RULE 1:
Answer questions using ONLY the information available
in the provided PDF context and the conversation memory.

RULE 2:
Do not invent facts.

RULE 3:
Do not use outside knowledge when the answer is not
supported by the PDF context.

RULE 4:
If the answer is not available in the PDF context,
clearly say:

"Sorry, I couldn't find that information in the uploaded PDF."

RULE 5:
Conversation memory can be used to understand follow-up
questions such as:
"what about that?"
"explain the previous point"
"tell me more about it"

RULE 6:
Keep answers clear and easy to understand.

RULE 7:
If the user asks something unrelated to the uploaded PDF,
say that the information is not available in the uploaded PDF.

----------------------------------------
CONVERSATION MEMORY
----------------------------------------

{memory_text}

----------------------------------------
PDF CONTEXT
----------------------------------------

{pdf_context}

----------------------------------------
CURRENT USER QUESTION
----------------------------------------

{user_message}

----------------------------------------
ANSWER
----------------------------------------
"""

        # -----------------------------------------
        # Gemini
        # -----------------------------------------

        answer = generate_gemini_response(
            prompt
        )

        # -----------------------------------------
        # Save assistant response
        # -----------------------------------------

        save_message(
            session_id,
            "assistant",
            answer
        )

        return jsonify({

            "success": True,

            "session_id": session_id,

            "answer": answer

        })

    except Exception as error:

        print(
            f"Chat error: {error}"
        )

        return jsonify({

            "success": False,

            "message": f"Gemini API error: {str(error)}"

        }), 500


# =========================================================
# 15. CLEAR MEMORY
# =========================================================

@app.route(
    "/clear-memory",
    methods=["POST"]
)
def clear_memory():

    try:

        data = request.get_json()

        session_id = data.get(
            "session_id"
        )

        if not session_id:

            return jsonify({

                "success": False,

                "message": "Session ID missing."

            }), 400

        clear_session_memory(
            session_id
        )

        return jsonify({

            "success": True,

            "message":
                "Conversation memory cleared."

        })

    except Exception as error:

        return jsonify({

            "success": False,

            "message": str(error)

        }), 500


# =========================================================
# 16. STATUS
# =========================================================

@app.route("/status")
def status():

    return jsonify({

        "app": "Memory + RAG Chatbot",

        "gemini_model": GEMINI_MODEL,

        "pdf_uploaded":
            uploaded_filename is not None,

        "pdf_name":
            uploaded_filename,

        "chunks":
            len(document_chunks)

    })


# =========================================================
# 17. RUN APP
# =========================================================

if __name__ == "__main__":

    print("")
    print("=" * 50)
    print("Memory + RAG Chatbot")
    print("=" * 50)
    print(
        f"Gemini Model: {GEMINI_MODEL}"
    )
    print(
        "Server: http://127.0.0.1:5000"
    )
    print("=" * 50)
    print("")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )