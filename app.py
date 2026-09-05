# Import necessary libraries
import os, re, time
import PyPDF2
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found. Add it to your .env file.")

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from flask import Flask, render_template, request, redirect
from PyPDF2 import PdfReader

#Please install PdfReader
from groq import Groq

from langchain_text_splitters import CharacterTextSplitter
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from groq import APIStatusError, APIConnectionError
from langchain_groq import ChatGroq
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain


start_greeting = ["hi","hello"]
end_greeting = ["bye"]
way_greeting = ["who are you?"]

#Using this folder for storing the uploaded docs. Creates the folder at runtime if not present
DATA_DIR = "__data__"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

#Flask App
app = Flask(__name__)

vectorstore = None
conversation_chain = None
chat_history = []
rubric_text = ""

groq_client = Groq(api_key=GROQ_API_KEY)

# Retryable errors: Groq's rate-limit (429) and transient connection errors.
# Local embeddings need no retry at all since they don't hit the network.
RETRYABLE_ERRORS = (APIStatusError, APIConnectionError)

class HumanMessage:
    def __init__(self, content):
        self.content = content
    
    def __repr__(self):
        return f'HumanMessage(content={self.content})'

class AIMessage:
    def __init__(self, content):
        self.content = content
    
    def __repr__(self):
        return f'AIMessage(content={self.content})'


def _is_rate_limit_error(exc):
    msg = str(exc)
    return '429' in msg or 'rate_limit' in msg.lower() or 'rate limit' in msg.lower()

def _adaptive_wait(retry_state):
    exc = retry_state.outcome.exception()
    if exc is not None and _is_rate_limit_error(exc):
        # Free tier RPM limits reset roughly every minute; wait past the window.
        return 65
    return wait_exponential(multiplier=1, min=0.5, max=4)(retry_state)


def get_pdf_text(pdf_docs):
    text = ""
    pdf_txt = ""
    for pdf in pdf_docs:
        filename = os.path.join(DATA_DIR,pdf.filename)
        pdf_txt = ""
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
            pdf_txt += page.extract_text()

        with (open(filename, "w", encoding="utf-8")) as op_file:
            op_file.write(pdf_txt)

    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    # Local, free, unlimited embeddings — runs on this machine, no API calls,
    # no rate limits. Downloads the model once (~90MB) on first run.
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=GROQ_API_KEY
    )
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def _grade_essay(essay):
    system_instruction = (
        "You are an academic essay grading assistant. "
        "Carefully evaluate the essay based on the rubric provided below. "
        "Respond in English only using clean GitHub-Flavored Markdown (GFM).\n\n"
        "Your response MUST include:\n"
        "1. Overall Score and Letter Grade (bolded at the top).\n"
        "2. A properly formatted Markdown table with EXACTLY these columns:\n"
        "| Criterion | Score (out of X) | Comments |\n"
        "|---|---:|---|\n"
        "Every row MUST have exactly 3 columns separated by vertical bars (|).\n"
        "Do NOT combine the column names. Do NOT omit the vertical bars.\n"
        "3. Clear sections with '##' headings for: Strengths, Areas for Improvement, and Actionable Recommendations.\n\n"
        f"RUBRIC:\n{rubric_text if rubric_text.strip() else 'Standard Academic Rubric (Content, Structure, Evidence, Mechanics)'}"
    )

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": f"ESSAY:\n{essay}"}
    ]

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
        temperature=0.4,
        max_tokens=2000
    )

    data = response.choices[0].message.content

    return data
@app.route('/')
def home():
    return render_template('new_home.html')


@app.route('/process', methods=['POST'])
def process_documents():
    global vectorstore, conversation_chain
    pdf_docs = request.files.getlist('pdf_docs')
    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    conversation_chain = get_conversation_chain(vectorstore)
    return redirect('/chat')

@retry(
    stop=stop_after_attempt(3),
    wait=_adaptive_wait,
    retry=retry_if_exception_type(RETRYABLE_ERRORS),
    reraise=True
)
def _ask_conversation_chain(question):
    return conversation_chain({'question': question})


@app.route('/chat', methods=['GET', 'POST'])
def chat():
    global vectorstore, conversation_chain, chat_history
    msgs = []
    
    if request.method == 'POST':
        user_question = request.form['user_question']
        
        try:
            response = _ask_conversation_chain(user_question)
            chat_history = response['chat_history']
        except RETRYABLE_ERRORS as e:
            print(f"[/chat] Groq call failed: {type(e).__name__}: {e}")
            chat_history = chat_history + [
                HumanMessage(content=user_question),
                AIMessage(content="Sorry, the AI service hit a temporary error. Please try asking again.")
            ]
        
    return render_template('new_chat.html', chat_history=chat_history)

@app.route('/pdf_chat', methods=['GET', 'POST'])
def pdf_chat():
    return render_template('new_pdf_chat.html')

@app.route('/essay_grading', methods=['GET', 'POST'])
def essay_grading():
    global rubric_text

    result = None
    text = ""

    if request.method == 'POST':

        submitted_rubric = request.form.get('rubric_text', '').strip()

        if submitted_rubric:
            rubric_text = submitted_rubric

        essay_rubric = request.form.get('essay_rubric', '').strip()

        if essay_rubric:
            rubric_text = essay_rubric

        essay_text = request.form.get('essay_text', '').strip()

        if essay_text:
            text = essay_text
            result = _grade_essay(text)

        elif 'file' in request.files:
            file = request.files['file']

            if file and file.filename.lower().endswith('.pdf'):
                reader = PyPDF2.PdfReader(file)

                text = ""

                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"

                if text.strip():
                    result = _grade_essay(text)
                else:
                    result = "Unable to extract text from the PDF."

    return render_template(
        'new_essay_grading.html',
        result=result,
        input_text=text,
        rubric_text=rubric_text
    )
@app.route('/essay_rubric', methods=['GET', 'POST'])
def essay_rubric():
    return render_template('new_essay_rubric.html')

def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ''
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    return text

if __name__ == '__main__':
    app.run(debug=True)