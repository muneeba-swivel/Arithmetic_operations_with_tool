import os
import re
import fitz  # PyMuPDF
from sympy import sympify, symbols, solve
import math
from dotenv import load_dotenv
from openai import OpenAI
from langchain.tools import tool
from langchain.text_splitter import CharacterTextSplitter

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# --- Enhanced Tool Function to Solve Arithmetic and Financial Calculations ---
@tool
def solve_arithmetic(expression: str) -> str:
    """
    This tool solves various mathematical expressions including:
    - Basic arithmetic: addition, subtraction, multiplication, division
    - Exponentiation and roots
    - Percentage calculations
    - Simple financial calculations
    - Currency conversions (with given rates)
    
    Examples:
    - '45 + 32' → '77'
    - '70 + 67 + 18' → '155'
    - '100 * 1.05' → '105.0' (interest calculation)
    - '1000 * 0.12 / 12' → '10.0' (monthly calculation)
    """
    try:
        # Clean the expression
        expression = expression.strip()
        
        # Handle percentage calculations
        if '%' in expression:
            expression = expression.replace('%', '/100')
        
        # Handle common financial terms
        expression = re.sub(r'\bof\b', '*', expression, flags=re.IGNORECASE)
        
        # Evaluate the expression
        result = sympify(expression)
        
        # Return numeric result
        if result.is_number:
            return str(float(result))
        else:
            return str(result)
            
    except Exception as e:
        return f"Error solving expression '{expression}': {e}"

@tool  
def financial_calculator(calculation_type: str, **kwargs) -> str:
    """
    Specialized financial calculator for common accounting operations:
    - compound_interest: principal, rate, time, compounds_per_year
    - simple_interest: principal, rate, time
    - loan_payment: principal, rate, periods
    - future_value: present_value, rate, periods
    - present_value: future_value, rate, periods
    
    Example usage:
    financial_calculator("compound_interest", principal=1000, rate=0.05, time=2, compounds_per_year=12)
    """
    try:
        if calculation_type == "compound_interest":
            p = kwargs.get('principal', 0)
            r = kwargs.get('rate', 0)
            t = kwargs.get('time', 0)
            n = kwargs.get('compounds_per_year', 1)
            result = p * (1 + r/n) ** (n * t)
            return f"Compound Interest Result: {result:.2f}"
        
        elif calculation_type == "simple_interest":
            p = kwargs.get('principal', 0)
            r = kwargs.get('rate', 0)
            t = kwargs.get('time', 0)
            result = p * (1 + r * t)
            return f"Simple Interest Result: {result:.2f}"
        
        elif calculation_type == "loan_payment":
            p = kwargs.get('principal', 0)
            r = kwargs.get('rate', 0)
            n = kwargs.get('periods', 0)
            if r == 0:
                result = p / n
            else:
                result = p * (r * (1 + r)**n) / ((1 + r)**n - 1)
            return f"Monthly Payment: {result:.2f}"
            
        else:
            return f"Unknown calculation type: {calculation_type}"
            
    except Exception as e:
        return f"Error in financial calculation: {e}"

# --- PDF Reader ---
def extract_pdf_text(pdf_filename: str) -> str:
    """Extract text from PDF file in the docs folder."""
    pdf_path = os.path.join("docs", pdf_filename)
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file '{pdf_filename}' not found in 'docs' folder.")
    
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    doc.close()
    return text

# --- Text Chunker ---
def chunk_text(text: str, chunk_size: int = 2000) -> list:
    """Split text into manageable chunks for LLM processing."""
    splitter = CharacterTextSplitter(
        separator="\n", 
        chunk_size=chunk_size, 
        chunk_overlap=100
    )
    return splitter.split_text(text)

# --- Enhanced Arithmetic Detection ---
def detect_and_solve_arithmetic(text: str) -> tuple:
    """
    Detect arithmetic expressions in text and solve them.
    Returns (modified_text, arithmetic_results)
    """
    # Enhanced regex patterns for different types of calculations
    patterns = [
        r'(\d+(?:\.\d+)?\s*[\+\-\*/\^]\s*\d+(?:\.\d+)?)',  # Basic arithmetic
        r'(\d+(?:\.\d+)?%\s*of\s*\d+(?:\.\d+)?)',          # Percentage calculations
        r'(\d+(?:\.\d+)?\s*\*\s*\d+(?:\.\d+)?%)',          # Multiplication with percentage
        r'(\$?\d+(?:,\d{3})*(?:\.\d{2})?\s*[\+\-\*/]\s*\$?\d+(?:,\d{3})*(?:\.\d{2})?)', # Currency
    ]
    
    arithmetic_results = []
    modified_text = text
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            print(f"🔢 Detected arithmetic: {match}")
            # Clean the match for calculation
            cleaned_match = re.sub(r'[$,]', '', match)  # Remove $ and commas
            result = solve_arithmetic(cleaned_match)
            arithmetic_results.append(f"Calculation: {match} = {result}")
            # Replace in text with result for context
            modified_text = modified_text.replace(match, f"{match} (= {result})")
    
    return modified_text, arithmetic_results

# --- Enhanced LLM Query Processor ---
def llm_process(query: str, pdf_text: str) -> str:
    """Process user query with PDF content and handle arithmetic operations."""
    
    # First, detect and solve arithmetic in both PDF and query
    modified_pdf, pdf_arithmetic = detect_and_solve_arithmetic(pdf_text)
    modified_query, query_arithmetic = detect_and_solve_arithmetic(query)
    
    all_arithmetic = pdf_arithmetic + query_arithmetic
    
    # If arithmetic was found, include results in response
    arithmetic_summary = ""
    if all_arithmetic:
        arithmetic_summary = "\n🧮 Arithmetic Calculations Found:\n" + "\n".join(all_arithmetic) + "\n\n"
    
    # Chunk the PDF text to avoid token limit issues
    chunks = chunk_text(modified_pdf)
    combined_response = ""
    
    # Prepare system message with arithmetic awareness
    system_message = """You are a helpful assistant specialized in analyzing PDF documents. 
    You have access to arithmetic calculation tools for any mathematical operations found in the document.
    When you encounter numerical calculations, explain them clearly and reference the calculated results provided."""
    
    for i, chunk in enumerate(chunks):
        try:
            print(f"📄 Processing chunk {i+1}/{len(chunks)}...")
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"""
Document Chunk:
{chunk}

Arithmetic Calculations (if any):
{arithmetic_summary}

User Query: {modified_query}

Please provide a comprehensive answer based on the document content and calculated results.
"""}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            chunk_response = response.choices[0].message.content
            combined_response += chunk_response + "\n"
            
        except Exception as e:
            combined_response += f"❌ Error processing chunk {i+1}: {e}\n"
    
    # Combine arithmetic results with LLM response
    final_response = arithmetic_summary + combined_response.strip()
    return final_response

# --- Main Coordinator ---
def handle_query(pdf_filename: str, user_query: str):
    """Main function to handle user queries with PDF analysis."""
    try:
        print("📖 Extracting PDF text...")
        pdf_text = extract_pdf_text(pdf_filename)
        print(f"✅ Extracted {len(pdf_text)} characters from PDF")
        
        print("🤖 Processing query with LLM...")
        response = llm_process(user_query, pdf_text)
        return response
        
    except Exception as e:
        return f"❌ Error processing query: {e}"

# --- Enhanced Terminal Interface ---
def main():
    """Main terminal interface for the PDF Query Assistant."""
    print("🚀 PDF Query Assistant with Arithmetic Solver")
    print("=" * 50)
    
    # Get PDF filename
    pdf_filename = input("📁 Enter the name of the PDF file (in 'docs' folder): ").strip()
    
    # Validate PDF exists
    try:
        extract_pdf_text(pdf_filename)
        print("✅ PDF loaded successfully!")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    except Exception as e:
        print(f"❌ Error loading PDF: {e}")
        return
    
    print("\n🎯 You can now ask questions about the PDF content.")
    print("💡 The assistant will automatically solve any arithmetic operations found.")
    print("📝 Example queries:")
    print("   - 'What is the total revenue calculation?'")
    print("   - 'Calculate the percentage increase mentioned'")
    print("   - 'What are the financial projections?'")
    print("\nType 'exit' to quit.\n")
    
    while True:
        user_query = input("🧠 Your question: ").strip()
        
        if user_query.lower() in ['exit', 'quit', 'bye']:
            print("👋 Thank you for using PDF Query Assistant. Goodbye!")
            break
        
        if not user_query:
            print("⚠️ Please enter a question.")
            continue
        
        try:
            print("\n⏳ Processing your query...")
            response = handle_query(pdf_filename, user_query)
            print(f"\n💬 Response:\n{response}\n")
            print("-" * 50)
            
        except Exception as e:
            print(f"⚠️ Error: {e}\n")

if __name__ == "__main__":
    # Ensure docs directory exists
    os.makedirs("docs", exist_ok=True)
    main()