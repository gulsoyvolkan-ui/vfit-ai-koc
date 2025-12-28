import importlib
import sys

print(f"Python version: {sys.version}")

try:
    print("Testing langchain_community.document_loaders...")
    from langchain_community.document_loaders import PyPDFDirectoryLoader
    print("Success.")
except Exception as e:
    print(f"Failed: {e}")

try:
    print("Testing langchain_google_genai...")
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    print("Success.")
except Exception as e:
    print(f"Failed: {e}")

try:
    print("Testing langchain_community.vectorstores...")
    from langchain_community.vectorstores import Chroma
    print("Success.")
except Exception as e:
    print(f"Failed: {e}")
