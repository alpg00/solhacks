import os
from dotenv import load_dotenv
load_dotenv()  # Load environment variables, if needed

import subprocess
import time
import webview
import atexit

import importlib.metadata
print("openai version:", importlib.metadata.version("openai"))

# Command to run the Streamlit app (dashboard.py)
command = ["streamlit", "run", "dashboard.py"]

# Start the Streamlit app as a background process
proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Ensure cleanup on exit
def cleanup():
    print("Terminating Streamlit process...")
    proc.terminate()
    proc.wait()

atexit.register(cleanup)

# Wait for the Streamlit server to start (adjust if needed)
time.sleep(5)

# Open the desktop popup window using PyWebView
window = webview.create_window("Fair Housing Loan Approval App", "http://localhost:8501", width=1000, height=800)
webview.start()
