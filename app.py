import subprocess
import time
import webview
import atexit

# ----------------------------
# 1. Launch the Streamlit App
# ----------------------------
# This command will start the Streamlit app (dashboard.py)
command = ["streamlit", "run", "dashboard.py"]
proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# ----------------------------
# 2. Ensure Cleanup on Exit
# ----------------------------
def cleanup():
    print("Terminating Streamlit process...")
    proc.terminate()
    proc.wait()

atexit.register(cleanup)

# ----------------------------
# 3. Wait for the Streamlit Server to Start
# ----------------------------
# Adjust the sleep time if your app takes longer to start.
time.sleep(5)

# ----------------------------
# 4. Open the Desktop Window with PyWebView
# ----------------------------
window = webview.create_window("Fair Housing Loan Approval App", "http://localhost:8501", width=800, height=600)
webview.start()  # This will block until you close the window.
