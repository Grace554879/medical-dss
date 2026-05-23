import subprocess
import time

#Start Flask
flask_process = subprocess.Popen(["python", "app.py"])
time.sleep(3)  # wait for Flask to start

#Start Streamlit
streamlit_process = subprocess.Popen(["streamlit", "run", "dss_app.py"])

#Keep both running until you stop manually
try:
    flask_process.wait()
    streamlit_process.wait()
except KeyboardInterrupt:
    flask_process.terminate()
    streamlit_process.terminate()