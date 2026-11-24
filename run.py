import webbrowser
import threading
import uvicorn
from app.config import PORT

if __name__ == "__main__":
    threading.Timer(2.0, lambda: webbrowser.open(f"http://127.0.0.1:{PORT}")).start()
    uvicorn.run("app.main:app", host="127.0.0.1", port=PORT)
