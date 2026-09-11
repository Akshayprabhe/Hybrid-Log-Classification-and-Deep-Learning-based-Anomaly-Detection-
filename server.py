import os
import io
import pandas as pd
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from classify import classify

app = FastAPI()

# HTML directly in python - no templates folder needed
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Log Classification SIH</title>
    <style>
        body { font-family: Arial; padding: 40px; background: #f5f5f5; }
       .container { background: white; padding: 30px; border-radius: 10px; max-width: 1000px; margin: auto; }
        h1 { color: #2c3e50; }
        button { background: #3498db; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; }
        th { background: #3498db; color: white; }
       .badge { background: #27ae60; color: white; padding: 4px 8px; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Log Classification System - SIH</h1>
        <p>Upload CSV with `source` and `log_message` columns</p>
        <input type="file" id="csvFile" accept=".csv">
        <button onclick="uploadFile()">Classify Logs</button>
        <div id="result"></div>
    </div>
<script>
async function uploadFile() {
    const fileInput = document.getElementById('csvFile');
    const file = fileInput.files[0];
    if (!file) { alert("Please select a file"); return; }
    const formData = new FormData();
    formData.append("file", file);
    const res = await fetch("/classify/", { method: "POST", body: formData });
    const data = await res.json();
    let table = "<table><tr><th>Source</th><th>Log Message</th><th>Target Label</th></tr>";
    data.data.forEach(row => {
        table += `<tr><td>${row.source}</td><td>${row.log_message}</td><td><span class="badge">${row.target_label}</span></td></tr>`;
    });
    table += "</table>";
    document.getElementById("result").innerHTML = table;
}
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML_PAGE

@app.post("/classify/")
async def classify_logs(file: UploadFile):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV.")
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # FIX: accept Source / SOURCE also
        df.columns = [c.lower().strip() for c in df.columns]

        if "source" not in df.columns or "log_message" not in df.columns:
            raise HTTPException(status_code=400, detail=f"CSV must contain 'source' and 'log_message' columns. Found: {list(df.columns)}")

        # FIX: Never crash even if Groq fails
        try:
            df["target_label"] = classify(list(zip(df["source"], df["log_message"])))
        except Exception as e:
            print(f"Classify failed but continuing: {e}")
            df["target_label"] = "Unclassified"

        os.makedirs("resources", exist_ok=True)
        df.to_csv("resources/output.csv", index=False)

        return {"data": df.head(50).to_dict(orient="records")}

    except HTTPException as he:
        raise he
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"data": [{"source": "System", "log_message": f"Error: {e}", "target_label": "Failed"}]}
    finally:
        await file.close()
