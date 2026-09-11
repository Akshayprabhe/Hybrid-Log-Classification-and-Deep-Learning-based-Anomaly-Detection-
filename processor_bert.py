# --- START: Crash-proof wrapper for college laptop DLL block ---
BERT_AVAILABLE = False
model_embedding = None
model_classification = None

try:
    import joblib
    from sentence_transformers import SentenceTransformer

    try:
        model_embedding = SentenceTransformer('all-MiniLM-L6-v2')
        model_classification = joblib.load("models/log_classifier.joblib")
        BERT_AVAILABLE = True
        print("BERT loaded successfully")
    except Exception as e:
        print(f"BERT model load failed, will skip BERT: {e}")
        BERT_AVAILABLE = False

except Exception as e:
    print(f"CRITICAL DLL BLOCKED - BERT disabled, using Regex+LLM only: {e}")
    BERT_AVAILABLE = False

def classify_with_bert(log_message):
    # If BERT not available due to DLL block, don't crash
    if not BERT_AVAILABLE or model_embedding is None or model_classification is None:
        return "Unclassified"

    try:
        embeddings = model_embedding.encode([log_message])
        probabilities = model_classification.predict_proba(embeddings)[0]
        if max(probabilities) < 0.5:
            return "Unclassified"
        predicted_label = model_classification.predict(embeddings)[0]
        return predicted_label
    except Exception as e:
        print(f"BERT predict error, fallback to Unclassified: {e}")
        return "Unclassified"
# --- END ---

if __name__ == "__main__":
    logs = [
        "alpha.osapi_compute.wsgi.server - 12.10.11.1 - API returned 404 not found error",
        "GET /v2/3454/servers/detail HTTP/1.1 RCODE 404 len: 1583 time: 0.1878400",
        "System crashed due to drivers errors when restarting the server",
        "Hey bro, chill ya!",
        "Multiple login failures occurred on user 6454 account",
        "Server A790 was restarted unexpectedly during the process of data transfer"
    ]
    for log in logs:
        label = classify_with_bert(log)
        print(log, "->", label)
