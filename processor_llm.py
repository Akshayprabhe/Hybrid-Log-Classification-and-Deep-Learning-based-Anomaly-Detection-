import re
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("ERROR: GROQ_API_KEY not found in.env file!")
    # Don't crash, create dummy client that will fallback
    groq = None
else:
    groq = Groq(api_key=api_key)

def classify_with_llm(log_msg):
    if groq is None:
        return "Unclassified"

    MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "llama-3.1-8b-instant"]
    prompt = f'Classify into Workflow Error, Deprecation Warning, Unclassified. Log: "{log_msg}"\n\nRespond like <category>Workflow Error</category>'

    for model_name in MODELS:
        try:
            res = groq.chat.completions.create(
                messages=[{"role":"user","content":prompt}],
                model=model_name,
                temperature=0
            )
            content = res.choices[0].message.content
            match = re.search(r'<category>(.*?)</category>', content, re.DOTALL)
            if match:
                return match.group(1).strip()
            return content.strip()
        except Exception as e:
            print(f"{model_name} failed: {e}")
            continue

    return "Unclassified"
