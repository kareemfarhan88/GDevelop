# Security Policy

## Supported Versions

Use this section to tell people about which versions of your project are
currently being supported with security updates.

| Version | Supported          |
| ------- | ------------------ |
| 5.1.x   | :white_check_mark: |
| 5.0.x   | :x:                |
| 4.0.x   | :white_check_mark: |
| < 4.0   | :x:                |

## Reporting a Vulnerability

Use this section to tell people how to report a vulnerability.

Tell them where to go, how often they can expect to get an update on a
reported vulnerability, what to expect if the vulnerability is accepted or
declined, etc.

# smart_helper.py
# يعتمد على:
# pip install flask openai python-dotenv transformers sentence-transformers torch

from flask import Flask, request, jsonify
import os, json

# نحاول استخدام OpenAI أولاً، لو ما فيه مفتاح نستخدم الذكاء المحلي
try:
    from dotenv import load_dotenv
    import openai
except:
    openai = None

from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import torch

app = Flask(__name__)

# ========== إعداد الذكاء الاصطناعي المحلي ==========
sentiment_pipe = pipeline("sentiment-analysis", model="avichr/heBERT_sentiment") if torch.cuda.is_available() else pipeline("sentiment-analysis")
embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
generator = pipeline("text-generation", model="google/flan-t5-small", device=0 if torch.cuda.is_available() else -1)

CATEGORY_EXAMPLES = {
    "مشكلة تقنية": ["البرنامج لا يعمل", "يوجد خطأ", "اللعبة لا تفتح", "تعليق في التشغيل"],
    "طلب ميزة": ["أريد إضافة ميزة", "لو تضيفون خيار جديد", "أتمنى تضيفون"],
    "شكوى": ["التطبيق سيء", "التجربة سيئة", "ما عجبني", "مشكلة في الأداء"],
    "سؤال عام": ["كيف أستخدم", "متى يصدر", "ما معنى"],
    "إساءة": ["غبي", "تافه", "برنامج فاشل"]
}
category_embeddings = {k: embedder.encode(v, convert_to_tensor=True).mean(dim=0) for k, v in CATEGORY_EXAMPLES.items()}

def classify_category(comment):
    emb = embedder.encode(comment, convert_to_tensor=True)
    best_cat, best_score = None, -1.0
    for cat, cat_emb in category_embeddings.items():
        score = util.cos_sim(emb, cat_emb).item()
        if score > best_score:
            best_score = score
            best_cat = cat
    return best_cat, float(best_score)

def local_ai_solution(comment, category, sentiment_label):
    prompt = f"""
التعليق: "{comment}"
التصنيف: {category}
المشاعر: {sentiment_label}
أعطني حلًا قصيرًا وردًا مهذبًا للمستخدم.
"""
    out = generator(prompt, max_length=150, do_sample=False)
    text = out[0]["generated_text"]
    return text.strip()

# ========== إعداد OpenAI إن وجد ==========
try:
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if OPENAI_API_KEY:
        openai.api_key = OPENAI_API_KEY
except Exception:
    OPENAI_API_KEY = None

def openai_solution(comment):
    prompt = f"""
التعليق العربي: "{comment}"

رجاءً أعطني JSON منسق يحتوي:
- "category": نوع المشكلة (مثلاً: مشكلة تقنية / طلب ميزة / شكوى / سؤال / إساءة)
- "sentiment": إيجابي أو سلبي أو محايد
- "solution": خطوات الحل المقترحة
- "reply": رد محترم للمستخدم
"""
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "أنت مساعد ذكي متخصص في فهم اللغة العربية وتحليل التعليقات."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=400
        )
        text = resp["choices"][0]["message"]["content"].strip()
        return json.loads(text)
    except Exception as e:
        return {"error": str(e)}

# ========== نقطة الوصول الرئيسية ==========
@app.route("/solve", methods=["POST"])
def solve():
    data = request.json or {}
    comment = data.get("comment", "").strip()
    if not comment:
        return jsonify({"error": "يجب إدخال تعليق"}), 400

    # لو عندنا مفتاح OpenAI نستخدمه
    if OPENAI_API_KEY:
        result = openai_solution(comment)
        if "error" not in result:
            return jsonify(result)

    # الذكاء المحلي كبديل
    sent = sentiment_pipe(comment)[0]
    sentiment = sent.get("label", "neutral")
    score = float(sent.get("score", 0.0))
    category, cat_score = classify_category(comment)
    solution = local_ai_solution(comment, category, sentiment)

    return jsonify({
        "category": category,
        "sentiment": sentiment,
        "sentiment_score": score,
        "solution_and_reply": solution
    })

pip install flask openai python-dotenv transformers sentence-transformers torch
OPENAI_API_KEY=sk-xxxxxxxx
curl -X POST http://localhost:5000/solve -H "Content-Type: application/json" -d "{\"comment\":\"اللعبة ما تشتغل عندي على ويندوز 11\"}"


if __name__ == "__main__":
    app.run(port=5000, debug=True)
