import os
import re
import numpy as np
import pandas as pd
import pdfplumber
import pytesseract
import pypdfium2 as pdfium
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from collections import defaultdict

MODEL_NAME = "jhgan/ko-sbert-multitask"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

STOP_PATTERNS = [
    "성실히 참여", "적극적으로 참여", "열심히 임함",
    "바람직한 태도", "꾸준히 참여", "책임감을 보임"
]

CREATIVE_WEIGHTS = {"동아리": 1.0, "진로": 0.8, "자율": 0.6}


# 1. 과목 패턴 (세특 과목 분리용)
SUBJECT_PATTERN = re.compile(r"""
(
  국어|국어Ⅰ|국어Ⅱ|문학|독서|화법과작문|언어와매체|
  수학|수학Ⅰ|수학Ⅱ|미적분|기하|확률\s*과\s*통계|고급수학|
  영어|영어Ⅰ|영어Ⅱ|
  한국사|
  통합사회|사회·문화|생활과윤리|윤리와사상|정치와법|
  통합과학|과학탐구실험|
  물리학|물리학Ⅰ|물리학Ⅱ|
  화학|화학Ⅰ|화학Ⅱ|
  생명과학|생명과학Ⅰ|생명과학Ⅱ|
  지구과학|지구과학Ⅰ|지구과학Ⅱ|
  정보|정보과학|컴퓨터과학|프로그래밍
)
\s*[:：]
""", re.VERBOSE)


# 2. 전공군-과목 매핑 (Top-3)
MAJOR_FIELD_MAP = {
    "컴퓨터공학": {
        "keywords": ["컴퓨터", "소프트웨어", "컴공", "정보공학"],
        "subjects": {"정보": 1.0, "프로그래밍": 1.0, "수학": 0.8, "확률과통계": 0.7}
    },
    "인공지능": {
        "keywords": ["AI", "인공지능", "머신러닝", "딥러닝"],
        "subjects": {"정보": 1.0, "확률과통계": 1.0, "수학": 0.9, "미적분": 0.8}
    },
    "데이터사이언스": {
        "keywords": ["데이터", "데이터사이언스", "빅데이터", "통계"],
        "subjects": {"확률과통계": 1.0, "수학": 0.9, "정보": 0.8, "통합사회": 0.5}
    },
    "환경·지속가능": {
        "keywords": ["환경", "에너지", "기후", "지속가능", "생태"],
        "subjects": {"통합과학": 1.0, "생명과학": 0.8, "통합사회": 0.8, "수학": 0.6}
    },
    "바이오의생명": {
        "keywords": ["바이오", "의생명", "의공학", "생명공학"],
        "subjects": {"생명과학": 1.0, "화학": 0.9, "확률과통계": 0.6, "수학": 0.6}
    },
    "경영": {
        "keywords": ["경영", "경영학", "회계", "마케팅"],
        "subjects": {"수학": 1.0, "확률과통계": 0.9, "통합사회": 0.7, "국어": 0.5}
    },
    "경제": {
        "keywords": ["경제", "경제학", "금융", "국제통상"],
        "subjects": {"수학": 1.0, "확률과통계": 1.0, "통합사회": 0.8}
    },
    "행정·정책": {
        "keywords": ["행정", "정책", "공공", "정치"],
        "subjects": {"통합사회": 1.0, "국어": 0.8, "확률과통계": 0.6}
    },
    "국어·문학": {
        "keywords": ["국어", "문학", "국문", "한국어"],
        "subjects": {"국어": 1.0, "문학": 1.0, "독서": 0.9}
    },
    "미디어·콘텐츠": {
        "keywords": ["미디어", "콘텐츠", "영상", "언론", "방송"],
        "subjects": {"국어": 0.8, "독서": 0.8, "정보": 0.6, "통합사회": 0.6}
    }
}


# 3. PDF 텍스트 추출
def extract_text_from_pdf(pdf_path):
    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            pt = page.extract_text()
            if pt and len(re.findall(r"[가-힣]", pt)) > 30:
                pages_text.append(pt)
            else:
                doc = pdfium.PdfDocument(pdf_path)
                img = doc[i].render_to(pdfium.BitmapConv.pil_image, scale=2)
                pages_text.append(pytesseract.image_to_string(img, lang="kor"))
    text = "\n".join(pages_text)

    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)  
    text = re.sub(r"\n{3,}", "\n\n", text) 
    return text.strip()


# 4. 세특 과목별 내용 분리 (subject_dict)
def build_subject_se_teuk_dict(text):
    matches = list(SUBJECT_PATTERN.finditer(text))
    subject_dict = {}
    for i, m in enumerate(matches):
        subject = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if len(content) >= 50:
            subject_dict[subject] = content
    return subject_dict


# 5. 문장 분리
def split_sentences(text):
    raw = re.split(r"(?<=[다함됨임]\.)\s+|\n+", text)
    out = []
    for s in raw:
        s = s.strip()
        if len(s) < 20:
            continue
        if any(p in s for p in STOP_PATTERNS):
            continue
        out.append(s)
    return out


# 6. 창체 분리- "자율활동/동아리활동/진로활동" 같은 표식이 등장하면 섹션 전환
def split_creative_sections_from_sentences(sentences):
    sections = {"자율": [], "동아리": [], "진로": []}
    current = None

    for s in sentences:
        if ("자율활동" in s) or ("자율 활동" in s) or ("자율" in s and "활동" in s):
            current = "자율"
        if ("동아리활동" in s) or ("동아리 활동" in s) or ("동아리" in s and "활동" in s):
            current = "동아리"
        if ("진로활동" in s) or ("진로 활동" in s) or ("진로희망" in s) or ("진로" in s and "활동" in s):
            current = "진로"

        if current:
            sections[current].append(s)

    return sections


# 7. 전공군 Top-K 추론 (학과명 + majors_db 키워드 사용)
def infer_major_fields_topk(major_name, major_keywords=None, top_k=3):
    text = (major_name or "")
    if major_keywords:
        text += " " + " ".join(major_keywords)

    flat = text.replace(" ", "")

    scores = []
    for field, info in MAJOR_FIELD_MAP.items():
        score = 0
        for kw in info["keywords"]:
            if kw.replace(" ", "") in flat:
                score += 1
        if score > 0:
            scores.append((field, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


# 8. 임베딩 캐시 (학생문장 / 과목세특 / 전공DB)
def embed_sentences_once(sentences, model):
    if not sentences:
        return {}
    embs = model.encode(sentences, normalize_embeddings=True)
    return dict(zip(sentences, embs))

def embed_subject_contents_once(subject_dict, model):
    if not subject_dict:
        return {}
    subjects = list(subject_dict.keys())
    contents = [subject_dict[s] for s in subjects]
    embs = model.encode(contents, normalize_embeddings=True)
    return dict(zip(subjects, embs))

def embed_majors_once(majors, model):
    texts = []
    for m in majors:
        t = (str(m.get("major_desc", "")) + " " + " ".join(m.get("keywords", []))).strip()
        if not t:
            t = str(m.get("major", ""))
        texts.append(t)
    return model.encode(texts, normalize_embeddings=True)


# 9. 전공별 학생 벡터 (캐시 기반)
def build_major_specific_student_emb(sentences, major_emb, sentence_embs, top_k=10):
    if not sentences:
        return None

    sims = []
    for s in sentences:
        emb = sentence_embs.get(s)
        if emb is None:
            continue
        sims.append((float(np.dot(emb, major_emb)), emb))

    if not sims:
        return None

    sims.sort(key=lambda x: x[0], reverse=True)
    top_embs = [e for _, e in sims[:top_k]]
    return np.mean(top_embs, axis=0)


# 10. 세특 과목 점수 (전공군 Top-3 + 과목가중치 + 캐싱)
def calc_subject_content_score_cached(subject_embs, major_emb, major_name, major_keywords):
    fields = infer_major_fields_topk(major_name, major_keywords, top_k=3)
    if not fields:
        return 0.0

    scores, weights = [], []

    for field, field_score in fields:
        subj_weights = MAJOR_FIELD_MAP[field]["subjects"]

        for sub_key, w in subj_weights.items():
            key_flat = sub_key.replace(" ", "")
            for real_sub, emb in subject_embs.items():
                if key_flat in real_sub.replace(" ", ""):
                    scores.append(float(np.dot(emb, major_emb)))
                    weights.append(w * field_score)

    return float(np.average(scores, weights=weights)) if scores else 0.0


# 11. 창체 점수
def calc_creative_weighted_score(creative_sections, major_emb, sentence_embs):
    scores, weights = [], []
    for sec, sents in creative_sections.items():
        if not sents:
            continue

        emb = build_major_specific_student_emb(sents, major_emb, sentence_embs, top_k=10)
        if emb is None:
            continue

        score = (float(np.dot(emb, major_emb)) + 1) * 50
        scores.append(score)
        weights.append(CREATIVE_WEIGHTS.get(sec, 0.5))

    return float(np.average(scores, weights=weights)) if scores else None


# 12. 근거 요약
def summarize_evidence_one_line(evidences, kw_model):
    if not evidences:
        return "전공 관련 학습 성향이 생활기록부 전반에서 확인됨"

    text = " ".join(evidences)
    keywords = [
        k for k, _ in kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            use_mmr=True,
            diversity=0.5,
            top_n=3
        )
    ]

    if not keywords:
        return "전공과 연계된 학습 경험이 생활기록부에 드러남"

    return f"{', '.join(keywords)} 등을 중심으로 전공 적합성이 드러남"


# 13. 추천 엔진 (세특/창체 50:50)
def recommend_by_school(raw_text, subject_dict, majors, model, kw_model):
    sentences = split_sentences(raw_text)

    # 캐싱
    sentence_embs = embed_sentences_once(sentences, model)
    subject_embs = embed_subject_contents_once(subject_dict, model)
    major_embs = embed_majors_once(majors, model)

    creative_sections = split_creative_sections_from_sentences(sentences)

    grouped = defaultdict(list)

    for idx, m in enumerate(majors):
        major_emb = major_embs[idx]

        # 세특 점수
        se_emb = build_major_specific_student_emb(sentences, major_emb, sentence_embs, top_k=10)
        if se_emb is None:
            continue

        se_sem = (float(np.dot(se_emb, major_emb)) + 1) * 50
        se_sub = (calc_subject_content_score_cached(
            subject_embs,
            major_emb,
            m.get("major", ""),
            m.get("keywords", [])
        ) + 1) * 50

        se_score = 0.7 * se_sem + 0.3 * se_sub

        # 창체 점수
        cc_score = calc_creative_weighted_score(creative_sections, major_emb, sentence_embs)
        if cc_score is None:
            cc_score = se_sem

        # 최종 점수
        final = round(0.5 * se_score + 0.5 * cc_score, 2)

        # 전공군 라벨
        fields = infer_major_fields_topk(m.get("major", ""), m.get("keywords", []), top_k=3)
        field_label = " + ".join([f[0] for f in fields]) if fields else "미분류"

        # Evidence top3 (캐시로만 계산)
        evid = sorted(
            sentences,
            key=lambda s: float(np.dot(sentence_embs[s], major_emb)),
            reverse=True
        )[:3]

        grouped[m.get("university", "UNKNOWN")].append({
            "major": m.get("major", ""),
            "final": final,
            "se": round(se_score, 2),
            "cc": round(cc_score, 2),
            "field": field_label,
            "evidence": evid
        })

    for uni in grouped:
        grouped[uni] = sorted(grouped[uni], key=lambda x: x["final"], reverse=True)[:5]

    return grouped



if __name__ == "__main__":
    pdf_path = input("📄 생활기록부 PDF 경로 입력: ").strip().strip('"')
    if not os.path.exists(pdf_path):
        print("❌ 파일이 존재하지 않습니다.")
        raise SystemExit(1)

    model = SentenceTransformer(MODEL_NAME)
    kw_model = KeyBERT(model)

    raw = extract_text_from_pdf(pdf_path)
    subject_dict = build_subject_se_teuk_dict(raw)
    majors = pd.read_csv("majors_db.csv").to_dict("records")

    results = recommend_by_school(raw, subject_dict, majors, model, kw_model)

    for uni, items in results.items():
        print(f"\n🏫 {uni} 추천 학과 TOP 5")
        for i, r in enumerate(items, 1):
            reason = summarize_evidence_one_line(r["evidence"], kw_model)
            print(f"[{i}] {r['major']} (총점 {r['final']}점 / 전공군: {r['field']})")
            print(f"   ├ 세특 점수: {r['se']}점")
            print(f"   ├ 창체 점수: {r['cc']}점")
            print(f"   └ 근거 요약: {reason}")
            print("-" * 60)
