import os
import re
import numpy as np
import pandas as pd
import fitz
import pytesseract
from PIL import Image
import io
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import ast
import tkinter as tk
from tkinter import filedialog
import time
import platform
import sys

# 환경, 경로 및 상수 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "majors_db.csv")
MODEL_NAME = "jhgan/ko-sbert-multitask"

def setup_tesseract():
    if platform.system() == "Windows":
        possible_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.join(os.getenv("LOCALAPPDATA", ""), r"Tesseract-OCR\tesseract.exe")
        ]
        if pytesseract.pytesseract.tesseract_cmd != 'tesseract':
             return True
        for path in possible_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                return True
    return False

# 노이즈 제거 -> 추가할 필요 있음
STOP_PATTERNS = [
    "성실히 참여", "적극적으로 참여", "열심히 임함", "바람직한 태도", "꾸준히 참여", 
    "책임감을 보임", "우수한 성적", "모범이 됨", "탁월함", "돋보임", "능동적인", 
    "자기주도적인", "긍정적인", "인상적임", "훌륭함", "성장함", "함양함",
    "수강자수", "석차등급", "표준편차", "이수단위", "원점수", "과목평균", 
    "세부능력 및 특기사항", "세부능력및특기사항", "세 부 능 력", "특 기 사 항",
    "창의적체험활동상황", "창의적 체험활동상황", "행동특성 및 종합의견",
    "학년", "학기", "교과", "과목", "성취도", "영역", "시간", "반", "번호", "이름",
    "담당교사", "담임교사", "학교장", "직인", "생략", "이상", "이하", "여백",
    "학교폭력", "예방교육", "안전교육", "심폐소생술", "소방안전", "재난대비",
    "성인지", "장애이해", "흡연예방", "자살예방", "봉사활동", "캠페인", "정화활동", "멘토링"
]

HEADER_PATTERNS = re.compile(r"""(
    \d{4}년\s*\d{1,2}월\s*\d{1,2}일|
    \d{1,2}\s*/\s*\d{1,2}|
    이\s*반|반\s*\d+|번호\s*\d+|
    이름\s*[가-힣]+|
    [가-힣]+고등학교
)""", re.VERBOSE)

# 과목 추출 패턴
SUBJECT_PATTERN = re.compile(r"""
(
  국어[ⅠⅡ]?|문학|독서|화법과\s*작문|언어와\s*매체|
  수학[ⅠⅡ]?|미적분|기하|확률\s*과\s*통계|고급수학|경제수학|수학과제\s*탐구|수학적\s*사고와\s*적분|
  영어[ⅠⅡ]?|영어\s*회화|영어\s*독해와\s*작문|심화\s*영어\s*독해|
  한국사|
  통합사회|사회[·\.]?문화|생활과\s*윤리|윤리와\s*사상|정치와\s*법|한국지리|세계지리|사회문제\s*탐구|
  통합과학|과학탐구실험|
  물리학[ⅠⅡ]?|화학[ⅠⅡ]?|생명과학[ⅠⅡ]?|지구과학[ⅠⅡ]?|물리학\s*실험|
  정보|정보과학|컴퓨터과학|프로그래밍|인공지능\s*기초
)
\s*[:：]
""", re.VERBOSE)

# 가중치 설정-> 필요하면 수정
STEM_SUBJECTS = ["수학", "미적분", "기하", "물리학", "화학", "생명과학", "지구과학", "정보", "소프트웨어", "프로그래밍", "과학탐구", "인공지능"]
HUMANITIES_SUBJECTS = ["국어", "문학", "언어와매체", "화법과작문", "영어", "한국사", "통합사회", "생활과윤리", "윤리와사상", "정치와법", "경제", "사회문화", "지리", "세계사"]

MAJOR_FIELD_MAP = {
    # 이공계열
    "컴퓨터공학": {
        "keywords": ["소프트웨어", "코딩", "알고리즘", "보안", "네트워크", "서버", "개발", "앱", "웹", "시스템", "클라우드", "블록체인", "자료구조", "운영체제", "해킹", "정보보호", "가상현실", "VR", "AR", "메타버스", "게임"],
        "type": "STEM"
    },
    "인공지능": {
        "keywords": ["AI", "머신러닝", "딥러닝", "데이터사이언스", "빅데이터", "신경망", "로봇지능", "자연어", "비전", "인식", "예측", "모델링", "텐서", "파이썬", "R", "데이터마이닝"],
        "type": "STEM"
    },
    "전자반도체": {
        "keywords": ["회로", "반도체", "임베디드", "신호처리", "통신", "전기", "디스플레이", "IoT", "사물인터넷", "센서", "아두이노", "라즈베리파이", "집적", "소자", "공정", "나노기술", "광학", "무선"],
        "type": "STEM"
    },
    "기계로봇": {
        "keywords": ["역학", "설계", "자동차", "항공", "제어", "로봇", "메카트로닉스", "드론", "자율주행", "3D프린팅", "CAD", "모델링", "유체", "열역학", "엔진", "모빌리티", "기구"],
        "type": "STEM"
    },
    "화학신소재": {
        "keywords": ["고분자", "신소재", "에너지", "배터리", "유기화학", "나노", "이차전지", "촉매", "세라믹", "플라스틱", "탄소", "물질", "합성", "분석", "실험", "화공", "재료"],
        "type": "STEM"
    },
    "바이오생명": {
        "keywords": ["유전", "세포", "바이러스", "면역", "의약", "생물", "DNA", "RNA", "백신", "미생물", "효소", "단백질", "게놈", "신약", "제약", "질병", "생명공학", "바이오"],
        "type": "STEM"
    },
    "환경에너지": { 
        "keywords": ["환경", "기후", "오염", "정화", "지속가능", "생태", "신재생", "태양광", "수소", "탄소중립", "ESG", "대기", "수질", "폐기물", "에너지효율"],
        "type": "STEM"
    },
    "건축토목": { 
        "keywords": ["건축", "설계", "공간", "도시", "구조", "토목", "주거", "인테리어", "환경디자인", "재생", "스마트시티", "건설", "내진", "디자인", "조경"],
        "type": "STEM"
    },
    "수학통계": {
        "keywords": ["해석학", "대수학", "위상", "통계적", "확률", "수리", "데이터분석", "최적화", "기하", "미적분", "증명", "논리", "금융수학", "암호"],
        "type": "STEM"
    },
    "보건의료": { 
        "keywords": ["간호", "보건", "의료", "임상", "환자", "재활", "건강", "치료", "병원", "해부", "생리", "약리", "의학"],
        "type": "STEM"
    },

    # 인문사회계열
    "경영경제": {
        "keywords": ["마케팅", "재무", "회계", "창업", "소비자", "무역", "유통", "경영", "경제", "금융", "주식", "투자", "시장", "기업", "비즈니스", "통계", "국제통상", "CEO"],
        "type": "HUMAN"
    },
    "사회과학": {
        "keywords": ["정치", "외교", "복지", "행정", "사회문제", "법", "인권", "심리", "상담", "여론", "정책", "국제", "글로벌", "다문화", "지역", "지리", "사회학"],
        "type": "HUMAN"
    },
    "인문어문": {
        "keywords": ["문헌", "도서관", "기록", "철학", "역사", "고전", "문화재", "언어", "문학", "작문", "번역", "통역", "영어", "중국어", "일본어", "문화", "인문학", "글쓰기"],
        "type": "HUMAN"
    },
    "미디어콘텐츠": {
        "keywords": ["영상", "방송", "저널리즘", "광고", "PD", "기자", "콘텐츠", "미디어", "커뮤니케이션", "유튜브", "SNS", "홍보", "디자인", "연출", "편집", "스토리텔링"],
        "type": "HUMAN"
    },
    "교육사범": {
        "keywords": ["교수법", "교육과정", "멘토링", "학습", "교사", "수업", "청소년", "지도", "교육심리", "평가", "학교", "교육학"],
        "type": "HUMAN"
    },
    "의류디자인": { 
        "keywords": ["의류", "패션", "디자인", "소재", "트렌드", "스타일", "MD", "VMD", "색채", "미술", "창작"],
        "type": "HUMAN" 
    }
}


# 텍스트 정제
def clean_text_segment(text):
    text = HEADER_PATTERNS.sub(" ", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


# 1. 텍스트 추출
def extract_text_hybrid(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = []
    print(f">> 총 {len(doc)}페이지 분석 시작...")
    for i, page in enumerate(doc):
        text = page.get_text()
        if len(re.findall(r"[가-힣]", text)) > 10:
            full_text.append(text)
        else:
            print(f">> {i+1}페이지 OCR 수행 중...", end="", flush=True)
            try:
                pix = page.get_pixmap(dpi=450)
                img_data = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_data))
                ocr_result = pytesseract.image_to_string(image, lang="kor+eng")
                full_text.append(ocr_result)
                print(" 완료!")
            except:
                print(" 실패ㅠㅠ")
                full_text.append("")
    return "\n".join(full_text)


# 2. 텍스트 분리 (세특/창체)
def split_sentences(text):
    cleaned = clean_text_segment(text)
    raw = re.split(r"(?<=[다함됨임])[\.\s]+", cleaned)
    out = []
    for s in raw:
        s = s.strip()
        if len(s) < 20: continue
        if any(p in s for p in STOP_PATTERNS): continue
        out.append(s)
    return out

def build_datasets(raw_text):
    subject_sentences = []
    matches = list(SUBJECT_PATTERN.finditer(raw_text))
    subject_ranges = []

    for i, m in enumerate(matches):
        subject = m.group(1).replace(" ", "")
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        content = raw_text[start:end]
        subject_ranges.append((m.start(), end))
        
        norm_subj = next((key for key in STEM_SUBJECTS + HUMANITIES_SUBJECTS if key in subject), "기타")
        sents = split_sentences(content)
        for s in sents:
            subject_sentences.append({"text": s, "source": norm_subj, "type": "SETEUK"})

    changche_text = raw_text
    for start, end in sorted(subject_ranges, reverse=True):
        changche_text = changche_text[:start] + " " + changche_text[end:]
    
    changche_sentences = []
    sents = split_sentences(changche_text)
    for s in sents:
        changche_sentences.append({"text": s, "source": "ChangChe", "type": "CHANGCHE"})

    return subject_sentences, changche_sentences

def infer_major_field_info(major_name, major_keywords):
    text = (str(major_name) or "") + " " + " ".join(major_keywords or [])
    flat = text.replace(" ", "")
    for field, info in MAJOR_FIELD_MAP.items():
        for kw in info["keywords"]:
            if kw in flat:
                return {"field": field, "type": info["type"]}
    if any(x in flat for x in ["공학", "과학", "시스템", "기술", "수학", "물리", "AI"]):
        return {"field": "미분류(이공)", "type": "STEM"}
    return {"field": "미분류(인문)", "type": "HUMAN"}


# 3. 세특/창체 50:50 반영 추천 알고리즘 
def recommend_50_50(subject_data, changche_data, majors, model):
    if not subject_data and not changche_data: return {}

    # 1. 임베딩
    subj_texts = [d["text"] for d in subject_data]
    subj_embs = model.encode(subj_texts, normalize_embeddings=True) if subj_texts else np.array([])
    
    chang_texts = [d["text"] for d in changche_data]
    chang_embs = model.encode(chang_texts, normalize_embeddings=True) if chang_texts else np.array([])

    major_texts = []
    major_types = []
    for m in majors:
        m_kws = m.get("keywords", [])
        field_info = infer_major_field_info(m.get("major"), m_kws)
        major_types.append(field_info["type"])
        desc = str(m.get("major_desc", ""))
        kws = " ".join(m_kws)
        major_texts.append((desc + " " + kws).strip() or str(m.get("major", "")))
    
    major_embs = model.encode(major_texts, normalize_embeddings=True)

    # 2. 유사도 계산
    sim_subj = np.dot(subj_embs, major_embs.T) if len(subj_embs) > 0 else None
    sim_chang = np.dot(chang_embs, major_embs.T) if len(chang_embs) > 0 else None

    grouped = defaultdict(list)

    for j, m in enumerate(majors):
        target_type = major_types[j]

        # 세특 점수 (50%)
        score_se_teuk = 0
        evidence_se_teuk = ""
        if sim_subj is not None:
            weighted_scores = []
            for i, sent_data in enumerate(subject_data):
                raw_sim = sim_subj[i, j]
                if raw_sim < 0.15: continue
                
                source = sent_data["source"]
                weight = 1.0
                
                # 과목 가중치 로직
                if target_type == "STEM":
                    if source in STEM_SUBJECTS: weight = 1.3
                    elif source in HUMANITIES_SUBJECTS: weight = 0.7
                elif target_type == "HUMAN":
                    if source in HUMANITIES_SUBJECTS: weight = 1.2
                    elif source in STEM_SUBJECTS: weight = 0.8
                
                weighted_scores.append((raw_sim * weight, subject_data[i]["text"]))
            
            weighted_scores.sort(key=lambda x: x[0], reverse=True)
            top_k = weighted_scores[:3]
            if top_k:
                avg_val = np.mean([x[0] for x in top_k])
                score_se_teuk = avg_val * 100 * 1.5 
                evidence_se_teuk = top_k[0][1]
        
        # 창체 점수 (50%)
        score_chang = 0
        evidence_chang = ""
        if sim_chang is not None:
            raw_scores = []
            for i, sent_data in enumerate(changche_data):
                raw_sim = sim_chang[i, j]
                if raw_sim < 0.15: continue
                
                # 창체 문장 내에 전공 키워드가 직접 포함되어 있으면 가산점 부여
                bonus = 1.0
                m_kws = m.get("keywords", [])
                for kw in m_kws:
                    if kw in changche_data[i]["text"]:
                        bonus = 1.2 
                        break
                
                raw_scores.append((raw_sim * bonus, changche_data[i]["text"]))
            
            raw_scores.sort(key=lambda x: x[0], reverse=True)
            top_k = raw_scores[:3]
            if top_k:
                avg_val = np.mean([x[0] for x in top_k])
                # 창체 점수 스케일링
                # -> 창체 문장은 세특보다 수가 적고 정제되지 않은 경우가 많아 점수가 낮게 나오는 경향 보정
                score_chang = avg_val * 100 * 1.8 
                evidence_chang = top_k[0][1]

        # 최종 합산
        if score_se_teuk > 0 and score_chang > 0:
            final_score = (score_se_teuk * 0.5) + (score_chang * 0.5)
        elif score_se_teuk > 0:
            final_score = score_se_teuk * 0.8 # 창체가 없으므로 약간의 감점
        elif score_chang > 0:
            final_score = score_chang * 0.8 # 세특이 없으므로 약간의 감점
        else:
            final_score = 0

        # 근거 텍스트 포맷팅
        evidence_text = ""
        if evidence_se_teuk: 
            # 너무 길면 자르기
            if len(evidence_se_teuk) > 70: evidence_se_teuk = evidence_se_teuk[:70] + "..."
            evidence_text += f"[세특] {evidence_se_teuk}"
            
        if evidence_chang: 
            if len(evidence_chang) > 70: evidence_chang = evidence_chang[:70] + "..."
            if evidence_text: evidence_text += "\n   └ "
            evidence_text += f"[창체] {evidence_chang}"

        field_info = infer_major_field_info(m.get("major"), m.get("keywords"))
        
        grouped[m.get("university", "대학 미정")].append({
            "major": m.get("major"),
            "final": round(final_score, 2),
            "score_s": round(score_se_teuk, 2),
            "score_c": round(score_chang, 2),
            "field": field_info["field"],
            "evidence": evidence_text
        })

    for uni in grouped:
        grouped[uni] = sorted(grouped[uni], key=lambda x: x["final"], reverse=True)[:5]
        
    return grouped


# 4. 메인
if __name__ == "__main__":
    setup_tesseract()

    root = tk.Tk()
    root.withdraw()
    print("\n>> PDF 파일 선택...")
    pdf_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
    if not pdf_path: sys.exit()

    print("\n>> 모델 로딩 중...")
    model = SentenceTransformer(MODEL_NAME)
    
    if os.path.exists(CSV_PATH):
        majors_df = pd.read_csv(CSV_PATH)
        if 'keywords' in majors_df.columns:
             majors_df['keywords'] = majors_df['keywords'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else [])
        majors = majors_df.to_dict("records")
    else:
        majors = []

    if majors:
        start = time.time()
        print(f">> 분석 시작: {os.path.basename(pdf_path)}")
        
        raw = extract_text_hybrid(pdf_path)
        subj_data, chang_data = build_datasets(raw)
        
        print(f" >> 데이터 추출: 세특 문장 {len(subj_data)}개 / 창체 문장 {len(chang_data)}개")
        
        if not subj_data and not chang_data:
            print("유효한 문장이 없습니다.")
        else:
            results = recommend_50_50(subj_data, chang_data, majors, model)
            # 소요 시간은 로그용으로 넣은거임
            print(f"\n>> 소요 시간: {time.time() - start:.2f}초\n")
            print("=" * 70)
            for uni, items in results.items():
                print(f">> [{uni}] TOP 5")
                for i, r in enumerate(items, 1):
                    ev = r['evidence']
                    if len(ev) > 80: ev = ev[:80] + "..."
                    print(f"[{i}] {r['major']} (총점: {r['final']:.2f} / 세특: {r['score_s']:.2f} / 창체: {r['score_c']:.2f})")
                    print(f"   └ {ev}")
                print("-" * 70)