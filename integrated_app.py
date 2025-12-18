import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os
import re
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import ast
import platform
from sklearn.linear_model import LinearRegression, HuberRegressor
from sentence_transformers import SentenceTransformer
from collections import defaultdict

# ---------------------------------------------------------
# 0. PAGE CONFIGURATION (Must be first)
# ---------------------------------------------------------
st.set_page_config(page_title="SMU 입시 분석 솔루션", layout="wide", page_icon="🎓")

# ---------------------------------------------------------
# 1. SHARED & COMPREHENSIVE (JONGHAP) LOGIC
# ---------------------------------------------------------

# 상수 및 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "jhgan/ko-sbert-multitask"

# 노이즈 제거 패턴
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

STEM_SUBJECTS = ["수학", "미적분", "기하", "물리학", "화학", "생명과학", "지구과학", "정보", "소프트웨어", "프로그래밍", "과학탐구", "인공지능"]
HUMANITIES_SUBJECTS = ["국어", "문학", "언어와매체", "화법과작문", "영어", "한국사", "통합사회", "생활과윤리", "윤리와사상", "정치와법", "경제", "사회문화", "지리", "세계사"]

MAJOR_FIELD_MAP = {
    "컴퓨터공학": {"keywords": ["소프트웨어", "코딩", "알고리즘", "보안", "네트워크", "서버", "개발", "앱", "웹", "시스템"], "type": "STEM"},
    "인공지능": {"keywords": ["AI", "머신러닝", "딥러닝", "데이터사이언스", "빅데이터", "신경망", "로봇지능"], "type": "STEM"},
    "전자반도체": {"keywords": ["회로", "반도체", "임베디드", "신호처리", "통신", "전기", "디스플레이", "IOT"], "type": "STEM"},
    "기계로봇": {"keywords": ["역학", "설계", "자동차", "항공", "제어", "로봇", "메카트로닉스"], "type": "STEM"},
    "화학신소재": {"keywords": ["고분자", "신소재", "에너지", "배터리", "유기화학", "나노"], "type": "STEM"},
    "바이오": {"keywords": ["유전", "세포", "바이러스", "면역", "의약", "생물"], "type": "STEM"},
    "수학통계": {"keywords": ["해석학", "대수학", "위상", "통계적", "확률", "수리"], "type": "STEM"},
    "인문": {"keywords": ["문헌", "도서관", "기록", "철학", "역사", "고전", "문화재", "언어", "심리"], "type": "HUMAN"},
    "경영경제": {"keywords": ["마케팅", "재무", "회계", "창업", "소비자", "무역", "유통"], "type": "HUMAN"},
    "사회과학": {"keywords": ["정치", "외교", "복지", "행정", "사회문제", "법", "인권"], "type": "HUMAN"}, 
    "미디어": {"keywords": ["영상", "방송", "저널리즘", "광고", "PD", "기자"], "type": "HUMAN"},
    "교육": {"keywords": ["교수법", "교육과정", "멘토링", "학습", "교사"], "type": "HUMAN"}
}

@st.cache_resource
def load_sbert_model():
    return SentenceTransformer(MODEL_NAME)

def setup_tesseract():
    # Streamlit Cloud나 로컬 환경에 맞게 경로 설정 필요
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

def clean_text_segment(text):
    text = HEADER_PATTERNS.sub(" ", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_text_hybrid(pdf_file):
    """
    Streamlit uploaded file object를 받아서 처리
    """
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    full_text = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_pages = len(doc)
    
    for i, page in enumerate(doc):
        status_text.text(f"Processing page {i+1}/{total_pages}...")
        progress_bar.progress((i + 1) / total_pages)
        
        text = page.get_text()
        if len(re.findall(r"[가-힣]", text)) > 10:
            full_text.append(text)
        else:
            try:
                pix = page.get_pixmap(dpi=300) # 속도를 위해 dpi 약간 조정
                img_data = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_data))
                ocr_result = pytesseract.image_to_string(image, lang="kor+eng")
                full_text.append(ocr_result)
            except Exception as e:
                full_text.append("")
    
    progress_bar.empty()
    status_text.empty()
    return "\n".join(full_text)

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

def recommend_50_50(subject_data, changche_data, majors, model):
    if not subject_data and not changche_data: return {}

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
                score_chang = avg_val * 100 * 1.8 
                evidence_chang = top_k[0][1]

        if score_se_teuk > 0 and score_chang > 0:
            final_score = (score_se_teuk * 0.5) + (score_chang * 0.5)
        elif score_se_teuk > 0:
            final_score = score_se_teuk * 0.8
        elif score_chang > 0:
            final_score = score_chang * 0.8
        else:
            final_score = 0

        evidence_text = ""
        if evidence_se_teuk: 
            evidence_text += f"**[세특]** {evidence_se_teuk}"
        if evidence_chang: 
            if evidence_text: evidence_text += "\n\n"
            evidence_text += f"**[창체]** {evidence_chang}"

        field_info = infer_major_field_info(m.get("major"), m.get("keywords"))
        
        grouped[m.get("university", "대학 미정")].append({
            "전공": m.get("major"),
            "종합점수": round(final_score, 2),
            "세특점수": round(score_se_teuk, 2),
            "창체점수": round(score_chang, 2),
            "계열": field_info["field"],
            "근거문장": evidence_text
        })

    for uni in grouped:
        grouped[uni] = sorted(grouped[uni], key=lambda x: x["종합점수"], reverse=True)[:5]
        
    return grouped

# ---------------------------------------------------------
# 2. SUBJECT (KYOGWA) LOGIC
# ---------------------------------------------------------

@st.cache_data
def load_admission_data():
    """
    Loads and processes both admission data files
    """
    # File Check
    main_file = "smu_admission_results.xlsx - 상명대입결.csv"
    sub_file = "smu_admission_results.xlsx - 24학년도 수시입결.csv"
    
    if not os.path.exists(main_file) or not os.path.exists(sub_file):
        return pd.DataFrame()

    # --- Part A: Main History (2020-2023, 2025) ---
    try:
        df_main_raw = pd.read_csv(main_file, header=None, encoding='utf-8')
    except:
        return pd.DataFrame()

    years_map = {2025: (2, 8), 2023: (8, 14), 2022: (14, 20), 2021: (20, 26), 2020: (26, 32)}
    clean_rows = []

    if not df_main_raw.empty:
        start_row = 3 if len(df_main_raw) > 3 else 0
        for index, row in df_main_raw.iloc[start_row:].iterrows():
            if len(row) < 32:
                padding = pd.Series([np.nan] * (32 - len(row)))
                row = pd.concat([row, padding], ignore_index=True)
            dept_group = row[0]
            major = row[1]
            if pd.isna(major): continue
            for year, (start_col, end_col) in years_map.items():
                try:
                    year_data = row.iloc[start_col:end_col].values
                    if len(year_data) < 6 or all(pd.isna(x) for x in year_data): continue
                    clean_rows.append({"Year": int(year), "Department": dept_group, "Major": major, "Category": "Initial Accepted (최초합격자)", "Max": year_data[0], "Avg": year_data[1], "Min": year_data[2]})
                    clean_rows.append({"Year": int(year), "Department": dept_group, "Major": major, "Category": "Final Registered (최종등록자)", "Max": year_data[3], "Avg": year_data[4], "Min": year_data[5]})
                except: continue
    df_history = pd.DataFrame(clean_rows)

    # --- Part B: 2024 File ---
    try:
        df_2024_raw = pd.read_csv(sub_file, encoding='utf-8')
    except:
        return df_history

    rows_2024 = []
    if not df_2024_raw.empty and len(df_2024_raw.columns) >= 3:
        current_dept = None
        for index, row in df_2024_raw.iterrows():
            raw_dept = row.iloc[0]
            raw_major = row.iloc[1]
            raw_grade = row.iloc[2]
            if pd.notna(raw_dept) and str(raw_dept).strip() != "모집단위": current_dept = raw_dept
            if pd.isna(raw_major) or str(raw_major).strip() in ["모집단위", "최종등록자 70% cut 성적"]: continue
            try: grade_val = float(str(raw_grade).replace(',', ''))
            except: grade_val = None
            if grade_val is not None:
                rows_2024.append({"Year": 2024, "Department": current_dept, "Major": raw_major, "Category": "Final Registered (최종등록자)", "Max": np.nan, "Avg": grade_val, "Min": np.nan})
    
    df_2024 = pd.DataFrame(rows_2024)
    final_df = pd.concat([df_history, df_2024], ignore_index=True)
    for col in ['Max', 'Avg', 'Min']: final_df[col] = pd.to_numeric(final_df[col], errors='coerce')
    return final_df

def run_ensemble_prediction(df_major):
    data = df_major[df_major['Category'] == "Final Registered (최종등록자)"].dropna(subset=['Avg']).sort_values('Year')
    if len(data) < 3: return None, None

    X = data['Year'].values.reshape(-1, 1)
    y = data['Avg'].values
    next_year = [[2026]]
    
    weights = (data['Year'] - 2019) ** 2
    model_weighted = LinearRegression()
    model_weighted.fit(X, y, sample_weight=weights)
    pred_weighted = model_weighted.predict(next_year)[0]
    
    model_robust = HuberRegressor(epsilon=1.35) 
    model_robust.fit(X, y)
    pred_robust = model_robust.predict(next_year)[0]
    
    final_pred = (pred_weighted + pred_robust) / 2
    return final_pred, data

# ---------------------------------------------------------
# 3. GUI MODES
# ---------------------------------------------------------

def mode_kyogwa():
    st.header("📊 학생부교과전형 (내신 성적 예측)")
    st.markdown("상명대 입시 데이터를 기반으로 2026학년도 합격선을 예측합니다.")
    
    df = load_admission_data()
    if df.empty:
        st.error("데이터 파일을 찾을 수 없습니다. (smu_admission_results.xlsx - ... .csv)")
        st.info("실행 경로에 CSV 파일이 있는지 확인해주세요.")
        return

    col_input, col_main = st.columns([1, 3])

    with col_input:
        st.subheader("입력 정보")
        if 'Major' in df.columns:
            all_majors = sorted(df['Major'].dropna().unique().astype(str))
            selected_major = st.selectbox("희망 전공 선택", all_majors)
        else:
            st.error("Major column not found")
            st.stop()
        
        user_grade = st.number_input("나의 내신 등급", min_value=1.0, max_value=9.0, value=2.5, step=0.1, format="%.2f")
        st.divider()
        st.caption("AI 앙상블 모델(가중 선형 회귀 + Huber 회귀)을 사용하여 2026학년도 입결을 예측합니다.")

    major_data = df[df['Major'] == selected_major].sort_values(by="Year")
    predicted_grade, regression_data = run_ensemble_prediction(major_data)

    with col_main:
        c1, c2 = st.columns([2, 1])
        
        with c1:
            st.subheader(f"📈 {selected_major} 입결 트렌드")
            fig = go.Figure()
            color_map = {"Initial Accepted (최초합격자)": "#3366CC", "Final Registered (최종등록자)": "#DC3912"}
            
            for category in major_data['Category'].unique():
                subset = major_data[major_data['Category'] == category]
                fig.add_trace(go.Scatter(
                    x=subset['Year'], y=subset['Avg'], mode='lines+markers+text',
                    name=f"{category} 평균",
                    line=dict(color=color_map.get(category, "gray"), width=3),
                    marker=dict(size=8), text=subset['Avg'], textposition="top center"
                ))

            if predicted_grade:
                last_year_val = regression_data.iloc[-1]['Avg']
                last_year_x = regression_data.iloc[-1]['Year']
                fig.add_trace(go.Scatter(
                    x=[last_year_x, 2026], y=[last_year_val, predicted_grade],
                    mode='lines+markers+text', name="2026 AI 예측",
                    line=dict(color="purple", width=3, dash='dot'),
                    marker=dict(size=10, symbol='star'),
                    text=[None, f"{predicted_grade:.2f}"], textposition="top center"
                ))
            
            fig.add_hline(y=user_grade, line_dash="dash", line_color="green", annotation_text="내 성적", annotation_position="bottom right")
            fig.update_layout(xaxis=dict(tickmode='linear', tick0=2020, dtick=1), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("💡 합격 가능성 분석")
            if predicted_grade:
                st.metric("2026 예상 컷", f"{predicted_grade:.2f} 등급")
                diff = user_grade - predicted_grade
                
                if diff < -0.5:
                    st.success("🟢 매우 안정 (Very Safe)")
                    st.write("예상 합격선보다 점수가 여유롭습니다.")
                elif diff <= 0:
                    st.success("🔵 안정 (Safe)")
                    st.write("예상 합격선 이내입니다.")
                elif diff < 0.2:
                    st.warning("🟡 소신/적정 (Competitive)")
                    st.write("예상 합격선보다 약간 낮습니다.")
                else:
                    st.error("🔴 위험 (High Risk)")
                    st.write("예상 합격선과 차이가 큽니다.")
            else:
                st.warning("데이터 부족으로 예측 불가")

def mode_jonghap():
    st.header("📑 학생부종합전형 (생기부 기반 전공 추천)")
    st.markdown("생활기록부(PDF)를 업로드하면 AI가 내용을 분석하여 적합한 전공을 추천합니다.")

    setup_tesseract()
    
    # DB Check
    db_path = "majors_db.csv"
    if not os.path.exists(db_path):
        st.warning(f"'{db_path}' 파일이 없습니다. 전공 데이터베이스가 필요합니다.")
        uploaded_db = st.file_uploader("전공 DB (majors_db.csv) 업로드", type=["csv"])
        if uploaded_db:
            majors_df = pd.read_csv(uploaded_db)
        else:
            return
    else:
        majors_df = pd.read_csv(db_path)

    if 'keywords' in majors_df.columns:
        majors_df['keywords'] = majors_df['keywords'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else [])
    majors = majors_df.to_dict("records")

    uploaded_file = st.file_uploader("생기부 PDF 파일 업로드", type=["pdf"])

    if uploaded_file and st.button("분석 시작"):
        with st.spinner("AI 모델 로딩 중... (처음 실행 시 시간이 소요됩니다)"):
            model = load_sbert_model()
        
        with st.spinner("PDF 텍스트 추출 및 OCR 수행 중..."):
            raw_text = extract_text_hybrid(uploaded_file)
        
        with st.spinner("문장 분석 및 매칭 중..."):
            subj_data, chang_data = build_datasets(raw_text)
            
            st.info(f"추출 결과: 세특 문장 {len(subj_data)}개 / 창체 문장 {len(chang_data)}개")
            
            if not subj_data and not chang_data:
                st.error("유효한 문장을 추출하지 못했습니다. PDF가 이미지 형태라면 OCR이 실패했을 수 있습니다.")
            else:
                results = recommend_50_50(subj_data, chang_data, majors, model)
                
                st.divider()
                st.subheader("🎯 추천 전공 TOP 5")
                
                # 결과 출력
                for uni, items in results.items():
                    with st.expander(f"🏫 {uni} 추천 결과 보기", expanded=True):
                        for i, r in enumerate(items, 1):
                            st.markdown(f"#### {i}위: {r['전공']} (총점: {r['종합점수']})")
                            st.caption(f"세특: {r['세특점수']} / 창체: {r['창체점수']} | 계열: {r['계열']}")
                            st.markdown(r['근거문장'])
                            st.divider()

# ---------------------------------------------------------
# 4. MAIN APP STRUCTURE
# ---------------------------------------------------------

def main():
    st.sidebar.title("상명대 입시 도우미 🎓")
    st.sidebar.markdown("---")
    
    # 앱 모드 선택
    app_mode = st.sidebar.radio(
        "전형 선택",
        ["학생부교과 (내신 예측)", "학생부종합 (생기부 분석)"],
        captions=["성적 데이터를 통한 합격 예측", "PDF 분석을 통한 전공 추천"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("문의: 입학처 / 개발팀")

    if app_mode == "학생부교과 (내신 예측)":
        mode_kyogwa()
    else:
        mode_jonghap()

if __name__ == "__main__":
    main()