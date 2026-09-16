import json
import numpy as np
import cv2
from PIL import Image
import streamlit as st

from src.recommend import recommend_foundation
from src.calibration import apply_correction_image
from src import github_storage

st.set_page_config(page_title="파운데이션 색상 추천", page_icon="💄")

LOCAL_DB_PATH = "data/foundation_db.json"

# ---------------------------------------------------------------------------
# 화면 디자인 + 화면 흐름. 실제 인식/보정/추천 로직에는 손대지 않고, 화면
# 구성만 "폰 앱처럼 한 번에 한 화면씩" 넘어가는 구조로 만들어요.
#
# st.session_state.step 값 하나로 지금 어느 화면인지 관리해요:
#   "intro" -> "card" -> "scope" -> "capture" -> "result"
# 매 화면은 아래 if/elif 블록 중 하나만 그려지고, 버튼을 누르면 step 값을
# 바꾸고 st.rerun()으로 다시 그려요 -- 페이지 자체를 이동하는 게 아니라
# 한 페이지 안에서 내용만 통째로 바뀌는 방식이라, 실제 브라우저 새로고침
# 없이도 화면이 매끄럽게 넘어가요.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Streamlit 기본 상단 헤더/여백을 줄여서, 좁은 화면(폰)에서 스크롤을
    최대한 줄여요. */
    header[data-testid="stHeader"] { height: 0; visibility: hidden; }
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 480px; }
    #MainMenu, footer { visibility: hidden; }

    /* 화면이 바뀔 때마다 살짝 페이드인 -- 완전한 화면 전환 애니메이션은
    아니지만, 버튼 누르면 뚝 끊기지 않고 자연스럽게 나타나는 느낌을 줘요. */
    .block-container { animation: fadeIn 0.25s ease-in; }
    @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }

    /* 전체적으로 조금 더 둥글고 부드러운 느낌 */
    .stButton > button, .stDownloadButton > button {
        border-radius: 999px;
        border: none;
        padding: 0.7em 1.4em;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(255, 143, 171, 0.35);
        transition: transform 0.15s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        transform: translateY(-1px);
    }
    /* 뒤로가기용 텍스트 버튼은 작고 수수하게 */
    .back-btn button {
        box-shadow: none;
        background: transparent !important;
        color: #A97D89 !important;
        padding: 0.2em 0.4em;
        font-weight: 500;
    }

    /* 상단 타이틀 영역을 부드러운 핑크 그라데이션 배너로 */
    .hero-banner {
        background: linear-gradient(135deg, #FFE3ED 0%, #FFF3F7 100%);
        border-radius: 20px;
        padding: 1.6em 1.8em;
        margin-bottom: 1.2em;
        border: 1px solid #FFD6E4;
        text-align: center;
    }
    .hero-banner .hero-emoji { font-size: 2.8em; margin-bottom: 0.15em; }
    .hero-banner h1 { margin: 0 0 0.35em 0; color: #4A3238; font-size: 1.5em; }
    .hero-banner p { margin: 0; color: #7A5C64; font-size: 0.95em; line-height: 1.5; }

    /* 소개 화면의 3가지 특징 줄 */
    .feature-row { display: flex; align-items: center; gap: 0.6em; padding: 0.5em 0.2em; color: #4A3238; }
    .feature-row .fe-icon { font-size: 1.3em; }

    /* 진행 단계 표시 (1/3, 2/3, 3/3) */
    .step-indicator { color: #C98CA0; font-size: 0.85em; font-weight: 600; margin-bottom: 0.3em; letter-spacing: 0.02em; }

    /* 선택 화면의 박스형 카드 */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 16px !important;
        border: 1.5px solid #FFD6E4 !important;
    }

    /* 카메라를 켜기 전 안내 카드 */
    .camera-placeholder {
        background: #FFF3F7;
        border: 1.5px dashed #FFB6C9;
        border-radius: 18px;
        padding: 2em 1.5em;
        text-align: center;
        color: #7A5C64;
    }
    .camera-placeholder .big-emoji { font-size: 2.4em; margin-bottom: 0.2em; }

    /* 파일 업로드 탭 안내 카드 */
    .upload-hint {
        background: #FFF3F7;
        border-radius: 14px;
        padding: 0.9em 1.2em;
        color: #7A5C64;
        margin-bottom: 0.8em;
        font-size: 0.95em;
    }

    /* 탭 밑줄/선택색을 핑크 계열로 */
    .stTabs [aria-selected="true"] { color: #FF6B94 !important; }
    .stTabs [data-baseweb="tab-highlight"] { background-color: #FF8FAB !important; }

    /* 결과 카드(추천 색상 위 배경) 둥글게 */
    div[data-testid="stExpander"] {
        border-radius: 14px;
        border: 1px solid #FFD6E4 !important;
    }

    /* 전면 카메라 미리보기 좌우 반전 -- 셀카 찍듯 자연스럽게 보이도록 하는
    순수 화면상 CSS 효과일 뿐, st.camera_input이 실제로 넘겨주는 사진
    바이트 자체는 그대로라 카드/얼굴 인식은 원래 촬영본 그대로 처리돼요. */
    [data-testid="stCameraInput"] video,
    [data-testid="stCameraInputWebcamStyledBox"] video {
        transform: scaleX(-1);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=60)
def load_shades():
    """Prefer the live GitHub-backed DB; fall back to the bundled local
    file if secrets aren't configured yet or GitHub is briefly unreachable."""
    try:
        shades, _ = github_storage.read_shades()
        if shades:
            return shades, "github"
    except Exception:
        pass
    with open(LOCAL_DB_PATH, encoding="utf-8") as f:
        return json.load(f)["shades"], "local"


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def lab_to_hex(L, a, b):
    from skimage.color import lab2rgb

    lab = np.array([[[L, a, b]]], dtype=np.float64)
    rgb = np.clip(lab2rgb(lab)[0, 0], 0, 1) * 255
    r, g, bch = [int(round(c)) for c in rgb]
    return f"#{r:02x}{g:02x}{bch:02x}"


def draw_region_overlay(rgb_img: np.ndarray, regions_debug: list) -> np.ndarray:
    """Draws a circle at each sampled skin ROI directly on the photo, so
    it's visible (not just described in text) which part of the face was
    used -- green + thicker for the region actually adopted, gray/thin
    (dashed-look via a smaller filled dot) for the rest."""
    overlay = rgb_img.copy()
    for rd in regions_debug:
        center = tuple(int(v) for v in rd["center"])
        radius = int(rd["radius"])
        is_final = rd.get("used_as_final")
        color = (0, 230, 0) if is_final else (255, 80, 80)
        thickness = 4 if is_final else 2
        cv2.circle(overlay, center, radius, color, thickness)

        label = rd["name"] + (" (채택)" if is_final else "")
        text_pos = (center[0] - radius, max(0, center[1] - radius - 10))
        cv2.putText(
            overlay, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (0, 0, 0), 3, cv2.LINE_AA,
        )
        cv2.putText(
            overlay, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            color, 1, cv2.LINE_AA,
        )
    return overlay


def go(step: str):
    st.session_state.step = step
    st.rerun()


def back_button(target_step: str, label: str = "← 이전"):
    st.markdown('<div class="back-btn">', unsafe_allow_html=True)
    if st.button(label, key=f"back-{target_step}"):
        go(target_step)
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 화면 상태 초기화
# ---------------------------------------------------------------------------
if "step" not in st.session_state:
    st.session_state.step = "intro"
if "has_card" not in st.session_state:
    st.session_state.has_card = None
if "use_verified_only" not in st.session_state:
    st.session_state.use_verified_only = None
if "camera_active" not in st.session_state:
    st.session_state.camera_active = False
if "result" not in st.session_state:
    st.session_state.result = None
if "result_image_bgr" not in st.session_state:
    st.session_state.result_image_bgr = None

shades, source = load_shades()
if not shades:
    st.error("파운데이션 데이터베이스가 비어 있어요. 관리자 페이지에서 색상을 추가해주세요.")
    st.stop()


# =============================================================================
# 화면 1: 소개 (인트로)
# =============================================================================
if st.session_state.step == "intro":
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-emoji">💄</div>
            <h1>나에게 맞는 파운데이션 찾기</h1>
            <p>색상 카드와 얼굴을 같이 촬영한 사진 한 장이면,<br>
            카메라·조명 왜곡을 자동으로 보정해서<br>
            가장 잘 맞는 파운데이션 색을 찾아드려요.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="feature-row"><span class="fe-icon">🪄</span> 색상 카드 자동 인식 + 자동 보정</div>
        <div class="feature-row"><span class="fe-icon">📷</span> 카드가 없어도 사용 가능</div>
        <div class="feature-row"><span class="fe-icon">🎯</span> 내 피부톤과 가장 가까운 색 Top 3 추천</div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    if st.button("시작하기 →", use_container_width=True, type="primary"):
        go("card")

# =============================================================================
# 화면 2: 컬러체커 카드 유무 선택
# =============================================================================
elif st.session_state.step == "card":
    back_button("intro")
    st.markdown('<div class="step-indicator">1 / 3</div>', unsafe_allow_html=True)
    st.markdown("### 컬러체커 카드가 있으신가요?")
    st.write("")

    with st.container(border=True):
        st.markdown("#### 📇 카드 있어요")
        st.caption("색상 카드로 카메라·조명을 자동 보정해서 더 정확한 결과를 받아요.")
        if st.button("이걸로 선택", key="pick-card-yes", use_container_width=True, type="primary"):
            st.session_state.has_card = True
            go("scope")

    st.write("")

    with st.container(border=True):
        st.markdown("#### 🙅 카드 없어요")
        st.caption("카드 없이 얼굴 사진만으로도 진행할 수 있어요 (정확도는 조금 낮아질 수 있어요).")
        if st.button("이걸로 선택", key="pick-card-no", use_container_width=True):
            st.session_state.has_card = False
            go("scope")

# =============================================================================
# 화면 3: 추천 데이터 범위 선택
# =============================================================================
elif st.session_state.step == "scope":
    back_button("card")
    st.markdown('<div class="step-indicator">2 / 3</div>', unsafe_allow_html=True)
    st.markdown("### 어떤 데이터에서 추천받을까요?")
    st.write("")

    n_verified = sum(1 for s in shades if s.get("verified", True))

    with st.container(border=True):
        st.markdown(f"#### ✅ 정확측정 데이터만 ({n_verified}개)")
        st.caption("실제 카드+스와치 사진으로 정밀 측정해 넣은 색상 안에서만 추천해요.")
        if st.button("이걸로 선택", key="pick-scope-verified", use_container_width=True, type="primary"):
            st.session_state.use_verified_only = True
            go("capture")

    st.write("")

    with st.container(border=True):
        st.markdown(f"#### 📦 전체 데이터 ({len(shades)}개)")
        st.caption("정밀 측정 여부와 상관없이, 등록된 모든 색상 중에서 추천해요.")
        if st.button("이걸로 선택", key="pick-scope-all", use_container_width=True):
            st.session_state.use_verified_only = False
            go("capture")

# =============================================================================
# 화면 4: 촬영 / 업로드
# =============================================================================
elif st.session_state.step == "capture":
    back_button("scope")
    st.markdown('<div class="step-indicator">3 / 3</div>', unsafe_allow_html=True)

    has_card = st.session_state.has_card
    use_verified_only = st.session_state.use_verified_only
    shade_pool = [s for s in shades if s.get("verified", True)] if use_verified_only else shades

    if not shade_pool:
        st.error("선택하신 범위(정확측정 데이터)에는 아직 등록된 색상이 없어요.")
        if st.button("← 범위 다시 선택하기"):
            go("scope")
        st.stop()

    st.markdown("### 사진을 준비해주세요")
    if not has_card:
        st.caption("⚠️ 카드 없이 측정하면 정확도가 낮을 수 있어요.")

    camera_instruction = (
        "색상 카드 + 얼굴이 같이 나오게 촬영해주세요" if has_card else "얼굴이 잘 보이게 촬영해주세요 (카드는 없어도 돼요)"
    )
    upload_instruction = (
        "카드 + 얼굴이 같이 나온 사진을 선택해주세요" if has_card else "얼굴이 잘 보이는 사진을 선택해주세요 (카드는 없어도 돼요)"
    )

    tab1, tab2 = st.tabs(["📷 카메라로 촬영", "🖼️ 사진첩에서 선택"])
    image_bgr = None

    with tab1:
        if not st.session_state.camera_active:
            st.markdown(
                f"""
                <div class="camera-placeholder">
                    <div class="big-emoji">🤳</div>
                    아래 버튼을 누르면 카메라가 켜져요.<br>
                    {camera_instruction} 준비한 뒤 눌러주세요.
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")
            if st.button("📷 카메라 켜고 촬영하기", use_container_width=True, type="primary"):
                st.session_state.camera_active = True
                st.rerun()
        else:
            cam_img = st.camera_input(camera_instruction)
            if st.button("✕ 카메라 끄기", use_container_width=True):
                st.session_state.camera_active = False
                st.rerun()
            if cam_img is not None:
                image_bgr = pil_to_bgr(Image.open(cam_img))

    with tab2:
        st.markdown(
            """
            <div class="upload-hint">
                📁 아래 버튼을 누르면 휴대폰의 사진첩(갤러리) 또는 파일 앱에서
                바로 사진을 골라올 수 있어요.
            </div>
            """,
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(upload_instruction, type=["jpg", "jpeg", "png"])
        if uploaded is not None:
            image_bgr = pil_to_bgr(Image.open(uploaded))

    if image_bgr is not None:
        with st.spinner("분석 중..."):
            h, w = image_bgr.shape[:2]
            scale = 1600 / max(h, w)
            if scale < 1:
                image_bgr = cv2.resize(image_bgr, None, fx=scale, fy=scale)

            result = recommend_foundation(image_bgr, shade_pool, top_n=3, has_card=has_card)

        if not result.success:
            st.error(result.message)
            if has_card:
                st.info("💡 색상 카드와 얼굴이 모두 선명하게 나오도록, 너무 어둡지 않은 곳에서 다시 촬영해보세요.")
            else:
                st.info("💡 얼굴이 선명하게 잘 보이도록, 너무 어둡지 않은 곳에서 다시 촬영해보세요.")
        else:
            # 결과는 세션에 저장해두고, 완전히 분리된 "결과" 화면으로 넘어가요.
            st.session_state.result = result
            st.session_state.result_image_bgr = image_bgr
            st.session_state.camera_active = False
            go("result")

# =============================================================================
# 화면 5: 결과
# =============================================================================
elif st.session_state.step == "result":
    result = st.session_state.result
    image_bgr = st.session_state.result_image_bgr
    has_card = st.session_state.has_card
    use_verified_only = st.session_state.use_verified_only

    if result is None or image_bgr is None:
        st.info("아직 분석된 결과가 없어요.")
        if st.button("촬영하러 가기"):
            go("capture")
        st.stop()

    st.markdown('<div class="step-indicator">결과</div>', unsafe_allow_html=True)
    st.success("분석 완료!")
    best = result.matches[0]
    st.subheader(f"✨ 추천: {best['brand']} · {best['name']}")
    st.markdown(
        f"<div style='width:100%;height:60px;border-radius:8px;"
        f"background-color:{lab_to_hex(best['L'], best['a'], best['b'])};'></div>",
        unsafe_allow_html=True,
    )

    if not result.debug.get("brighter_pool_used", True):
        st.caption("ℹ️ 측정된 피부색보다 밝은 색상이 DB에 없어서, 이번엔 전체 DB 중 가장 가까운 색으로 추천했어요.")

    st.write("가까운 순서 Top 3")
    cols = st.columns(3)
    for col, m in zip(cols, result.matches):
        with col:
            st.markdown(
                f"<div style='width:100%;height:40px;border-radius:6px;"
                f"background-color:{lab_to_hex(m['L'], m['a'], m['b'])};'></div>",
                unsafe_allow_html=True,
            )
            st.caption(f"{m['brand']} {m['name']}\n\nΔE {m['delta_e']:.1f}")

    with st.expander("🔍 색 보정 전/후 비교" if has_card else "🔍 측정 상세 보기", expanded=True):
        rgb_before = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        if has_card:
            correction_matrix = np.array(result.debug["correction_matrix"])
            rgb_after = apply_correction_image(rgb_before, correction_matrix)

            st.caption("사진 전체에 카드 보정을 그대로 적용해보면 이렇게 달라져요 (카메라/조명 왜곡 제거):")
            c1, c2 = st.columns(2)
            c1.image(rgb_before, caption="보정 전 (원본)", use_container_width=True)
            c2.image(rgb_after, caption="보정 후", use_container_width=True)
        else:
            st.caption("ℹ️ 카드가 없어서 카메라/조명 보정 없이, 촬영된 그대로의 색을 사용했어요.")

        st.caption("내 얼굴에서 실제로 어느 부분을 측정했는지 (초록 = 최종 채택된 부위):")
        overlay_img = draw_region_overlay(rgb_before, result.debug["regions"])
        st.image(overlay_img, caption="샘플링 영역 위치", use_container_width=True)

        st.caption("피부색 샘플링에 사용된 영역별 색" + (" (보정 전 → 보정 후):" if has_card else ":"))
        for rd in result.debug["regions"]:
            rc1, rc2, rc3 = st.columns([1, 1, 2])
            raw_hex = "#%02x%02x%02x" % tuple(int(max(0, min(255, v))) for v in rd["raw_rgb"])
            corr_hex = "#%02x%02x%02x" % tuple(int(max(0, min(255, v))) for v in rd["corrected_rgb"])
            rc1.markdown(f"<div style='width:100%;height:32px;border-radius:4px;background-color:{raw_hex};'></div>", unsafe_allow_html=True)
            if has_card:
                rc2.markdown(f"<div style='width:100%;height:32px;border-radius:4px;background-color:{corr_hex};'></div>", unsafe_allow_html=True)
            tag = " ⭐최종채택" if rd.get("used_as_final") else (" (제외됨)" if rd.get("excluded_as_outlier") else "")
            rc3.caption(f"{rd['name']}{tag} — Lab: L={rd['lab'][0]:.1f}, a={rd['lab'][1]:.1f}, b={rd['lab'][2]:.1f}")

    with st.expander("자세히 보기 (분석 정보)"):
        mean_err = result.debug.get("calibration_mean_error")
        st.json(
            {
                "보정 후 피부색 Lab": [round(float(x), 2) for x in result.corrected_lab],
                "최종 채택 영역": result.debug["final_region"],
                "카드 사용 여부": has_card,
                "카드 보정 평균 오차": round(mean_err, 2) if mean_err is not None else "카드 미사용",
                "DB 출처": source,
                "정확측정 데이터만 사용": use_verified_only,
                "밝은 색상만 비교했는지": result.debug["brighter_pool_used"],
                "비교 대상 색상 개수": result.debug["n_candidates_considered"],
                "영역별 샘플": result.debug["regions"],
            }
        )

    st.write("")
    if st.button("🔄 처음부터 다시하기", use_container_width=True):
        st.session_state.step = "intro"
        st.session_state.has_card = None
        st.session_state.use_verified_only = None
        st.session_state.result = None
        st.session_state.result_image_bgr = None
        st.session_state.camera_active = False
        st.rerun()
