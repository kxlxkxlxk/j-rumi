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
# 화면 디자인 (핑크 톤). 실제 인식/보정/추천 로직에는 전혀 손대지 않고,
# 화면에 보이는 부분(색상, 카메라를 켜는 시점, 버튼 모양)만 바꿔요.
# 전체 배경/버튼/포인트 색은 .streamlit/config.toml 의 테마 색상을 따르고,
# 여기서는 그 테마로는 부족한 세부 꾸밈(카드 모양, 그라데이션 헤더 등)만
# 추가로 얹어요.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* 전체적으로 조금 더 둥글고 부드러운 느낌 */
    .stButton > button, .stDownloadButton > button {
        border-radius: 999px;
        border: none;
        padding: 0.6em 1.4em;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(255, 143, 171, 0.35);
        transition: transform 0.15s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        transform: translateY(-1px);
    }

    /* 상단 타이틀 영역을 부드러운 핑크 그라데이션 배너로 */
    .hero-banner {
        background: linear-gradient(135deg, #FFE3ED 0%, #FFF3F7 100%);
        border-radius: 20px;
        padding: 1.6em 1.8em;
        margin-bottom: 1.4em;
        border: 1px solid #FFD6E4;
    }
    .hero-banner h1 {
        margin: 0 0 0.3em 0;
        color: #4A3238;
    }
    .hero-banner p {
        margin: 0;
        color: #7A5C64;
    }

    /* 카메라를 켜기 전 안내 카드 */
    .camera-placeholder {
        background: #FFF3F7;
        border: 1.5px dashed #FFB6C9;
        border-radius: 18px;
        padding: 2.4em 1.5em;
        text-align: center;
        color: #7A5C64;
    }
    .camera-placeholder .big-emoji {
        font-size: 2.6em;
        margin-bottom: 0.2em;
    }

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
    .stTabs [aria-selected="true"] {
        color: #FF6B94 !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #FF8FAB !important;
    }

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


st.markdown(
    """
    <div class="hero-banner">
        <h1>💄 나에게 맞는 파운데이션 찾기</h1>
        <p>색상 카드와 얼굴을 같이 두고 촬영한 사진을 올려주세요.
        카메라/조명에 따른 색 왜곡을 자동으로 보정해서 가장 비슷한 파운데이션을 추천해드려요.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

shades, source = load_shades()
if not shades:
    st.error("파운데이션 데이터베이스가 비어 있어요. 관리자 페이지에서 색상을 추가해주세요.")
    st.stop()

if "camera_active" not in st.session_state:
    st.session_state.camera_active = False

# ---------------------------------------------------------------------------
# 1단계: 컬러체커 카드 유무. 다들 카드를 갖고 있는 건 아니니까, 카드 없이도
# (정확도는 떨어지더라도) 결과를 볼 수 있는 경로를 따로 둬요.
# ---------------------------------------------------------------------------
st.markdown("#### 1️⃣ 컬러체커 카드가 있으신가요?")
has_card = st.radio(
    "컬러체커 카드 유무",
    ["📇 카드 있어요 (더 정확해요)", "🙅 카드 없어요"],
    horizontal=True,
    label_visibility="collapsed",
) == "📇 카드 있어요 (더 정확해요)"

if not has_card:
    st.caption(
        "⚠️ 카드가 없으면 카메라/조명에 따른 색 왜곡을 보정할 기준이 없어서, "
        "카드가 있을 때보다 추천 정확도가 낮을 수 있어요."
    )

# ---------------------------------------------------------------------------
# 2단계: 추천에 사용할 색상 범위. "정확측정"은 실제 사진 촬영으로 정밀하게
# 측정해 넣은 색상만, "전체"는 그 외에 추가된 색상까지 전부 포함해요.
# ---------------------------------------------------------------------------
st.markdown("#### 2️⃣ 어떤 데이터에서 추천받을까요?")
n_verified = sum(1 for s in shades if s.get("verified", True))
scope = st.radio(
    "추천 범위",
    [f"✅ 정확측정 데이터에서만 ({n_verified}개)", f"📦 전체 데이터에서 ({len(shades)}개)"],
    horizontal=True,
    label_visibility="collapsed",
)
use_verified_only = scope.startswith("✅")
shade_pool = [s for s in shades if s.get("verified", True)] if use_verified_only else shades

if not shade_pool:
    st.error("선택하신 범위(정확측정 데이터)에는 아직 등록된 색상이 없어요. '전체 데이터'를 선택해주세요.")
    st.stop()

st.divider()

tab1, tab2 = st.tabs(["📷 카메라로 촬영", "🖼️ 사진첩에서 선택"])
image_bgr = None
camera_instruction = (
    "색상 카드 + 얼굴이 같이 나오게 촬영해주세요" if has_card else "얼굴이 잘 보이게 촬영해주세요 (카드는 없어도 돼요)"
)
upload_instruction = (
    "카드 + 얼굴이 같이 나온 사진을 선택해주세요" if has_card else "얼굴이 잘 보이는 사진을 선택해주세요 (카드는 없어도 돼요)"
)

with tab1:
    if not st.session_state.camera_active:
        # 페이지에 들어오자마자 카메라가 자동으로 켜지지 않도록, 버튼을
        # 눌러야만 실제 카메라 위젯(st.camera_input)이 나타나게 해요.
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
        # Downscale very large photos for speed; keeps enough detail for
        # both card-patch and face-landmark detection.
        h, w = image_bgr.shape[:2]
        scale = 1600 / max(h, w)
        if scale < 1:
            image_bgr = cv2.resize(image_bgr, None, fx=scale, fy=scale)

        # 카드 위치는 완전 자동으로 찾아요: 카드 자체의 정해진 색상들이
        # 일정한 격자로 모여있는 부분을 사진에서 직접 찾기 때문에, 머리카락이나
        # 옷 무늬가 옆에 있어도 사람이 따로 잘라줄 필요가 없어요.
        result = recommend_foundation(image_bgr, shade_pool, top_n=3, has_card=has_card)

    if not result.success:
        st.error(result.message)
        if has_card:
            st.info("💡 색상 카드와 얼굴이 모두 선명하게 나오도록, 너무 어둡지 않은 곳에서 다시 촬영해보세요.")
        else:
            st.info("💡 얼굴이 선명하게 잘 보이도록, 너무 어둡지 않은 곳에서 다시 촬영해보세요.")
    else:
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
