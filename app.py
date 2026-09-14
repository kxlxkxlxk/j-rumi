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

# Front-camera live preview only (mirror it so aiming the shot feels like a
# normal mirror, the way phone selfie cameras do). This is a pure CSS flip
# of what's on screen -- the actual photo bytes st.camera_input hands back
# to the code below are untouched, so card/face detection keeps working on
# the real (unmirrored) capture exactly as before.
st.markdown(
    """
    <style>
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


st.title("💄 나에게 맞는 파운데이션 찾기")
st.write(
    "색상 카드와 얼굴을 같이 두고 촬영한 사진을 올려주세요. "
    "카메라/조명에 따른 색 왜곡을 자동으로 보정해서 가장 비슷한 파운데이션을 추천해드려요."
)

shades, source = load_shades()
if not shades:
    st.error("파운데이션 데이터베이스가 비어 있어요. 관리자 페이지에서 색상을 추가해주세요.")
    st.stop()

tab1, tab2 = st.tabs(["📷 카메라로 촬영", "🖼️ 사진 업로드"])
image_bgr = None

with tab1:
    cam_img = st.camera_input("색상 카드 + 얼굴이 같이 나오게 촬영해주세요")
    if cam_img is not None:
        image_bgr = pil_to_bgr(Image.open(cam_img))

with tab2:
    uploaded = st.file_uploader("이미지 파일 선택", type=["jpg", "jpeg", "png"])
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
        result = recommend_foundation(image_bgr, shades, top_n=3)

    if not result.success:
        st.error(result.message)
        st.info("💡 색상 카드와 얼굴이 모두 선명하게 나오도록, 너무 어둡지 않은 곳에서 다시 촬영해보세요.")
    else:
        st.success("분석 완료!")
        best = result.matches[0]
        st.subheader(f"✨ 추천: {best['brand']} · {best['name']}")
        st.markdown(
            f"<div style='width:100%;height:60px;border-radius:8px;"
            f"background-color:{lab_to_hex(best['L'], best['a'], best['b'])};'></div>",
            unsafe_allow_html=True,
        )

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

        with st.expander("🔍 색 보정 전/후 비교", expanded=True):
            correction_matrix = np.array(result.debug["correction_matrix"])
            rgb_before = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            rgb_after = apply_correction_image(rgb_before, correction_matrix)

            st.caption("사진 전체에 카드 보정을 그대로 적용해보면 이렇게 달라져요 (카메라/조명 왜곡 제거):")
            c1, c2 = st.columns(2)
            c1.image(rgb_before, caption="보정 전 (원본)", use_container_width=True)
            c2.image(rgb_after, caption="보정 후", use_container_width=True)

            st.caption("피부색 샘플링에 사용된 영역별 색 (보정 전 → 보정 후):")
            for rd in result.debug["regions"]:
                rc1, rc2, rc3 = st.columns([1, 1, 2])
                raw_hex = "#%02x%02x%02x" % tuple(int(max(0, min(255, v))) for v in rd["raw_rgb"])
                corr_hex = "#%02x%02x%02x" % tuple(int(max(0, min(255, v))) for v in rd["corrected_rgb"])
                rc1.markdown(f"<div style='width:100%;height:32px;border-radius:4px;background-color:{raw_hex};'></div>", unsafe_allow_html=True)
                rc2.markdown(f"<div style='width:100%;height:32px;border-radius:4px;background-color:{corr_hex};'></div>", unsafe_allow_html=True)
                tag = " ⭐최종채택" if rd.get("used_as_final") else (" (제외됨)" if rd.get("excluded_as_outlier") else "")
                rc3.caption(f"{rd['name']}{tag} — Lab: L={rd['lab'][0]:.1f}, a={rd['lab'][1]:.1f}, b={rd['lab'][2]:.1f}")

        with st.expander("자세히 보기 (분석 정보)"):
            st.json(
                {
                    "보정 후 피부색 Lab": [round(float(x), 2) for x in result.corrected_lab],
                    "최종 채택 영역": result.debug["final_region"],
                    "카드 보정 평균 오차": round(result.debug["calibration_mean_error"], 2),
                    "DB 출처": source,
                    "영역별 샘플": result.debug["regions"],
                }
            )
