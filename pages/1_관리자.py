import uuid
import numpy as np
import cv2
from PIL import Image
import streamlit as st
from streamlit_cropper import st_cropper

from src.calibration import calibrate_from_image
from src.color_match import rgb_to_lab
from src import github_storage

st.set_page_config(page_title="관리자 - 파운데이션 DB", page_icon="🔒")


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# ---- password gate ----
if "admin_ok" not in st.session_state:
    st.session_state.admin_ok = False

if not st.session_state.admin_ok:
    st.title("🔒 관리자 로그인")
    pw = st.text_input("비밀번호", type="password")
    if st.button("로그인"):
        correct = st.secrets.get("admin", {}).get("password")
        if correct and pw == correct:
            st.session_state.admin_ok = True
            st.rerun()
        else:
            st.error("비밀번호가 틀렸어요")
    st.stop()

st.title("🗂️ 파운데이션 색상 DB 관리")

try:
    shades, sha = github_storage.read_shades()
except Exception as e:
    st.error(f"DB를 불러오지 못했어요: {e}")
    st.stop()

# ---- brand-grouped listing ----
st.subheader("등록된 색상")
brands = sorted(set(s["brand"] for s in shades))
if not brands:
    st.info("아직 등록된 색상이 없어요.")
else:
    for brand in brands:
        with st.expander(f"{brand} ({sum(1 for s in shades if s['brand']==brand)}개)", expanded=False):
            for s in [s for s in shades if s["brand"] == brand]:
                c1, c2, c3 = st.columns([3, 3, 1])
                c1.write(f"**{s['name']}**")
                c2.write(f"L={s['L']:.1f}, a={s['a']:.1f}, b={s['b']:.1f}")
                if c3.button("삭제", key=f"del-{s['id']}"):
                    github_storage.delete_shade(s["id"])
                    st.cache_data.clear()
                    st.rerun()

st.divider()

# ---- add new shade ----
st.subheader("새 색상 추가")
st.caption("포토부스에서 색상카드 + 파운데이션 스와치를 같은 카메라로 5회 촬영한 사진을 올려주세요.")

brand = st.text_input("브랜드명")
name = st.text_input("색상명")
photos = st.file_uploader(
    "사진 업로드 (최대 5장)", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

if photos:
    if len(photos) > 5:
        st.warning("사진은 최대 5장까지만 사용돼요. 앞의 5장만 사용할게요.")
        photos = photos[:5]

    labs = []
    st.write("각 사진에서 **파운데이션이 발린 부분만** 네모 박스로 선택해주세요:")
    for i, photo in enumerate(photos):
        img = Image.open(photo)
        st.write(f"사진 {i+1}/{len(photos)}")
        cropped = st_cropper(img, realtime_update=True, box_color="#FF4B4B", aspect_ratio=None, key=f"crop-{i}")

        bgr_full = pil_to_bgr(img)
        calib = calibrate_from_image(bgr_full)
        if not calib.success:
            st.error(f"사진 {i+1}: 색상카드 인식 실패 ({calib.message}) — 이 사진은 제외돼요")
            continue

        crop_rgb = np.array(cropped.convert("RGB"))
        median_rgb = np.median(crop_rgb.reshape(-1, 3), axis=0)
        corrected_rgb = np.clip(calib.correction_matrix @ np.append(median_rgb, 1.0), 0, 255)
        lab = rgb_to_lab(corrected_rgb)
        labs.append(lab)
        st.caption(f"사진 {i+1} 보정 후 Lab: L={lab[0]:.1f}, a={lab[1]:.1f}, b={lab[2]:.1f} (카드 보정 오차 {calib.mean_delta_e:.1f})")

    if labs:
        mean_lab = np.mean(labs, axis=0)
        st.info(f"**{len(labs)}장 평균** → L={mean_lab[0]:.2f}, a={mean_lab[1]:.2f}, b={mean_lab[2]:.2f}")

        if st.button("이 색상으로 저장", type="primary", disabled=not (brand and name)):
            new_shade = {
                "id": f"{brand}-{name}-{uuid.uuid4().hex[:6]}".lower().replace(" ", "-"),
                "brand": brand,
                "name": name,
                "L": round(float(mean_lab[0]), 3),
                "a": round(float(mean_lab[1]), 3),
                "b": round(float(mean_lab[2]), 3),
            }
            try:
                github_storage.add_shade(new_shade)
                st.cache_data.clear()
                st.success("저장 완료!")
                st.rerun()
            except Exception as e:
                st.error(f"저장 실패: {e}")
        if not (brand and name):
            st.caption("⚠️ 브랜드명과 색상명을 입력해야 저장할 수 있어요.")
