import uuid
import numpy as np
import cv2
from PIL import Image
import streamlit as st
from streamlit_cropper import st_cropper

from src.calibration import calibrate_from_image, capture_card_reference
from src.color_match import rgb_to_lab
from src import github_storage
from src import reference_colors

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
    st.write("각 사진에서 **파운데이션이 발린 부분만** 네모 박스로 선택해주세요 (색상 카드는 자동으로 인식돼요):")
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

st.divider()

# ---- optional: custom card reference (fixes "정확하지만 과하게 보정됨") ----
st.subheader("📇 카드 기준값 재설정 (선택)")
st.caption(
    "지금은 컬러체커 클래식의 공식 발표 수치를 정답으로 놓고 보정하고 있어요. "
    "이게 특정 카드/조명 조합에서 과하게 색이 튀어 보이면, 카드만 깨끗하게 "
    "(반사·그림자 없이 정면에서 크게) 찍은 사진 한 장을 등록해서, 그 사진에서 "
    "측정한 실측값을 새 기준으로 바꿀 수 있어요."
)

try:
    current_ref, _ref_sha = github_storage.read_card_reference()
except Exception as e:
    current_ref = None
    st.warning(f"현재 기준값 상태를 확인하지 못했어요: {e}")

if current_ref and current_ref.get("patches"):
    st.success(f"지금은 커스텀 기준값을 쓰고 있어요 ({len(current_ref['patches'])}/24개 패치 등록됨)")
    if st.button("공식 기준값으로 되돌리기"):
        try:
            github_storage.delete_card_reference()
            reference_colors.clear_reference_cache()
            st.success("공식 기준값으로 되돌렸어요")
            st.rerun()
        except Exception as e:
            st.error(f"되돌리기 실패: {e}")
else:
    st.caption("지금은 공식 기준값을 쓰고 있어요.")

ref_photo = st.file_uploader("카드만 깨끗하게 찍은 사진 (얼굴 없이 카드만)", type=["jpg", "jpeg", "png"], key="ref_photo")
if ref_photo is not None:
    ref_bgr = pil_to_bgr(Image.open(ref_photo))
    ok, msg, patches = capture_card_reference(ref_bgr)
    if not ok:
        st.error(msg)
    else:
        st.success(msg)
        swatch_cols = st.columns(6)
        for i, (pname, rgb) in enumerate(patches.items()):
            with swatch_cols[i % 6]:
                hexcolor = "#%02x%02x%02x" % tuple(int(max(0, min(255, v))) for v in rgb)
                st.markdown(
                    f"<div style='width:100%;height:28px;border-radius:4px;background-color:{hexcolor};'></div>",
                    unsafe_allow_html=True,
                )
                st.caption(pname)

        if st.button("이 값을 새 기준으로 저장", type="primary"):
            try:
                github_storage.write_card_reference(patches)
                reference_colors.clear_reference_cache()
                st.success("저장 완료! 이제부터 이 값을 기준으로 보정돼요.")
                st.rerun()
            except Exception as e:
                st.error(f"저장 실패: {e}")
