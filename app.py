from __future__ import annotations

import re
from pathlib import Path

import easyocr
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


DATA_PATH = Path("ingredients_seed.csv")

RISK_SCORE = {
    "低": 1,
    "低-中": 2,
    "中": 3,
    "中-高": 4,
    "高": 5,
}


@st.cache_data
def load_ingredient_db(path: Path) -> pd.DataFrame:
    """Load ingredient database from CSV and normalize matching keys."""
    if not path.exists():
        raise FileNotFoundError(f"找不到資料庫檔案：{path}")

    df = pd.read_csv(path)

    required_columns = {
        "ingredient",
        "chinese_name",
        "category",
        "function",
        "concern",
        "risk_level",
        "groups",
        "advice",
        "source",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少必要欄位：{', '.join(sorted(missing))}")

    df["match_key"] = df["ingredient"].astype(str).str.strip().str.lower()
    df["risk_score"] = df["risk_level"].map(RISK_SCORE).fillna(0).astype(int)
    return df


@st.cache_resource
def load_ocr_reader() -> easyocr.Reader:
    """Load EasyOCR reader once.

    English is enough for most cosmetic ingredient lists.
    Add 'ch_tra' if you also want Traditional Chinese OCR, but it will be heavier.
    """
    return easyocr.Reader(["en"], gpu=False)


def normalize_text(text: str) -> str:
    """Normalize OCR or pasted ingredient text."""
    return (
        text.lower()
        .replace("（", "(")
        .replace("）", ")")
        .replace("，", ",")
        .replace("；", ";")
        .replace("、", ",")
    )


def clean_ocr_text(text: str) -> str:
    """Clean OCR output for cosmetic ingredient parsing."""
    text = normalize_text(text)

    # Common OCR cleanup.
    text = text.replace("|", "l")
    text = text.replace("•", ",")
    text = text.replace("·", ",")
    text = text.replace("ingredients:", "")
    text = text.replace("ingredient:", "")

    # Keep letters, numbers, commas, semicolons, spaces, hyphens, dots, parentheses, slash.
    text = re.sub(r"[^a-z0-9,;.\-\s()/]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def split_ingredients(text: str) -> list[str]:
    """Split pasted ingredient list by common separators."""
    normalized = clean_ocr_text(text)

    # Ingredient lists usually use commas.
    parts = re.split(r"[,;\n]+", normalized)

    cleaned_parts = []
    for part in parts:
        item = part.strip(" .-")
        if len(item) >= 2:
            cleaned_parts.append(item)

    return cleaned_parts


def ocr_image(image: Image.Image, reader: easyocr.Reader) -> str:
    """Run OCR on a PIL image and return detected text."""
    image = image.convert("RGB")

    # Upscale slightly for small package text.
    width, height = image.size
    if max(width, height) < 1600:
        scale = 1600 / max(width, height)
        image = image.resize((int(width * scale), int(height * scale)))

    image_array = np.array(image)

    # detail=0 returns text only.
    results = reader.readtext(image_array, detail=0, paragraph=True)

    return "\n".join(results)


def match_ingredients(user_ingredients: list[str], db: pd.DataFrame) -> pd.DataFrame:
    """Match ingredients using exact matching and simple alias containment."""
    records = []
    db_by_key = {row["match_key"]: row for _, row in db.iterrows()}

    for item in user_ingredients:
        match = db_by_key.get(item)

        # Backup matching: sometimes OCR loses punctuation.
        if match is None:
            item_no_punct = re.sub(r"[^a-z0-9]+", " ", item).strip()
            for key, row in db_by_key.items():
                key_no_punct = re.sub(r"[^a-z0-9]+", " ", key).strip()
                if item_no_punct == key_no_punct:
                    match = row
                    break

        if match is not None:
            record = match.to_dict()
            record["input_ingredient"] = item
            record["matched"] = True
            records.append(record)
        else:
            records.append(
                {
                    "input_ingredient": item,
                    "ingredient": item,
                    "chinese_name": "",
                    "category": "未收錄",
                    "function": "",
                    "concern": "資料庫尚未收錄，建議查詢官方資料庫或成分安全評估資料。",
                    "risk_level": "未知",
                    "groups": "",
                    "advice": "目前無法判斷，請勿直接視為安全或危險。",
                    "source": "",
                    "match_key": item,
                    "risk_score": 0,
                    "matched": False,
                }
            )

    return pd.DataFrame(records)


def summarize_groups(result_df: pd.DataFrame) -> pd.DataFrame:
    """Count how many matched ingredients are flagged for each group."""
    group_counter: dict[str, int] = {}

    for groups in result_df["groups"].fillna(""):
        for group in str(groups).split(";"):
            group = group.strip()
            if group:
                group_counter[group] = group_counter.get(group, 0) + 1

    if not group_counter:
        return pd.DataFrame(columns=["族群", "提醒次數"])

    return (
        pd.DataFrame(
            [{"族群": group, "提醒次數": count} for group, count in group_counter.items()]
        )
        .sort_values("提醒次數", ascending=False)
        .reset_index(drop=True)
    )


def overall_risk_label(result_df: pd.DataFrame) -> str:
    """Create a simple overall risk label based on matched ingredient scores."""
    matched = result_df[result_df["matched"]]

    if matched.empty:
        return "未知：目前輸入成分沒有命中資料庫。"

    max_score = int(matched["risk_score"].max())
    medium_or_above_count = int((matched["risk_score"] >= 3).sum())

    if max_score >= 5:
        return "高：含有需高度注意成分，建議確認使用情境與族群限制。"
    if max_score >= 4 or medium_or_above_count >= 3:
        return "中-高：有多個需注意成分，敏感族群建議保守使用。"
    if max_score >= 3:
        return "中：含有可能刺激或致敏成分，建議依膚質觀察反應。"
    return "低：目前資料庫命中的成分多屬低風險，但仍需考慮濃度與使用頻率。"


def display_results(result_df: pd.DataFrame) -> None:
    """Display risk analysis results."""
    display_columns = [
        "input_ingredient",
        "chinese_name",
        "category",
        "function",
        "concern",
        "risk_level",
        "groups",
        "advice",
        "source",
    ]

    st.subheader("整體風險判斷")
    st.info(overall_risk_label(result_df))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("成分類型統計")
        category_counts = (
            result_df["category"]
            .value_counts()
            .rename_axis("成分類型")
            .reset_index(name="數量")
        )
        st.bar_chart(category_counts.set_index("成分類型"))

    with col2:
        st.subheader("需注意族群統計")
        group_summary = summarize_groups(result_df)
        if group_summary.empty:
            st.write("目前沒有對應族群提醒。")
        else:
            st.bar_chart(group_summary.set_index("族群"))

    st.subheader("辨識結果")
    st.dataframe(
        result_df[display_columns].rename(
            columns={
                "input_ingredient": "輸入成分",
                "chinese_name": "中文名稱",
                "category": "成分類型",
                "function": "常見用途",
                "concern": "可能風險",
                "risk_level": "風險等級",
                "groups": "需注意族群",
                "advice": "使用建議",
                "source": "資料來源",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    unmatched = result_df[~result_df["matched"]]
    if not unmatched.empty:
        st.warning(
            f"有 {len(unmatched)} 個成分尚未收錄於資料庫。"
            "這不代表安全或危險，只代表需要補充資料。"
        )


def main() -> None:
    st.set_page_config(
        page_title="化妝品成分風險辨識工具",
        page_icon="🧴",
        layout="wide",
    )

    st.title("🧴 化妝品成分風險辨識工具")
    st.caption("拍攝或上傳商品背面成分表，OCR 擷取文字後進行風險辨識。")

    db = load_ingredient_db(DATA_PATH)

    tab_camera, tab_upload, tab_manual = st.tabs(
        ["📷 拍照掃描", "🖼️ 上傳照片", "⌨️ 手動輸入"]
    )

    extracted_text = ""

    with tab_camera:
        st.write("請拍攝商品背面的 Ingredients 成分表。盡量保持光線充足、文字清楚、不要反光。")
        camera_image = st.camera_input("拍攝成分表")

        if camera_image is not None:
            image = Image.open(camera_image)
            st.image(image, caption="拍攝影像", use_container_width=True)

            if st.button("對拍攝影像執行 OCR", key="ocr_camera"):
                with st.spinner("OCR 辨識中，第一次載入可能需要比較久..."):
                    reader = load_ocr_reader()
                    extracted_text = ocr_image(image, reader)
                    st.session_state["ocr_text"] = extracted_text

    with tab_upload:
        uploaded_file = st.file_uploader(
            "上傳商品背面成分表照片",
            type=["jpg", "jpeg", "png", "webp"],
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="上傳影像", use_container_width=True)

            if st.button("對上傳影像執行 OCR", key="ocr_upload"):
                with st.spinner("OCR 辨識中，第一次載入可能需要比較久..."):
                    reader = load_ocr_reader()
                    extracted_text = ocr_image(image, reader)
                    st.session_state["ocr_text"] = extracted_text

    with tab_manual:
        st.write("也可以直接貼上商品成分表。")
        manual_text = st.text_area(
            "手動輸入成分",
            value=(
                "Aqua, Glycerin, Alcohol Denat., Fragrance, "
                "Salicylic Acid, Phenoxyethanol, Titanium Dioxide"
            ),
            height=140,
        )

        if st.button("分析手動輸入內容", key="analyze_manual"):
            ingredients = split_ingredients(manual_text)
            result_df = match_ingredients(ingredients, db)
            display_results(result_df)

    if "ocr_text" in st.session_state:
        st.divider()
        st.subheader("OCR 擷取結果")
        st.write("OCR 可能會有錯，建議先手動修正再分析。")

        corrected_text = st.text_area(
            "可修正 OCR 文字",
            value=st.session_state["ocr_text"],
            height=180,
        )

        cleaned_text = clean_ocr_text(corrected_text)

        with st.expander("清理後文字"):
            st.write(cleaned_text)

        if st.button("分析 OCR 成分"):
            ingredients = split_ingredients(corrected_text)

            if not ingredients:
                st.warning("沒有偵測到可分析的成分，請修正 OCR 文字或改用手動輸入。")
                return

            result_df = match_ingredients(ingredients, db)
            display_results(result_df)

    st.divider()
    st.caption(
        "注意：本工具僅作為初步資訊整理與風險提醒，不構成醫療診斷、皮膚科建議或產品安全保證。"
    )


if __name__ == "__main__":
    main()