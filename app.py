
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# =========================
# 1. CẤU HÌNH TRANG
# =========================
st.set_page_config(
    page_title="Dashboard Dự đoán Qua Môn",
    layout="wide"
)

# =========================
# 2. GIAO DIỆN
# =========================

st.markdown("""
<style>
    .stApp {
        background-color: #F5F7FA;
        color: #1E293B;
    }

    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"] {
        background-color: #F5F7FA;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Màu chữ chung */
    h1, h2, h3, h4, p, label,
    [data-testid="stCaptionContainer"],
    [data-testid="stMarkdownContainer"],
    [data-testid="stText"],
    [data-testid="stWidgetLabel"] p {
        color: #1E293B !important;
    }

    /* Tiêu đề */
    h1, h2, h3 {
        color: #1E3A5F !important;
    }

    /* Thẻ chỉ số */
    [data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 1px solid #DCE3EC;
        border-radius: 10px;
        padding: 16px;
    }

    [data-testid="stMetricLabel"] p {
        color: #475569 !important;
    }

    [data-testid="stMetricValue"] {
        color: #1E3A5F !important;
    }

    /* Container */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF;
        border-radius: 12px;
    }

    /* Form nhập dữ liệu */
    [data-testid="stForm"] {
        background-color: #FFFFFF !important;
        border: 1px solid #DCE3EC;
        border-radius: 10px;
        padding: 20px;
    }

    /* Ô nhập dữ liệu */
    input, textarea {
        color: #1E293B !important;
        background-color: #FFFFFF !important;
    }

    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
    }

    /* Nút bấm */
    div[data-testid="stFormSubmitButton"] button {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border: none;
        border-radius: 8px;
        padding: 10px 16px;
    }

    div[data-testid="stFormSubmitButton"] button p {
        color: #FFFFFF !important;
    }

    div[data-testid="stFormSubmitButton"] button:hover {
        background-color: #1D4ED8 !important;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# 3. TẢI MÔ HÌNH
# =========================
data = joblib.load("dashboard_data.pkl")
metrics = data["metrics"]

# =========================
# 4. TIÊU ĐỀ
# =========================
st.title("Dashboard Dự đoán Khả năng Qua Môn")
st.markdown(
    '<div class="subtitle">'
    'Mô hình Logistic Regression · Dữ liệu mô phỏng'
    '</div>',
    unsafe_allow_html=True
)

# =========================
# 5. CHỈ SỐ ĐÁNH GIÁ
# =========================
with st.container(border=True):
    st.subheader("1. Chỉ số đánh giá mô hình")
    st.caption("Kết quả đánh giá đã được lưu sẵn trên tập Test.")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Precision", f'{metrics["precision"]:.3f}')
    c2.metric("Recall", f'{metrics["recall"]:.3f}')
    c3.metric("F1-score", f'{metrics["f1"]:.3f}')
    c4.metric("Threshold", f'{metrics["threshold"]:.2f}')

# =========================
# 6. CONFUSION MATRIX
# =========================
with st.container(border=True):
    st.subheader(
            f"2. Confusion Matrix - Tập Test "
            f"({metrics['confusion_matrix'].sum()} mẫu)"
        )
    st.caption("So sánh kết quả dự đoán với nhãn thực tế trên tập Test.")

    cm = metrics["confusion_matrix"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor("white")

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Không qua môn", "Qua môn"]
    )

    display.plot(
        ax=ax,
        values_format="d",
        colorbar=False,
        cmap="Blues"
    )

    ax.set_xlabel("Nhãn dự đoán")
    ax.set_ylabel("Nhãn thực tế")
    ax.set_title("Kết quả phân loại")
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

# =========================
# 7. DỰ ĐOÁN SINH VIÊN
# =========================
with st.container(border=True):
    st.subheader("3. Dự đoán khả năng qua môn")
    st.caption("Nhập thông tin học tập để nhận kết quả dự đoán.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            study_hours = st.number_input(
                "Giờ học mỗi tuần",
                min_value=0.0,
                max_value=20.0,
                value=8.0,
                step=0.5
            )

            attendance = st.number_input(
                "Chuyên cần (%)",
                min_value=40.0,
                max_value=100.0,
                value=80.0,
                step=1.0
            )

        with col2:
            assignments = st.number_input(
                "Số bài tập đã nộp",
                min_value=0,
                max_value=10,
                value=7
            )

            midterm = st.number_input(
                "Điểm giữa kỳ",
                min_value=0.0,
                max_value=10.0,
                value=7.0,
                step=0.5
            )

        submitted = st.form_submit_button(
            "Dự đoán",
            use_container_width=True
        )

    if submitted:
        student = pd.DataFrame([{
            "GioHoc/Tuan": study_hours,
            "ChuyenCan": attendance,
            "SoBT": assignments,
            "diemGK": midterm
        }])

        student_scaled = data["scaler"].transform(student)

        probability = data["model"].predict_proba(
            student_scaled
        )[0, 1]

        threshold = data["threshold"]
        prediction = int(probability >= threshold)

        st.divider()

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.metric("Xác suất qua môn", f"{probability:.2%}")

        with result_col2:
            st.metric("Ngưỡng phân loại", f"{threshold:.2f}")

        if prediction == 1:
            st.success("Kết quả dự đoán: Qua môn")
        else:
            st.warning("Kết quả dự đoán: Không qua môn")



st.divider()
st.caption(
    "**Lựa chọn mô hình:** Logistic Regression được chọn vì phù hợp "
    "với bài toán phân loại nhị phân (qua môn/không qua môn), "
    "dễ triển khai, dễ diễn giải và cho phép ước tính xác suất qua môn."
)

