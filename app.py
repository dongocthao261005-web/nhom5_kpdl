import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --- 1. CẤU HÌNH TRANG (Phải để dòng này trên cùng) ---
st.set_page_config(page_title="Customer Segmentation Pro", page_icon="✨", layout="wide")

# --- 2. CSS CUSTOM CHO GIAO DIỆN "NGỰA NGỰA" NHƯNG ĐỔI TONE MÀU ---
st.markdown("""
<style>
/* Gradient text cho tiêu đề chính (Tone Xanh Tím sang trọng) */
.title-text {
    background: -webkit-linear-gradient(45deg, #4158D0, #C850C0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 45px;
    font-weight: 900;
    margin-bottom: 0px;
}
/* Làm đẹp các Tab chuyển trang */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0px 0px;
    padding: 10px 25px;
    background-color: #f8f9fa;
    border: 1px solid #e9ecef;
    border-bottom: none;
    font-size: 16px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(45deg, #4158D0, #C850C0);
    color: white !important;
    font-weight: bold;
    box-shadow: 0px -2px 10px rgba(200, 80, 192, 0.3);
}
/* Style cho các thẻ số liệu (Metric) */
div[data-testid="metric-container"] {
    background-color: #ffffff;
    border: 1px solid #f0f2f6;
    padding: 15px 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.05);
    border-left: 5px solid #C850C0;
}
</style>
""", unsafe_allow_html=True)

# --- 3. TIÊU ĐỀ CHÍNH ---
st.markdown('<p class="title-text">✨ BẢNG ĐIỀU KHIỂN PHÂN KHÚC KHÁCH HÀNG</p>', unsafe_allow_html=True)
st.markdown("*Ứng dụng kỹ thuật gom nhóm trong khai thác dữ liệu để phân khúc thị trường và cá nhân hóa Marketing.*")
st.divider()

# --- 4. SIDEBAR (THANH ĐIỀU HƯỚNG BÊN TRÁI) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3126/3126647.png", width=120)
    st.header("⚙️ Thiết lập Phân tích")
    uploaded_file = st.file_uploader("📂 Tải file dữ liệu (.csv/.xlsx)", type=["csv", "xlsx"])
    
    st.markdown("---")
    st.subheader("🎛️ Tham số K-Means")
    # YÊU CẦU: Chỉ cho phép chọn 4 hoặc 5 cụm
    n_clusters = st.radio("Chọn số lượng phân khúc thị trường (K):", options=[4, 5], index=1, horizontal=True)

# --- 5. LUỒNG XỬ LÝ CHÍNH ---
if uploaded_file is not None:
    # Đọc dữ liệu
    df = pd.read_csv(uploaded_file)
    
    # Tiền xử lý (Lọc nhiễu thu nhập nếu có)
    if 'Annual Income (k$)' in df.columns:
        df_clean = df[df['Annual Income (k$)'] <= 130].copy()
    else:
        df_clean = df.copy()

    features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    X = df_clean[features].dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Chạy K-Means
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
    df_clean['Cluster'] = kmeans.fit_predict(X_scaled)
    # Gắn nhãn cho đẹp ("Nhóm 1", "Nhóm 2" thay vì 0, 1)
    df_clean['Cluster'] = df_clean['Cluster'].apply(lambda x: f"Nhóm {x+1}")

    # --- 6. PHÂN CHIA GIAO DIỆN THÀNH 3 TABS ---
    tab1, tab2, tab3 = st.tabs(["📊 Tổng quan Dữ liệu", "✨ Biểu đồ Phân cụm (3D)", "🎯 Chiến lược Marketing"])

    # Tốc độ tải và tổng quan dữ liệu
    with tab1:
        st.subheader("1. Thống kê chỉ số nhanh")
        col1, col2, col3 = st.columns(3)
        col1.metric("👥 Tổng khách hàng", f"{len(df_clean)} người")
        col2.metric("💰 Thu nhập trung bình", f"${round(df_clean['Annual Income (k$)'].mean() * 1000):,}")
        col3.metric("⭐ Điểm chi tiêu TB", round(df_clean['Spending Score (1-100)'].mean(), 1))
        
        st.write("---")
        st.subheader("2. Dữ liệu sau xử lý")
        st.dataframe(df_clean.head(15), use_container_width=True)

    # Trực quan hóa Biểu đồ 3D
    with tab2:
        st.subheader(f"Mô hình phân bổ {n_clusters} nhóm khách hàng")
        # Trả lại màu Plotly rực rỡ, tương phản cao cho dễ nhìn các cụm
        fig = px.scatter_3d(
            df_clean, x='Age', y='Annual Income (k$)', z='Spending Score (1-100)',
            color='Cluster',
            hover_data=['Annual Income (k$)', 'Spending Score (1-100)'],
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        # Ẩn nền trắng của biểu đồ 3D để tiệp màu với trang web
        fig.update_layout(
            scene=dict(
                xaxis_title='Độ tuổi',
                yaxis_title='Thu nhập (k$)',
                zaxis_title='Điểm chi tiêu'
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Đề xuất chiến lược (Giữ nguyên cấu trúc, chỉ thay đổi màu Custom HTML)
    with tab3:
        st.subheader("Phân tích đặc trưng & Đề xuất thực tế")
        st.info("💡 Hệ thống tự động phân tích hành vi của từng nhóm để đưa ra chiến lược kinh doanh cá nhân hóa.")
        
        # Nhóm dữ liệu để lấy giá trị trung bình từng cụm
        cluster_summary = df_clean.groupby('Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean().reset_index()
        
        # Hiển thị dạng lưới 2 cột cho gọn gàng
        cols = st.columns(2)
        for i, row in cluster_summary.iterrows():
            with cols[i % 2]:
                with st.expander(f"📌 {row['Cluster']} - Xem chi tiết chiến lược", expanded=True):
                    st.write(f"**Tuổi TB:** {int(row['Age'])} | **Thu nhập TB:** ${int(row['Annual Income (k$)']*1000):,} | **Chi tiêu TB:** {int(row['Spending Score (1-100)'])}")
                    st.write("---")
                    
                    # Logic tự động đánh giá đặc trưng (Custom CSS Box thay cho Bootstrap Alerts)
                    if row['Annual Income (k$)'] > 60 and row['Spending Score (1-100)'] > 60:
                        st.markdown('<div style="background-color: #F4EEFF; color: #424874; padding: 12px; border-radius: 8px; margin-bottom: 10px;">💎 <b>Đặc trưng:</b> Khách hàng VIP (Thu nhập cao, chi tiêu rất mạnh).</div>', unsafe_allow_html=True)
                        st.write("**Chiến lược Marketing:** Gửi thiệp cảm ơn định kỳ, ưu tiên trải nghiệm dịch vụ chăm sóc cao cấp, mời tham dự sự kiện ra mắt sản phẩm mới.")
                    
                    elif row['Annual Income (k$)'] > 60 and row['Spending Score (1-100)'] < 40:
                        st.markdown('<div style="background-color: #E2F0CB; color: #5B8A72; padding: 12px; border-radius: 8px; margin-bottom: 10px;">🛡️ <b>Đặc trưng:</b> Khách hàng Tiềm năng (Thu nhập cao, ít chi tiêu).</div>', unsafe_allow_html=True)
                        st.write("**Chiến lược Marketing:** Cung cấp thẻ thành viên hạng sang, cam kết chất lượng dài hạn, tiếp thị các sản phẩm có tính tích lũy/đầu tư.")
                    
                    elif row['Annual Income (k$)'] < 40 and row['Spending Score (1-100)'] > 60:
                        st.markdown('<div style="background-color: #FFDFD3; color: #B35454; padding: 12px; border-radius: 8px; margin-bottom: 10px;">🛍️ <b>Đặc trưng:</b> Khách hàng Bốc đồng (Thu nhập thấp, chi tiêu mạnh).</div>', unsafe_allow_html=True)
                        st.write("**Chiến lược Marketing:** Gửi thông báo Push Notification qua app, đẩy mạnh Flash Sale, tặng Voucher giảm giá có thời hạn ngắn.")
                    
                    elif row['Annual Income (k$)'] < 40 and row['Spending Score (1-100)'] < 40:
                        st.markdown('<div style="background-color: #E8F6EF; color: #2B7A78; padding: 12px; border-radius: 8px; margin-bottom: 10px;">🛒 <b>Đặc trưng:</b> Khách hàng Tiết kiệm (Thu nhập thấp, chi tiêu dè dặt).</div>', unsafe_allow_html=True)
                        st.write("**Chiến lược Marketing:** Khuyến mãi mua Combo để kích cầu, gợi ý các mặt hàng thiết yếu (Cross-selling) với giá cả hợp lý.")
                    
                    else:
                        st.markdown('<div style="background-color: #F0F5F9; color: #1E2022; padding: 12px; border-radius: 8px; margin-bottom: 10px;">📊 <b>Đặc trưng:</b> Khách hàng Tiêu chuẩn (Mức thu nhập và chi tiêu đều đặn).</div>', unsafe_allow_html=True)
                        st.write("**Chiến lược Marketing:** Xây dựng chương trình tích điểm đổi quà, gửi khảo sát ý kiến kèm mã giảm giá để kích thích mua thêm.")

else:
    st.warning("👈 Vui lòng tải file **Mall_Customers.csv** ở thanh điều hướng bên trái để hệ thống bắt đầu render giao diện!")