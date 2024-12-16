import streamlit as st
import pandas as pd
import pickle
import numpy as np
from surprise import Reader, Dataset, SVD, SVDpp, NMF, SlopeOne, KNNBasic, KNNBaseline, KNNWithMeans, KNNWithZScore, CoClustering, BaselineOnly
from surprise.model_selection.validation import cross_validate

# Hàm gợi ý sản phẩm tương tự từ content-based
def get_recommendations(df, ma_san_pham, cosine_sim, df_hinh, nums=5):
    matching_indices = df.index[df['ma_san_pham'] == ma_san_pham].tolist()
    if not matching_indices:
        st.warning(f"Không tìm thấy sản phẩm với ID: {ma_san_pham}")
        return pd.DataFrame()
    idx = matching_indices[0]

    # Tính toán độ tương đồng
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:nums + 1]
    product_indices = [i[0] for i in sim_scores]

    # Lấy các sản phẩm gợi ý
    recommended_products = df.iloc[product_indices]

    # Gắn hình ảnh từ df_hinh
    recommended_products_with_images = pd.merge(recommended_products, df_hinh[['ma_san_pham', 'hinh_anh']], on='ma_san_pham', how='left')
    
    return recommended_products_with_images

# Hàm hiển thị các sản phẩm gợi ý
def display_recommended_products(recommended_products, cols=5):
    for i in range(0, len(recommended_products), cols):
        col_list = st.columns(cols)
        for j, col in enumerate(col_list):
            if i + j < len(recommended_products):
                product = recommended_products.iloc[i + j]
                with col:
                    st.write(f"### {product['ten_san_pham']}")
                    gia_ban_formatted = f"{product['gia_ban']:,.0f}".replace(",", ".")
                    gia_goc_formatted = f"{product['gia_goc']:,.0f}".replace(",", ".")
                    st.write(f"**Giá bán:** {gia_ban_formatted} VND")
                    st.write(f"**Giá gốc:** {gia_goc_formatted} VND")
                    expander = st.expander("Mô tả")
                    product_description = product['mo_ta']
                    truncated_description = ' '.join(product_description.split()[:100]) + "..."
                    expander.write(truncated_description)

                    # Hiển thị hình ảnh
                    if 'hinh_anh' in product and pd.notna(product['hinh_anh']):
                        image_path = product['hinh_anh']
                        
                        # Kiểm tra loại tệp hình ảnh (PNG hoặc JPG)
                        if image_path.lower().endswith(('png', 'jpg', 'jpeg')):
                            st.image(image_path, use_container_width=True)  # Hiển thị ảnh nếu đúng định dạng
                        else:
                            st.write("**Định dạng ảnh không hợp lệ.**")
                    else:
                        st.write("**Không có ảnh sản phẩm**")

# Hàm gợi ý sản phẩm cho khách hàng dựa trên collaborative filtering
def recommendation_by_user(userId, df, df_products, algorithm):
    df_select = df[(df['ma_khach_hang'] == userId) & (df['so_sao'] >= 3)]
    df_select = df_select.set_index('ma_san_pham')
    df_score = pd.DataFrame(df['ma_san_pham'].unique(), columns=['ma_san_pham'])
    df_score['EstimateScore'] = df_score['ma_san_pham'].apply(lambda x: algorithm.predict(userId, x).est)
    df_score = df_score.sort_values(by=['EstimateScore'], ascending=False)
    df_score = df_score.drop_duplicates()
    df2_selected = df_products[['ma_san_pham', 'ten_san_pham', 'gia_ban', 'gia_goc', 'mo_ta']]
    df_score = df_score.merge(df2_selected, on='ma_san_pham', how='left')

    # Sau khi gợi ý xong, gắn hình ảnh
    df_score_with_images = pd.merge(df_score, df_hinh[['ma_san_pham', 'hinh_anh']], on='ma_san_pham', how='left')
    return df_score_with_images.head(5)

# Tải dữ liệu
df_hinh = pd.read_csv('/mount/src/recommendation_system/product_recommendation-main/Hinhanh.csv')
df_products = pd.read_csv('San_pham_2xuly.csv')
df_customers = pd.read_csv('Khach_hang_2xuly.csv')
df = pd.read_csv('Danh_gia_final.csv')

# Tải mô hình
with open('product_surprise.pkl', 'rb') as f:
    algorithm_loaded = pickle.load(f)
with open('products_cosine_sim.pkl', 'rb') as f:
    cosine_sim_new = pickle.load(f)

limited_products = df_products.head(20)
limited_customers = df_customers.head(20)

# Giao diện với Streamlit
# Tiêu đề trang
st.image('hasaki1.jpg', use_container_width=True)
st.title("💎 Hệ thống gợi ý sản phẩm Recommender System 💎")

# Menu ở sidebar
menu = ["Business Objective","Build Project", "Hiển thị chart", "Hệ thống gợi ý"]
choice = st.sidebar.selectbox('Menu', menu)

st.sidebar.write("""#### Thành viên thực hiện:
                 Phan Văn Minh & Cao Anh Hào""")
st.sidebar.write("""#### Giảng viên hướng dẫn: Ms Phương """)
st.sidebar.write("""#### Ngày báo cáo tốt nghiệp: 16/12/2024""")

# Phần lựa chọn menu
if choice == 'Business Objective':
    st.subheader("Business Objective")
    st.write("""  
    ###### HASAKI.VN là hệ thống cửa hàng mỹ phẩm chính hãng và dịch vụ chăm sóc sắc đẹp chuyên sâu, với mạng lưới cửa hàng phủ rộng trên toàn quốc. Mục tiêu của họ là nâng cao trải nghiệm mua sắm cho khách hàng thông qua việc cung cấp những sản phẩm phù hợp và chất lượng cao, đồng thời thấu hiểu sở thích và nhu cầu của khách hàng thông qua các đánh giá và tương tác của họ.

    Để đáp ứng nhu cầu phát triển kinh doanh và tối ưu hóa trải nghiệm người dùng, HASAKI.VN muốn xây dựng một hệ thống gợi ý sản phẩm thông minh. Hệ thống này sẽ giúp khách hàng dễ dàng tìm thấy sản phẩm họ cần, từ đó thúc đẩy doanh thu và cải thiện sự hài lòng của khách hàng.

    ###### => Giải pháp: 
    Để giải quyết vấn đề này, chúng tôi sẽ áp dụng hai phương pháp thuật toán Machine Learning trong Python:
    - **Content-Based Filtering**: Dựa trên đặc điểm của sản phẩm, hệ thống sẽ gợi ý các sản phẩm tương tự với những sản phẩm mà khách hàng đã xem hoặc đánh giá cao.
    - **Collaborative Filtering**: Dựa trên hành vi mua sắm và đánh giá của các khách hàng tương tự, hệ thống sẽ dự đoán những sản phẩm mà khách hàng có thể thích, dựa trên những người dùng có sở thích giống nhau.

    Cả hai thuật toán này sẽ giúp hệ thống đưa ra những gợi ý chính xác và cá nhân hóa cho mỗi khách hàng, đồng thời giúp HASAKI.VN tối ưu hóa việc phân phối sản phẩm và tăng trưởng doanh thu.
""")  
    st.image("2.png")

elif choice == 'Hiển thị chart':
    st.subheader("Biểu đồ Heatmap")
    st.write("Lấy một phần nhỏ trong Cosine_sim, tương ứng với ma trận 18 x18. Gồm các giá trị liên quan đến 18 sản phẩm đầu tiên trong danh sách để trực quan hoá")
    st.image('heatmap.png', use_container_width=True)

elif choice == 'Build Project':
    st.subheader("Build Project")

    # Ở đây sẽ hiển thị phần nội dung của Build Project
    st.image('thuat_toan.jpg', use_container_width=True)  # Hiển thị ảnh banner
    st.write("### Recommendation System")
    st.write(""" 
    Hệ thống gợi ý tại Hasaki.vn được xây dựng dựa trên hai phương pháp chính:

    1. **Lọc nội dung (Content-Based Filtering)**: Gợi ý sản phẩm dựa trên phân tích đặc điểm và nội dung mô tả của sản phẩm, tập trung vào sự tương đồng giữa các sản phẩm.
    2. **Lọc cộng tác (Collaborative Filtering)**: Gợi ý sản phẩm dựa trên phân tích hành vi của người dùng, tận dụng dữ liệu từ các khách hàng có sở thích hoặc hành vi tương tự.
    """)

    # Tạo các tab cho các phương pháp
    tab1, tab2 = st.tabs(["Content-Based Filtering", "Collaborative Filtering"])

    with tab1:
        st.write("Bạn đã chọn **Content-Based Filtering**.")

    with tab2:
         st.subheader("Collaborative Filtering - KNNBaseline vs ALS")

# Nội dung mô tả
         st.write("""
        **1. Chuẩn bị dữ liệu:**
        - Đọc và làm sạch dữ liệu (loại bỏ trùng lặp, giữ lại đánh giá gần nhất).
        - Lưu dữ liệu đã xử lý vào file mới.

        **2. Khám phá dữ liệu:**
        - Phân tích phân phối số sao và số lượng đánh giá trên mỗi sản phẩm và khách hàng.

        **3. Chọn mô hình:**
        - So sánh các mô hình (KNNBaseline từ Surprise và ALS) dựa trên RMSE, thời gian huấn luyện và thời gian kiểm tra.

        **4. Huấn luyện mô hình:**
        - Sử dụng KNNBaseline từ Surprise để huấn luyện mô hình.

        **5. Đánh giá mô hình:**
        - Kiểm tra RMSE, MAE, thời gian huấn luyện và thời gian kiểm tra.
        """)

        # Kết quả so sánh các thuật toán (di chuyển mục này vào ngay sau mục 5)
         st.subheader("Kết quả so sánh các thuật toán")

        # Dữ liệu cho bảng chọn mô hình
         model_comparison_data = {
            "Algorithm": ["KNNBaseline", "SVD++", "KNNBasic", "SVD", "KNNWithZScore", "KNNWithMeans", "BaselineOnly", "SlopeOne", "CoClustering", "NMF"],
            "Mean RMSE": [0.721577, 0.727282, 0.747729, 0.777478, 0.796437, 0.797296, 0.808854, 0.835495, 0.849559, 0.861742],
            "Mean MAE": [0.319179, 0.413897, 0.318029, 0.465879, 0.337843, 0.337929, 0.504948, 0.366483, 0.387690, 0.569235],
            "Execution Time (s)": [7.486039, 4.008024, 5.710793, 2.270491, 8.275915, 6.494299, 0.568223, 0.843610, 4.083520, 4.811119]
        }

        # Chuyển dữ liệu thành DataFrame và hiển thị bảng
         model_comparison_df = pd.DataFrame(model_comparison_data)

         st.write(model_comparison_df)

        # Kết luận chọn KNNBaseline
         st.subheader("Kết luận chọn mô hình")
         st.write("""
        Dựa trên bảng trên, **KNNBaseline** có RMSE nhỏ nhất (0.721577), điều này cho thấy mô hình này có độ chính xác cao nhất trong các thuật toán được so sánh.
        """)

        # Mục 6: Lưu mô hình
         st.subheader("6. Lưu mô hình")
         st.write("""
        - Lưu mô hình đã huấn luyện vào file `product_surprise.pkl`.
        """)

        # Mục 7: Tiến hành deploy
         st.subheader("7. Tiến hành deploy")
         st.write("""
        - Dự đoán đánh giá cho các sản phẩm chưa được đánh giá và đề xuất các sản phẩm phù hợp.
        """)

        # So sánh mô hình
         st.subheader("So sánh mô hình")

        # Tạo bảng so sánh mô hình
         comparison_data = {
            "Tiêu chí": ["Hiệu suất (Accuracy)", "Thời gian huấn luyện", "Thời gian kiểm tra", "RMSE", "Time"],
            "Mục đích": [
                "Đánh giá độ chính xác của mô hình",
                "Thời gian để huấn luyện mô hình",
                "Thời gian để mô hình thực hiện dự đoán",
                "Đánh giá độ lệch dự đoán",
                "Tổng thời gian huấn luyện và kiểm tra"
            ],
            "KNNBaseline (Surprise)": [
                "RMSE: 0.7233",
                "0.67 giây",
                "0.52 - 0.70 giây",
                "0.7233",
                "Thời gian huấn luyện nhanh, kiểm tra chấp nhận được"
            ],
            "ALS (PySpark)": [
                "RMSE: 0.7381",
                "9.39 giây",
                "0.08 giây",
                "0.7381",
                "Thời gian huấn luyện lâu, kiểm tra nhanh"
            ]
        }

        # Chuyển dữ liệu thành DataFrame và hiển thị bảng
         comparison_df = pd.DataFrame(comparison_data)
         st.write(comparison_df)

        # Kết luận
         st.subheader("Kết luận")
         st.write("""
        Từ bảng trên, có thể thấy:
        - **KNNBaseline** có thời gian huấn luyện nhanh và RMSE thấp hơn, trong khi 
        - **ALS** có thời gian kiểm tra rất nhanh nhưng thời gian huấn luyện lâu hơn.
        """)



elif choice == 'Hệ thống gợi ý':
    st.header("Hệ thống gợi ý")

    # Tạo tab cho phương pháp gợi ý
    tab1, tab2 = st.tabs(["Content-Based Filtering", "Collaborative Filtering"])

    with tab1:
        st.subheader("🔍 Gợi ý sản phẩm tương tự")

        # List of products to select from
        product_options = [(row['ten_san_pham'], row['ma_san_pham']) for _, row in limited_products.iterrows()]
        selected_product = st.selectbox(
            "Chọn sản phẩm:",
            options=product_options,
            format_func=lambda x: x[0]
        )

        if selected_product:
            # Get selected product details
            ma_san_pham = selected_product[1]
            selected_product_row = df_products[df_products['ma_san_pham'] == ma_san_pham].iloc[0]
            gia_ban_formatted = f"{selected_product_row['gia_ban']:,.0f}".replace(",", ".")
            gia_goc_formatted = f"{selected_product_row['gia_goc']:,.0f}".replace(",", ".")
            
            # Display selected product
            st.write("### Bạn đã chọn:")  
            st.write(f"**Tên sản phẩm:** {selected_product_row['ten_san_pham']}")
            st.write(f"**Giá bán:** {gia_ban_formatted} VND")
            st.write(f"**Giá gốc:** {gia_goc_formatted} VND")
        
            # Hiển thị hình ảnh sản phẩm
            product_image = df_hinh[df_hinh['ma_san_pham'] == selected_product_row['ma_san_pham']]['hinh_anh'].values

            if len(product_image) > 0 and pd.notna(product_image[0]):
                image_path = product_image[0]
                
                # Kiểm tra định dạng ảnh
                if image_path.lower().endswith(('png', 'jpg', 'jpeg')):  
                    st.image(image_path, use_container_width=True)  # Hiển thị ảnh
                else:
                    st.write("**Định dạng ảnh không hợp lệ.**")
            else:
                st.write("**Không có ảnh sản phẩm**")

        # Get recommendations for the selected product
        recommendations = get_recommendations(df_products, ma_san_pham, cosine_sim_new, df_hinh, nums=5)
        
        if not recommendations.empty:
            st.write("### Các sản phẩm tương tự:")
            display_recommended_products(recommendations, cols=3)  # Hiển thị các sản phẩm tương tự với hình ảnh
        else:
            st.write("Không tìm thấy sản phẩm tương tự.")

    with tab2:
        st.subheader("👤 Gợi ý sản phẩm theo mã khách hàng")

        # Nhập thông tin khách hàng
        with st.expander("Nhập thông tin khách hàng"):
            input_customer_name = st.text_input("Nhập tên khách hàng:")
            input_customer_id = st.text_input("Nhập mã khách hàng:")

        # Danh sách khách hàng để chọn
        customer_options = [(row['ho_ten'], row['userId']) for _, row in limited_customers.iterrows()]

        # Kiểm tra nếu người dùng đã nhập thông tin
        if not input_customer_name and not input_customer_id:
            st.write("### Hoặc chọn khách hàng từ danh sách:")
            selected_customer = st.selectbox(
                "Chọn khách hàng:",
                options=customer_options,
                format_func=lambda x: x[0]
            )
        else:
            selected_customer = None  # Ẩn lựa chọn khách hàng nếu đã nhập thông tin

        # Kiểm tra thông tin khách hàng nhập vào
        user_id = None
        if input_customer_name:
            matched_customer = [cust for cust in customer_options if cust[0] == input_customer_name]
            if matched_customer:
                user_id = matched_customer[0][1]
                customer_name = matched_customer[0][0]
            else:
                st.error("Tên khách hàng không có trong danh sách.")
        elif input_customer_id:
            matched_customer = [cust for cust in customer_options if cust[1] == int(input_customer_id)]
            if matched_customer:
                user_id = matched_customer[0][1]
                customer_name = matched_customer[0][0]
            else:
                st.error("Mã khách hàng không có trong danh sách.")
        elif selected_customer:
            user_id = selected_customer[1]
            customer_name = selected_customer[0]

        if user_id:
            st.write(f"### Khách hàng đã chọn: {customer_name}")
            recommendations = recommendation_by_user(user_id, df, df_products, algorithm_loaded)
            display_recommended_products(recommendations)
