import streamlit as st
import pandas as pd
import pulp
from io import StringIO

st.set_page_config(page_title="DEA CCR - Group 1", layout="centered")

st.markdown("""
<h1 style='text-align: center; 
            color: #00BFFF; 
            font-weight: bold; 
            font-size: 42px;
            text-shadow: 1px 1px 2px black;'>
    DEA CCR - GROUP 1
</h1>
""", unsafe_allow_html=True)

col1, col2 = st.columns([2, 3.5])

# ======= CỘT TRÁI=======
with col1:
    st.markdown("<h3 style='color:#1E90FF;'>1. Hướng dẫn</h3>", unsafe_allow_html=True)
    st.markdown("""
    - <b>Copy dữ liệu từ Excel</b> và dán vào ô bên phải.<br>
    - 🏷 <b>Tên cột phải có dạng:</b><br>
        - <code>DMU</code> – để định danh đơn vị<br>
        - <code>input:&lt;tên&gt;</code> – cho các cột input<br>
        - <code>output:&lt;tên&gt;</code> – cho các cột output<br>
    - 🟢 <b>Nhấn nút RUN</b> để phân tích.
    """, unsafe_allow_html=True)
 
    run_clicked = st.button("RUN", use_container_width=True, type="primary")

# ======= CỘT PHẢI =======
with col2:
    st.markdown("<h3 style='color:#43A047;'>2. Dán dữ liệu từ Excel</h3>", unsafe_allow_html=True)
    pasted_data = st.text_area("Dán dữ liệu tại đây (bao gồm tiêu đề):", height=400, placeholder="DMU\tinput:...\toutput:...")

with st.expander("Xem ví dụ dữ liệu chuẩn", expanded=False):
    st.code("""
DMU	input:labor	input:capital	output:product1
A	4	3	1
B	7	3	1
C	8	1	1
D	4	2	1
E	2	4	1
F	10	1	1
""", language="text")

# ======= Process =======
if pasted_data and run_clicked:
    try:
        df = pd.read_csv(StringIO(pasted_data), sep=r'\t|,|;', engine='python')

        dmu_col = [col for col in df.columns if col.lower() == 'dmu']
        input_cols = [col for col in df.columns if col.lower().startswith("input:")]
        output_cols = [col for col in df.columns if col.lower().startswith("output:")]

        if not dmu_col or not input_cols or not output_cols:
            st.error(" Dữ liệu cần có: 1 cột 'DMU', ít nhất 1 'input:...' và 1 'output:...'.")
        else:
            dmu_col = dmu_col[0]
            dmu_names = df[dmu_col].astype(str).tolist()
            inputs = df[input_cols].values.tolist()
            outputs = df[output_cols].values.tolist()
            n_inputs = len(input_cols)
            n_outputs = len(output_cols)

            results = []

            for k in range(len(dmu_names)):
                prob = pulp.LpProblem(f"DEA_CCR_{dmu_names[k]}", pulp.LpMinimize)
                theta = pulp.LpVariable("theta", lowBound=0)
                lambdas = [pulp.LpVariable(f"lambda_{j}", lowBound=0) for j in range(len(dmu_names))]

                prob += theta

                for i in range(n_inputs):
                    prob += pulp.lpSum([lambdas[j] * inputs[j][i] for j in range(len(dmu_names))]) <= theta * inputs[k][i]
                for r in range(n_outputs):
                    prob += pulp.lpSum([lambdas[j] * outputs[j][r] for j in range(len(dmu_names))]) >= outputs[k][r]

                prob.solve(pulp.PULP_CBC_CMD(msg=0))

                if pulp.LpStatus[prob.status] != "Optimal":
                    st.warning(f" Không tìm được nghiệm tối ưu cho {dmu_names[k]}")
                    continue

                theta_star = theta.value()
                ref_set = [dmu_names[j] for j in range(len(dmu_names)) if lambdas[j].value() > 1e-6]

                results.append({
                    'DMU': dmu_names[k],
                    'θ* (Efficiency)': round(theta_star, 4),
                    'Reference Set': ', '.join(ref_set)
                })

            df_result = pd.DataFrame(results).reset_index(drop=True)
            st.markdown("<h3 style='color:#00ACC1;'>3. Kết quả DEA CCR</h3>", unsafe_allow_html=True)

            st.dataframe(df_result, use_container_width=True)

            # Bar Chart
            chart_data = df_result[['DMU', 'θ* (Efficiency)']].set_index('DMU')
            st.markdown("<h3 style='color:#F57C00;'>4. Biểu đồ hiệu quả kỹ thuật theo DMU</h3>", unsafe_allow_html=True)


            st.bar_chart(chart_data)

            # Đownload CSV
            csv = df_result.to_csv(index=False).encode('utf-8')
            st.download_button("💾 Tải kết quả CSV", csv, "dea_ccr_result.csv", "text/csv")

    except Exception as e:
        st.error(f"Lỗi xử lý dữ liệu: {e}")
