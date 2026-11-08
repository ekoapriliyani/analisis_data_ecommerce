import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from pathlib import Path

sns.set_theme(style="whitegrid")
st.set_page_config(page_title="Dashboard E-Commerce Olist", page_icon="🛒", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_csv(Path(__file__).parent / "main_data.csv", parse_dates=["order_purchase_timestamp"])
    return df


df = load_data()

# ---------------- Sidebar ----------------
st.sidebar.header("Filter Data")
min_d, max_d = df["order_purchase_timestamp"].dt.date.min(), df["order_purchase_timestamp"].dt.date.max()
date_range = st.sidebar.date_input("Rentang tanggal pesanan", value=(min_d, max_d), min_value=min_d, max_value=max_d)
states = sorted(df["customer_state"].unique())
sel_states = st.sidebar.multiselect("State pelanggan", states, default=states)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = date_range
else:
    start, end = min_d, max_d

f = df[(df["order_purchase_timestamp"].dt.date >= start) & (df["order_purchase_timestamp"].dt.date <= end)
       & (df["customer_state"].isin(sel_states))]

st.title("🛒 Dashboard Analisis E-Commerce (Olist)")
st.caption("Pesanan berstatus *delivered*, Januari 2017 – Agustus 2018")

if f.empty:
    st.warning("Tidak ada data untuk filter yang dipilih.")
    st.stop()

orders = f.drop_duplicates("order_id")

# ---------------- KPI ----------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Pesanan", f"{orders['order_id'].nunique():,}")
c2.metric("Total Revenue", f"R$ {f['price'].sum():,.0f}")
c3.metric("Rata-rata Review", f"{orders['review_score'].mean():.2f}")
c4.metric("Pesanan Terlambat", f"{orders['is_late'].mean() * 100:.1f}%")

tab1, tab2, tab3 = st.tabs(["Revenue per Kategori", "Keterlambatan & Review", "Segmentasi RFM"])

# ---------------- Tab 1 ----------------
with tab1:
    st.subheader("Kategori produk apa yang menghasilkan revenue tertinggi dan terendah?")
    cat = f.groupby("category")["price"].sum().sort_values(ascending=False)
    n = st.slider("Jumlah kategori ditampilkan", 3, 10, 5)

    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(x=cat.head(n).values, y=cat.head(n).index, ax=ax, color="#1f77b4")
        ax.set_title(f"{n} Kategori Revenue Tertinggi"); ax.set_xlabel("Revenue (BRL)"); ax.set_ylabel("")
        st.pyplot(fig)
    with col_b:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(x=cat.tail(n).values, y=cat.tail(n).index, ax=ax, color="#d62728")
        ax.set_title(f"{n} Kategori Revenue Terendah"); ax.set_xlabel("Revenue (BRL)"); ax.set_ylabel("")
        st.pyplot(fig)

    st.markdown("**Tren revenue bulanan (kategori teratas)**")
    top = cat.head(5).index
    monthly = (f[f["category"].isin(top)]
               .pivot_table(index="order_month", columns="category", values="price", aggfunc="sum").fillna(0))
    fig, ax = plt.subplots(figsize=(12, 4))
    monthly.plot(ax=ax, marker="o")
    ax.set_xticks(range(len(monthly))); ax.set_xticklabels(monthly.index, rotation=45)
    ax.set_xlabel("Bulan"); ax.set_ylabel("Revenue (BRL)")
    st.pyplot(fig)

# ---------------- Tab 2 ----------------
with tab2:
    st.subheader("Seberapa besar keterlambatan pengiriman menurunkan review score?")
    o = orders.dropna(subset=["review_score"]).copy()
    o["Kelompok"] = pd.cut(o["delay_days"], bins=[-999, 0, 3, 7, 999],
                           labels=["Tepat waktu", "Telat 1-3 hari", "Telat 4-7 hari", "Telat >7 hari"])
    col_a, col_b = st.columns(2)
    with col_a:
        r = o.groupby("Kelompok", observed=True)["review_score"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(x="Kelompok", y="review_score", data=r, ax=ax,
                    palette=["#2ca02c", "#ffbb78", "#ff7f0e", "#d62728"], hue="Kelompok", legend=False)
        for i, v in enumerate(r["review_score"]):
            ax.text(i, v + 0.05, f"{v:.2f}", ha="center")
        ax.set_ylim(0, 5); ax.set_xlabel(""); ax.set_ylabel("Rata-rata Review Score")
        st.pyplot(fig)
    with col_b:
        min_orders = st.slider("Minimal jumlah pesanan per state", 50, 1000, 500, step=50)
        s = (orders.groupby("customer_state").agg(n=("order_id", "count"), late=("is_late", "mean"))
             .query("n >= @min_orders").sort_values("late", ascending=False).head(10))
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(x=s["late"] * 100, y=s.index, ax=ax, color="#d62728")
        ax.set_xlabel("% Pesanan Terlambat"); ax.set_ylabel("State")
        ax.set_title("10 State dengan Keterlambatan Tertinggi")
        st.pyplot(fig)

# ---------------- Tab 3 ----------------
with tab3:
    st.subheader("Segmentasi Pelanggan (RFM)")
    snap = f["order_purchase_timestamp"].max() + pd.Timedelta(days=1)
    rfm = (f.groupby("customer_unique_id")
           .agg(last=("order_purchase_timestamp", "max"), frequency=("order_id", "nunique"), monetary=("price", "sum"))
           .reset_index())
    rfm["recency"] = (snap - rfm["last"]).dt.days
    if len(rfm) >= 8:
        rfm["r"] = pd.qcut(rfm["recency"], 4, labels=[4, 3, 2, 1], duplicates="drop").astype(int)
        rfm["m"] = pd.qcut(rfm["monetary"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)

        def seg(x):
            if x["frequency"] >= 2:
                return "Repeat Customer"
            if x["m"] >= 3:
                return "Big Spender - Aktif" if x["r"] >= 3 else "Big Spender - Berisiko Hilang"
            return "Pelanggan Baru Bernilai Kecil" if x["r"] >= 3 else "Pelanggan Tidak Aktif"

        rfm["segment"] = rfm.apply(seg, axis=1)
        t = rfm.groupby("segment").agg(pelanggan=("customer_unique_id", "count"), rata2_recency_hari=("recency", "mean"),
                                       rata2_monetary=("monetary", "mean")).sort_values("pelanggan", ascending=False)
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.barplot(x=t["pelanggan"], y=t.index, ax=ax, color="#1f77b4")
        ax.set_xlabel("Jumlah Pelanggan"); ax.set_ylabel("")
        st.pyplot(fig)
        st.dataframe(t.round(1))
    else:
        st.info("Data terlalu sedikit untuk segmentasi RFM pada filter ini.")
