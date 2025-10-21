# app_no_group_border.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px

# -----------------------
# CONFIG - update if needed
# -----------------------
DB_USER = "root"
DB_PASS = ""
DB_HOST = "localhost"
DB_PORT = 3306
DB_NAME = "tenant_almasrypharmacy"

CONN_STR = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"

@st.cache_data(ttl=300)
def load_table(name):
    df = pd.read_csv(f"{name}.csv")
    if 'created_at' in df.columns:
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    return df

# -----------------------
# Helper: bordered metric card
# -----------------------
def bordered_metric(title, value, caption=None):
    st.markdown(
        f"""
        <div style="
            border: 2px solid white;
            border-radius: 10px;
            padding: 20px;
            height: 150px;  
            display: flex;
            flex-direction: column;
            justify-content: center;
            text-align: center;
            margin-bottom: 10px;
            background-color: transparent;
            transition: all 0.3s ease;
            overflow: hidden;
        "
        onmouseover="this.style.transform='scale(1.05)'; this.style.boxShadow='0 4px 20px rgba(255,255,255,0.3)';"
        onmouseout="this.style.transform='scale(1)'; this.style.boxShadow='none';"
        >
            <div style="font-size:14px; color:white; font-weight:500;">{title}</div>
            <div style="font-size:22px; font-weight:bold; margin-top:5px; color:white;">{value}</div>
            <div style="font-size:12px; color:white; margin-top:5px;">{caption if caption else ''}</div>
        </div>
        """,
        unsafe_allow_html=True
    )



# -----------------------
# Streamlit layout + filters
# -----------------------
st.set_page_config(layout="wide", page_title="2Loyal Analytics", page_icon="🎯")
st.title("🎯 2Loyal — Analytics Dashboard ")

# Sidebar Filters
st.sidebar.header("Filters")
today = datetime.utcnow().date()
default_start = today - timedelta(days=30)
start_date = st.sidebar.date_input("Start date", default_start)
end_date = st.sidebar.date_input("End date", today)
if start_date > end_date:
    st.sidebar.error("Start date must be <= End date")
st.sidebar.markdown("---")
st.sidebar.markdown("**Notes**\n- Date filters apply to transactions & invoices queries.\n")

start_ts = f"{start_date} 00:00:00"
end_ts = f"{end_date} 23:59:59"

# -----------------------
# Customer & Loyalty Overview
# -----------------------
st.markdown("## 👥 Customer & Loyalty Overview")

col1, col2, col3, col4 = st.columns(4)

# Total customers
customers = load_table("customers")
total_customers = len(customers)
with col1:
    bordered_metric("Total customers", f"{total_customers:,}", "All registered customers in the system.")

# Active customers (last 30 days)
active_start = (datetime.combine(end_date, datetime.min.time()) - timedelta(days=30))
active_end = datetime.combine(end_date, datetime.max.time())

transactions = load_table("transactions")
transactions['created_at'] = pd.to_datetime(transactions['created_at'])

active_customers = transactions[
    (transactions['created_at'] >= active_start) &
    (transactions['created_at'] <= active_end)
]['customer_id'].nunique()
with col2:
    bordered_metric("Active (last 30d)", f"{active_customers:,}", "Customers with ≥1 transaction in last 30 days.")

# New customers (selected range)
start_dt = pd.to_datetime(start_ts)
end_dt = pd.to_datetime(end_ts)
new_customers = customers[
    (pd.to_datetime(customers['created_at']) >= start_dt) &
    (pd.to_datetime(customers['created_at']) <= end_dt)
].shape[0]
with col3:
    bordered_metric("New customers (range)", f"{new_customers:,}", "New sign-ups in selected range.")

# Active programs
earning_rules = load_table("earning_rules")
active_programs = earning_rules[earning_rules['is_active'] == True].shape[0]
with col4:
    bordered_metric("Active programs", f"{active_programs}", "Number of active earning rules/programs.")

st.markdown("---")

# -----------------------
# Customer segmentation
# -----------------------
st.subheader("Customer Segmentation")
seg_col1, seg_col2 = st.columns([1,1])

# By group
customers = load_table("customers")
customer_group = load_table("customer_group")
groups = load_table("groups")

# دمج الجداول عشان نحسب عدد العملاء لكل group
df_seg_group = (
    customer_group.merge(customers, left_on='customer_id', right_on='id', how='left')
                  .merge(groups, left_on='group_id', right_on='id', how='left')
)
df_seg_group = df_seg_group[df_seg_group['group_name'].notna() & (df_seg_group['group_name'] != '')]
df_seg_group = df_seg_group.groupby('group_name').size().reset_index(name='customer_count')
df_seg_group = df_seg_group.sort_values('customer_count', ascending=False)

if df_seg_group.empty:
    seg_col1.info("No group segmentation data available.")
else:
    fig = px.pie(df_seg_group, names='group_name', values='customer_count', title="Customers by Group", hole=0.35)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    seg_col1.plotly_chart(fig, use_container_width=True)
    seg_col1.dataframe(df_seg_group, use_container_width=True)

# Activity segmentation
transactions = load_table("transactions")
transactions['created_at'] = pd.to_datetime(transactions['created_at'])
customers['created_at'] = pd.to_datetime(customers['created_at'])

def segment_customer(cust_row, trans_df, start_dt, end_dt):
    cust_id = cust_row['id']
    cust_created = cust_row['created_at']
    cust_trans = trans_df[trans_df['customer_id'] == cust_id]
    total_tx = len(cust_trans)
    last_tx_date = cust_trans['created_at'].max() if not cust_trans.empty else pd.Timestamp.min

    if start_dt <= cust_created <= end_dt:
        return 'New'
    elif total_tx >= 50:
        return 'VIP'
    elif last_tx_date < (end_dt - pd.Timedelta(days=90)):
        return 'Dormant'
    elif total_tx > 1:
        return 'Returning'
    else:
        return 'Other'

df_activity_seg = customers.copy()
df_activity_seg['customer_segment'] = df_activity_seg.apply(segment_customer, axis=1, trans_df=transactions, start_dt=start_dt, end_dt=end_dt)
df_activity_seg = df_activity_seg.groupby('customer_segment').size().reset_index(name='cnt')

if df_activity_seg.empty:
    seg_col2.info("No activity segmentation data.")
else:
    df_activity_seg['description'] = df_activity_seg['customer_segment'].map({
        'New': f"Customers who registered between {start_ts} and {end_ts}",
        'VIP': "Customers with ≥50 total transactions",
        'Dormant': f"Customers whose last transaction was more than 90 days before {end_ts}",
        'Returning': "Customers with more than 1 transaction",
        'Other': "All other customers"
    })

    fig2 = px.bar(
        df_activity_seg,
        x='customer_segment',
        y='cnt',
        text='cnt',
        color='customer_segment',
        hover_data={'description': True, 'cnt': True, 'customer_segment': False}  
    )

    fig2.update_traces(
        hovertemplate="<b>%{x}</b><br>Count: %{y}<br>%{customdata[0]}<extra></extra>"
    )

    seg_col2.plotly_chart(fig2, use_container_width=True)
    seg_col2.dataframe(df_activity_seg, use_container_width=True)

st.markdown("---")

# -----------------------
# Loyalty Program Performance
# -----------------------
st.subheader("Loyalty Program Performance")
k1, k2, k3, k4 = st.columns(4)

transactions = load_table("transactions")
transactions['created_at'] = pd.to_datetime(transactions['created_at'])

# تصفية الـ transactions حسب التاريخ
tx_range = transactions[(transactions['created_at'] >= pd.to_datetime(start_ts)) &
                        (transactions['created_at'] <= pd.to_datetime(end_ts))]

# Points issued
points_issued = tx_range[tx_range['balance_type']=='earn']['balance_change'].sum()
with k1:
    bordered_metric("Points Issued", f"{int(points_issued):,}", "Total loyalty points granted.")

# Points redeemed
points_redeemed = tx_range[tx_range['balance_type']=='spend']['balance_change'].sum()
with k2:
    bordered_metric("Points Redeemed", f"{int(points_redeemed):,}", "Points redeemed by customers.")

# Redemption rate
earned = tx_range[tx_range['balance_type']=='earn']['balance_change'].sum()
spent_or_loss = tx_range[tx_range['balance_type'].isin(['spend','loss'])]['balance_change'].sum()
redemption_pct = (spent_or_loss / earned * 100) if earned != 0 else 0
with k3:
    bordered_metric("Redemption Rate (%)", f"{redemption_pct:.2f}%", "Share of issued points redeemed.")

# Breakage rate
loss = tx_range[tx_range['balance_type']=='loss']['balance_change'].sum()
breakage_pct = (loss / earned * 100) if earned != 0 else 0
with k4:
    bordered_metric("Breakage Rate (%)", f"{breakage_pct:.2f}%", "Percent of issued points expired/lost.")

# Time series
st.markdown("### Points trend (daily)")
if tx_range.empty:
    st.info("No transactions in range to show points timeseries.")
else:
    df_points_ts = tx_range.copy()
    df_points_ts['date'] = df_points_ts['created_at'].dt.date
    df_points_ts = df_points_ts.groupby('date').agg(
        issued=('balance_change', lambda x: x[tx_range.loc[x.index, 'balance_type']=='earn'].sum()),
        redeemed=('balance_change', lambda x: x[tx_range.loc[x.index, 'balance_type']=='spend'].sum())
    ).reset_index()
    
    fig_pts = px.line(df_points_ts, x='date', y=['issued','redeemed'],
                      labels={'value':'Points','date':'Date','variable':'Series'},
                      title='Daily Points Issued vs Redeemed')
    st.plotly_chart(fig_pts, use_container_width=True)

st.markdown("---")


# -----------------------
# Monetary value of points & average points per customer
# -----------------------
st.subheader("Monetary Value & Points per Customer")

customers = load_table("customers")
transactions = load_table("transactions")
groups = load_table("groups")
customer_group = load_table("customer_group")

transactions['created_at'] = pd.to_datetime(transactions['created_at'])
customers['created_at'] = pd.to_datetime(customers['created_at'])

# تصفية transactions حسب الفترة
tx_range = transactions[(transactions['created_at'] >= pd.to_datetime(start_ts)) &
                        (transactions['created_at'] <= pd.to_datetime(end_ts)) &
                        (transactions['balance_type'] == 'earn')]

# Monetary value of points
merged = customers.merge(customer_group, left_on='id', right_on='customer_id', how='inner') \
                  .merge(groups, left_on='group_id', right_on='id', how='inner', suffixes=('_cust','_grp')) \
                  .merge(tx_range, left_on='id_cust', right_on='customer_id', how='inner')

mv_issued = (merged['number_of_points'] / merged['factor_of_profit_share']).sum()

# Avg points per customer
avg_points = tx_range.groupby('customer_id')['balance_change'].sum().mean()

colA, colB = st.columns(2)

with colA:
    bordered_metric(
        "Monetary value of points (issued)",
        f"{float(mv_issued):,.2f}",
        "Estimated cash-equivalent value of issued points (using group profit share factor)."
    )

with colB:
    bordered_metric(
        "Avg points / customer",
        f"{(float(avg_points) if avg_points else 0):.2f}",
        "Average points earned per customer in the date range."
    )

st.markdown("---")

# -----------------------
# Revenue Impact (KPIs + time series)
# -----------------------
st.subheader("Revenue Impact")
invoices = load_table("invoices")
invoices['created_at'] = pd.to_datetime(invoices['created_at'])
invoices['statusinvoice'] = invoices['statusinvoice'].str.lower()

r1, r2, r3 = st.columns(3)

# total revenue
total_rev = invoices[(invoices['statusinvoice'].isin(['paid','partly paid'])) &
                     ((invoices['is_deleted']==0) | (invoices['is_deleted'].isna())) &
                     (invoices['created_at'] >= pd.to_datetime(start_ts)) &
                     (invoices['created_at'] <= pd.to_datetime(end_ts))
                    ]['grand_total'].sum()

# revenue from loyalty vs non-loyalty
rev_loyal = invoices[((invoices['loyalty_point']>0) | (invoices['loyalty_amount']>0)) &
                     (invoices['created_at'] >= pd.to_datetime(start_ts)) &
                     (invoices['created_at'] <= pd.to_datetime(end_ts))
                    ]['grand_total'].sum()

rev_non = invoices[~((invoices['loyalty_point']>0) | (invoices['loyalty_amount']>0)) &
                   (invoices['created_at'] >= pd.to_datetime(start_ts)) &
                   (invoices['created_at'] <= pd.to_datetime(end_ts))
                  ]['grand_total'].sum()

with r1:
    bordered_metric("Total Revenue", f"{float(total_rev):,.2f} EGP", 
                    "Sum of invoice grand_total for paid/partly paid invoices in range.")
with r2:
    bordered_metric("Revenue from loyalty customers", f"{float(rev_loyal):,.2f} EGP", 
                    "Revenue where invoice indicates loyalty usage.")
with r3:
    bordered_metric("Revenue from non-loyalty customers", f"{float(rev_non):,.2f} EGP", 
                    "Revenue from other customers.")

# total cost of rewards
r4, r5, r6 = st.columns(3)
cost_rewards = invoices[(invoices['loyalty_amount']>0) &
                        ((invoices['is_deleted']==0) | (invoices['is_deleted'].isna())) &
                        (invoices['statusinvoice'].isin(['completed','paid'])) &
                        (invoices['created_at'] >= pd.to_datetime(start_ts)) &
                        (invoices['created_at'] <= pd.to_datetime(end_ts))
                       ]['loyalty_amount'].sum()

# net revenue
net_rev = invoices[((invoices['loyalty_point']>0) | (invoices['loyalty_amount']>0)) &
                   (invoices['created_at'] >= pd.to_datetime(start_ts)) &
                   (invoices['created_at'] <= pd.to_datetime(end_ts))
                  ]['grand_total'].sum() - cost_rewards

# ROI
roi_pct = (net_rev / cost_rewards * 100) if cost_rewards != 0 else None

with r4:
    bordered_metric("Total cost of rewards", f"{float(cost_rewards):,.2f} EGP", 
                    "Monetary cost of loyalty discounts.")
with r5:
    bordered_metric("Net revenue from loyalty", f"{float(net_rev):,.2f} EGP", 
                    "Revenue after loyalty discounts.")
with r6:
    bordered_metric("ROI (%)", f"{roi_pct:.2f}%" if roi_pct is not None else "N/A", 
                    "Return on investment from loyalty rewards.")

st.markdown("---")

# Revenue timeseries (daily)
st.markdown("### Revenue trend (daily)")
df_rev_ts = invoices[(invoices['created_at'] >= pd.to_datetime(start_ts)) &
                     (invoices['created_at'] <= pd.to_datetime(end_ts))
                    ].groupby(invoices['created_at'].dt.date)['grand_total'].sum().reset_index()
df_rev_ts.rename(columns={'created_at':'dt','grand_total':'revenue'}, inplace=True)

if df_rev_ts.empty:
    st.info("No invoice data for selected range.")
else:
    df_rev_ts['dt'] = pd.to_datetime(df_rev_ts['dt'])
    fig_rev = px.line(df_rev_ts, x='dt', y='revenue', title='Daily Revenue', labels={'revenue':'EGP','dt':'Date'})
    st.plotly_chart(fig_rev, use_container_width=True)

st.markdown("---")

# -----------------------
# Campaign Analytics
# -----------------------
st.subheader("📢 Campaign Analytics")

c1, c2 = st.columns(2)

# تحميل البيانات
whatsapp_history = load_table("whatsapp_history")
emails_history = load_table("emails_history")
sms_history = load_table("sms_history")

# تحويل created_at لتاريخ
for df in [whatsapp_history, emails_history, sms_history]:
    df['created_at'] = pd.to_datetime(df['created_at'])

# Total campaigns sent
total_wh = whatsapp_history[(whatsapp_history['created_at'] >= pd.to_datetime(start_ts)) &
                            (whatsapp_history['created_at'] <= pd.to_datetime(end_ts))].shape[0]

total_email = emails_history[(emails_history['created_at'] >= pd.to_datetime(start_ts)) &
                             (emails_history['created_at'] <= pd.to_datetime(end_ts))].shape[0]

total_sms = sms_history[(sms_history['created_at'] >= pd.to_datetime(start_ts)) &
                        (sms_history['created_at'] <= pd.to_datetime(end_ts))].shape[0]

total_campaigns = total_wh + total_email + total_sms

with c1:
    bordered_metric(
        "Total campaigns sent (all channels)",
        f"{int(total_campaigns):,}",
        "Count of messages/campaign entries across channels in selected date range."
    )

# WhatsApp open rate
wh_range = whatsapp_history[(whatsapp_history['created_at'] >= pd.to_datetime(start_ts)) &
                            (whatsapp_history['created_at'] <= pd.to_datetime(end_ts))]

delivered = wh_range[wh_range['status'] == 'DELIVERY_ACK'].shape[0]
read = wh_range[wh_range['status'] == 'READ'].shape[0]

wh_open_rate = (read / delivered * 100) if delivered != 0 else 0

with c2:
    bordered_metric(
        "WhatsApp open rate (%)",
        f"{wh_open_rate:.2f}%",
        "Reads divided by deliveries for WhatsApp messages in the period."
    )

st.markdown("---")


st.info("Dashboard built: KPIs + timeseries + segmentation + campaign metrics. Each KPI/chart includes a short caption explaining what it means.")
