import streamlit as st

st.logo('Logo_wide.png')
st.set_page_config(page_icon='icon.png', page_title='Chest Xray Scan')


scan = st.Page(r"scan.py", title="X-ray Scan", icon=":material/scan:")
model_card = st.Page(r"model_card.py", title="Model Card", icon=":material/analytics:")

pages = {
    "": [scan, model_card]
}

pg = st.navigation(pages)
pg.run()