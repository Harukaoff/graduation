import streamlit as st

st.title("ダミーアプリ: 挨拶アプリ")

name = st.text_input("名前を入力してね:")

if st.button("挨拶する"):
    if name:
        st.success(f"{name}さん、こんにちは！")
    else:
        st.warning("名前を入力してね！")
