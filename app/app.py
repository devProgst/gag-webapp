"""Streamlit web app"""

import cv2
import numpy as np
import streamlit as st
from gag_python.gag import GAGModel

st.set_option("deprecation.showfileUploaderEncoding", False)


def main():
    gag = GAGModel()

    st.title("We get you age and gender!")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        st.image(image, caption="Before", use_column_width=True, channels="BGR")
        st.write("")
        st.write("Detecting faces...")

        outputJson = gag.get_age_gender(image, jsonResult=True)
        if len(outputJson) > 0:
            print(outputJson)
            outputImg = gag.get_age_gender(image)
            st.image(outputImg, caption="After", use_column_width=True, channels="BGR")
        else:
            st.write("No faces detected")


if __name__ == "__main__":
    main()
