import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras import layers, regularizers
import os
import requests

# =========================
# CONFIGURAÇÃO STREAMLIT
# =========================
st.set_page_config(
    page_title="Classificação de Câncer de Tireoide",
    layout="wide"
)

# =========================
# MODEL CONFIG
# =========================
MODEL_URL = st.secrets["MODELO"]
MODEL_PATH = "algoritmo.h5"

IMG_SIZE = (224, 224)
DISPLAY_WIDTH = 250

# =========================
# CUSTOM LAYERS
# =========================

class Avg2MaxPooling(layers.Layer):
    def __init__(self, pool_size=3, strides=2, padding="same", **kwargs):
        super().__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding

        self.avg_pool = layers.AveragePooling2D(
            pool_size=pool_size,
            strides=strides,
            padding=padding
        )

        self.max_pool = layers.MaxPooling2D(
            pool_size=pool_size,
            strides=strides,
            padding=padding
        )

        self.bn = layers.BatchNormalization()

    def call(self, inputs):
        x_avg = self.avg_pool(inputs)
        x_max = self.max_pool(inputs)
        x = x_avg - 2 * x_max
        return self.bn(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "pool_size": self.pool_size,
            "strides": self.strides,
            "padding": self.padding
        })
        return config


class SEBlock(layers.Layer):
    def __init__(self, ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channels = input_shape[-1]

        self.gap = layers.GlobalAveragePooling2D()
        self.fc1 = layers.Dense(max(channels // self.ratio, 1), activation="swish")
        self.fc2 = layers.Dense(channels, activation="sigmoid")
        self.reshape = layers.Reshape((1, 1, channels))

    def call(self, inputs):
        x = self.gap(inputs)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.reshape(x)
        return inputs * x


class DepthwiseSeparableConv(layers.Layer):
    def __init__(self, filters, kernel_size=3, strides=1, se_ratio=16, reg=0.001, **kwargs):
        super().__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.se_ratio = se_ratio
        self.reg = reg

        self.depthwise = layers.DepthwiseConv2D(
            kernel_size,
            strides=strides,
            padding="same",
            depthwise_regularizer=regularizers.l2(reg)
        )

        self.pointwise = layers.Conv2D(
            filters,
            1,
            padding="same",
            kernel_regularizer=regularizers.l2(reg)
        )

        self.bn = layers.BatchNormalization()
        self.se = SEBlock(se_ratio)

    def call(self, inputs):
        x = self.depthwise(inputs)
        x = self.pointwise(x)
        x = self.bn(x)
        x = tf.nn.swish(x)
        x = self.se(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "se_ratio": self.se_ratio,
            "reg": self.reg
        })
        return config


# =========================
# MODEL LOAD
# =========================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        r = requests.get(MODEL_URL)
        r.raise_for_status()

        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
        custom_objects={
            "Avg2MaxPooling": Avg2MaxPooling,
            "DepthwiseSeparableConv": DepthwiseSeparableConv
        }
    )

    return model


model = load_model()


# =========================
# PREPROCESSAMENTO
# =========================

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img = np.array(image).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def gerar_thumbnail(image, size=(DISPLAY_WIDTH, DISPLAY_WIDTH)):
    image = image.convert("RGB")

    target_w, target_h = size
    target_ratio = target_w / target_h

    w, h = image.size
    current_ratio = w / h

    if current_ratio > target_ratio:
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        box = (left, 0, left + new_w, h)
    else:
        new_h = int(w / target_ratio)
        top = (h - new_h) // 2
        box = (0, top, w, top + new_h)

    image_cropped = image.crop(box)
    return image_cropped.resize(size, Image.LANCZOS)


def classificar_imagem(model, image):
    img = preprocess_image(image)
    prediction = model.predict(img, verbose=0)[0][0]

    if prediction >= 0.5:
        return "Maligno", prediction, prediction
    else:
        return "Benigno", 1 - prediction, prediction


# =========================
# UI STREAMLIT
# =========================

st.title("Classificação de Câncer de Tireoide")

# sessão
if "imagens" not in st.session_state:
    st.session_state.imagens = []

opcao = st.radio("Escolha uma opção", ["Upload", "Câmera"], horizontal=True)

if opcao == "Upload":

    uploaded_files = st.file_uploader(
        "Selecione imagens",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.session_state.imagens = [Image.open(f) for f in uploaded_files]

else:

    captured = st.camera_input("Capture uma imagem")

    col1, col2 = st.columns(2)

    with col1:
        if captured and st.button("Adicionar foto"):
            st.session_state.imagens.append(Image.open(captured))

    with col2:
        if st.button("Limpar imagens"):
            st.session_state.imagens = []


# =========================
# RESULTADOS
# =========================

imagens = st.session_state.imagens

if imagens:

    st.write(f"**{len(imagens)} imagem(ns)**")

    cols = st.columns(4)
    resultados = []

    with st.spinner("Classificando..."):
        for img in imagens:
            resultados.append((img, *classificar_imagem(model, img)))

    for i, (img, classe, confianca, pred) in enumerate(resultados):

        with cols[i % 4]:

            st.image(gerar_thumbnail(img), caption=f"Imagem {i+1}")

            if classe == "Maligno":
                st.error(classe)
            else:
                st.success(classe)

            st.progress(float(confianca))
            st.write(f"Confiança: {confianca*100:.2f}%")

            with st.expander("Detalhes"):
                st.write(f"Maligno: {pred*100:.2f}%")
                st.write(f"Benigno: {(1-pred)*100:.2f}%")

else:
    st.info("Nenhuma imagem selecionada.")
