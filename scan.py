import streamlit as st
from PIL import Image
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import matplotlib.cm as cm
import json

# st.logo('Logo_wide.png')
# st.set_page_config(page_icon='icon.png', page_title='Chest Xray Scan')

col1, col2 = st.columns([1,4], vertical_alignment='center')
with col1:
    st.image('Logo.png')
# st.title()
with col2:
    st.title("Chest Xray Diagnosis")
st.divider()

def my_load_model(url):
    model = tf.lite.Interpreter(model_path=url)
    model.allocate_tensors()
    return model

if 'info' not in st.session_state:
    with open('diseases_info.json', 'r', encoding="utf-8") as f:
        st.session_state.info = json.load(f)


if 'model1' not in st.session_state:
    with st.spinner("Please waite, Loading AI models..."):
        # identify image type 
        st.session_state.model1 = load_model("models/model_1_identify_image.keras")
        # identify medical imaging type
        st.session_state.model2 = load_model("models/model_2_imaging_type.keras")
        # xray anatomical recognition
        st.session_state.model3 = load_model("models/model_3_anatomical_recognition.keras")
        # xray chest diseases diagnosis
        st.session_state.model4 = load_model("models/model_4_chest_xray_diagnosis.keras")
        # CT anatomical recognition
        st.session_state.model5 = load_model("models/model_5_ct_body_part.keras")
        # CT chest diseases diagnosis
        st.session_state.model6 = load_model("models/model_6_ct_scan_chest_diagnosis_model.keras")

import cv2
def preprocessing_func(img):
    """Apply preprocessing to the images before prediction (da lel model bta3 al chest xray bs ba2et almodels mesh metdaraba 3la preprocessed imaages) """
    img_uint8 = img.astype(np.uint8)
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl_img = clahe.apply(gray)
    rgb_img = cv2.cvtColor(cl_img, cv2.COLOR_GRAY2RGB)    
    return rgb_img.astype(np.float32) / 255.0

def predict(img):
    """Predict the chest xray"""
    # img = cv2.imread(img_path)
    img = img_to_array(img)
    img_resized = cv2.resize(img, (224, 224))
    preprocessed_img = preprocessing_func(img_resized)
    input_tensor = np.expand_dims(preprocessed_img, axis=0)
    prediction = st.session_state.model4.predict(input_tensor)
    return prediction

classes1 = {0:"Food",
           1:"Medical Imaging Scan",
           2:"Other Unsupported Image!",
           3:"Medical Test Reports"}
classes2 = {0 : 'CT',
           1 : 'MRI',
           2 : 'OCT',
           3 : 'Xray'}
classes3 = {0:'Chest x-ray',
            1:'Feet x-ray',
            2:'Hand x-ray',
            3:'Nick x-ray',
            4:'Other:unsuported x-ray type!',
            5:'Skull x-ray'}
classes4 = ["Covid", "Normal", "Pneumonia"]

def predict_img(img, model_number):
    """Predict all other images except the chest xrays"""
    img_normal = img_to_array(img)
    img_normal = img_normal/255
    img_normal = img_normal.reshape(1,224,224,3)
    if model_number==1:
        p = st.session_state.model1.predict(img_normal)
    elif model_number==2:
        p = st.session_state.model2.predict(img_normal)
    elif model_number==3:
        p = st.session_state.model3.predict(img_normal)
    elif model_number==4:
        p = predict(img)
    elif model_number==5:
        p = st.session_state.model5.predict(img_normal)
    elif model_number==6:
        p = st.session_state.model6.predict(img_normal)
    return p

def get_grad_cam_elements(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D) or 'conv' in layer.name.lower():
            grad_model = tf.keras.models.Model(inputs=[model.inputs], outputs=[layer.output, model.output])
            return grad_model, None
    
    for layer in reversed(model.layers):
        if hasattr(layer, 'layers'):
            sub_model = layer
            for sub_layer in reversed(sub_model.layers):
                if isinstance(sub_layer, tf.keras.layers.Conv2D) or 'conv' in sub_layer.name.lower():
                    sub_grad_model = tf.keras.models.Model(inputs=[sub_model.inputs], outputs=[sub_layer.output, sub_model.output])
                    return sub_grad_model, sub_model
                    
    return None, None

def generate_gradcam(img_array, model, class_idx):
    grad_elements, sub_model = get_grad_cam_elements(model)
    
    if grad_elements is None:
        raise ValueError("There is no Conv2d")
        
    with tf.GradientTape() as tape:
        if sub_model is None:
            conv_outputs, predictions = grad_elements(img_array)
        else:
            x = img_array
            conv_outputs = None
            for layer in model.layers:
                if isinstance(layer, tf.keras.layers.InputLayer):
                    continue
                if layer == sub_model:
                    conv_outputs, x = grad_elements(x)
                else:
                    x = layer(x)
            predictions = x
            
        loss = predictions[:, class_idx]
        
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

def overlay_heatmap(heatmap, original_img, alpha=0.4):
    img_array = np.array(original_img)
    
    heatmap_255 = np.uint8(255 * heatmap)
    jet_colors = cm.jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_255]
    
    jet_heatmap = np.uint8(255 * jet_heatmap)
    jet_heatmap_img = Image.fromarray(jet_heatmap)
    jet_heatmap_img = jet_heatmap_img.resize(original_img.size, Image.BICUBIC)
    
    jet_heatmap_array = np.array(jet_heatmap_img)
    
    superimposed_img = jet_heatmap_array * alpha + img_array
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    return Image.fromarray(superimposed_img)

def scan(img):
    p = predict_img(img, 1).argmax()
    # Medical Imaging
    if p==1:
        st.info("Medical Imaging Scan Detected!")
        p = predict_img(img, 2).argmax()
        # Xray
        if p==3:
            st.info("X-ray Detected!")
            p = predict_img(img, 3).argmax()
            # Chest Xray
            if p == 0:
                st.info("Chest X-ray Detected!")
                p = predict_img(img, 4)
                prob = p.max()
                class_idx = p.argmax()
                diagnosis = classes4[class_idx]
                # Covid & Pneumonia
                if class_idx!=1:
                    img_normal = img_to_array(img) / 255.0
                    img_normal = img_normal.reshape(1, 224, 224, 3)
                    try:
                        heatmap = generate_gradcam(img_normal, st.session_state.model4, class_idx)
                        if heatmap is not None:
                            img_hm = overlay_heatmap(heatmap, img)
                        else:
                            img_hm = img
                            st.error(e)
                    except Exception as e:
                        img_hm = img
                        st.error(e)
                        st.exception(e)
                    st.info(f"""{diagnosis}, {round(prob*100)} %""")
                    return diagnosis, f"{round(prob*100)} %", img_hm
                else:
                    st.info(f"""{diagnosis}, {round(prob*100)} %""")
                    return diagnosis, f"{round(prob*100)} %", None
            else:
                st.error("Error: These X-ray image is not on chest!")
                return None, None, None
        # CT Scan
        elif p==0:
            st.info("CT Scan Detected!")
            p = predict_img(img,5)
            p = int((p>0.5).astype(int))
            # Brain CT
            if p==0:
                st.info("Brain CT Scan Detected!")
                st.error("Error: These CT Scan image is not on chest!")
                return None, None, None
            # Chest CT
            elif p==1:
                st.info("Chest CT Scan Detected!")
                p = predict_img(img,6)
                prob = p.max()
                class_idx = p.argmax()
                diagnosis = classes4[class_idx]
                # Covid & Pneumonia
                if class_idx!=1:
                    img_normal = img_to_array(img) / 255.0
                    img_normal = img_normal.reshape(1, 224, 224, 3)
                    try:
                        heatmap = generate_gradcam(img_normal, st.session_state.model6, class_idx)
                        if heatmap is not None:
                            img_hm = overlay_heatmap(heatmap, img)
                        else:
                            img_hm = img
                            st.error(e)
                    except Exception as e:
                        img_hm = img
                        st.error(e)
                        st.exception(e)
                    st.info(f"""{diagnosis}, {round(prob*100)} %""")
                    return diagnosis, f"{round(prob*100)} %", img_hm
                # Normal
                else:
                    st.info(f"""{diagnosis}, {round(prob*100)} %""")
                    return diagnosis, f"{round(prob*100)} %", None
                    # st.error("Error: These Medial Imaging image is CT Scan, not X-ray!")
                    # return None, None, None
        # MRI
        elif p==1:
            st.info("MRI Scan Detected Scan Detected!")
            st.error("Error: These Medial Imaging image is MRI Scan, not X-ray or CT!")
            return None, None, None
        # OCT
        elif p==2:
            st.info("OCT Scan Detected Scan Detected!")
            st.error("Error: These Medial Imaging image is OCT Scan, not X-ray or CT!")
            return None, None, None
    # Food, Medical Report, Other
    else:
        # st.info("MRI Scan Detected Scan Detected!")
        st.error("Error: these image is not an Medical Imaging Scan image!")
        return None, None, None

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png",'webp'])

if uploaded_file is not None:
    # Saving The image
    img_type = uploaded_file.type.split('/')[-1]
    img_path = rf'inputs/input_image.{img_type}'
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    img = load_img(rf'inputs/input_image.{img_type}', target_size=(224,224,3))
    
    with st.spinner("Scanning, Please wait..."):
        disease, prob, img_hm = scan(img)
        if img_hm:
            col1, col2 = st.columns(2)
            with col1:
                st.image(img, caption='Uploaded Image.', use_container_width=True)
            with col2:
                st.image(img_hm, caption='Image Heatmap', use_container_width=True)
            st.subheader(f"Diagnosis: :blue[{disease}], with probability :blue[{prob}]", width='stretch', text_alignment='center')
            st.divider()
            st.markdown(st.session_state.info[disease], text_alignment='justify')
            st.divider()
        else:
            st.columns([1,2,1])[1].image(img, caption='Uploaded Image.', use_container_width=True)
            # st.write(f"Diagnosis: :blue[{disease}], with probability :blue[{prob}]")
            st.subheader(f"Diagnosis: :blue[{disease}], with probability :blue[{prob}]", width='stretch', text_alignment='center')
            st.divider()
            

