import streamlit as st
import pickle as pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import base64


st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon=":female-doctor:",
    layout="wide",
    initial_sidebar_state="expanded",
)








#CSS **************************************************************************************





# Kendi yerel dosyanızın yolu
image_path = "data/img2.jpg"

# Resmi base64 formatına dönüştürme
def get_base64_of_bin_file(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Base64 formatında resim elde edin
base64_image = get_base64_of_bin_file(image_path)


# Arka planı ayarlama
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("data:image/jpeg;base64,{base64_image}");
background-size: cover;
background-position: center center;
background-repeat: no-repeat;
background-attachment: local;
}}
[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
[data-testid="stSidebar"]{{
    background-color: #e3e3e3;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)





















#MAIN CODE
def main():

    input_data = add_sidebar()

    with st.container():
        st.title("Göğüs Kanseri Durum Belirleyici")
        st.markdown(
            """
            <p style = 'font-size:18px; font-weight:bold;'>
            Doku örneğinizden meme kanseri tanısı koymanıza yardımcı olması için lütfen bu uygulamayı sitoloji laboratuvarınıza bağlayın. 
            Bu uygulama, sitoz laboratuvarınızdan aldığı ölçümlere dayanarak bir makine öğrenimi modeli kullanarak bir meme kitlesinin iyi huylu mu yoksa kötü huylu mu olduğunu tahmin eder. Ayrıca ölçümleri kenar çubuğundaki kaydırıcıları kullanarak elle güncelleyebilirsiniz.
            </p>
            
            """,
            unsafe_allow_html=True
        )

    col1, col2 = st.columns([4, 1])

    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
    with col2:
        add_predictions(input_data)




def get_clean_data():
    data = pd.read_csv('data/data.csv')
    data.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data


def add_sidebar():
    st.sidebar.header("Çekirdek Ölçüm Değerleri")

    data = get_clean_data()

    slider_labels = [
        ("Yarıçap (ortalama)", "radius_mean"),
        ("Doku (ortalama)", "texture_mean"),
        ("Çevre Uzunluğu (ortalama)", "perimeter_mean"),
        ("Alan (ortalama)", "area_mean"),
        ("Pürüzsüzlük (ortalama)", "smoothness_mean"),
        ("Kompaklık Seviyesi (ortalama)", "compactness_mean"),
        ("İçbükeylik Seviyesi (ortalama)", "concavity_mean"),
        ("İçbükey Noktalar (ortalama)", "concave points_mean"),
        ("Simetri (ortalama)", "symmetry_mean"),
        ("Fraktal Boyut (ortalama)", "fractal_dimension_mean"),
        ("Yarıçap (Standart)", "radius_se"),
        ("Doku (Standart)", "texture_se"),
        ("Çevre Uzunluğu (Standart)", "perimeter_se"),
        ("Alan (Standart)", "area_se"),
        ("Pürüzsüzlük (Standart)", "smoothness_se"),
        ("Kompaklık Seviyesi (Standart)", "compactness_se"),
        ("İçbükeylik Seviyesi (Standart)", "concavity_se"),
        ("İçbükey Noktalar (Standart)", "concave points_se"),
        ("Simetr (Standart)", "symmetry_se"),
        ("Fraktal Boyut (Standart)", "fractal_dimension_se"),
        ("Yarıçap (En Kötü Değer)", "radius_worst"),
        ("Doku (En Kötü Değer)", "texture_worst"),
        ("Çevre Uzunluğu (En Kötü Değer)", "perimeter_worst"),
        ("Alan (En Kötü Değer)", "area_worst"),
        ("Pürüzsüzlük (En Kötü Değer)", "smoothness_worst"),
        ("Kompaklık Seviyesi (En Kötü Değer)", "compactness_worst"),
        ("İçbükeylik Seviyesi (En Kötü Değer)", "concavity_worst"),
        ("İçbükey Noktalar (En Kötü Değer)", "concave points_worst"),
        ("Simetri (En Kötü Değer)", "symmetry_worst"),
        ("Fraktal Boyut (En Kötü Değer)", "fractal_dimension_worst"),
    ]

    input_dict = {}

    for label, key in slider_labels:

        input_dict[key] = st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )

    return input_dict


def get_scaled_values(input_dict):
    data = get_clean_data()

    x = data.drop(['diagnosis'], axis=1)
    scaled_dict = {}

    for key, value in input_dict.items():
        max_val = x[key].max()
        min_val = x[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value

    return scaled_dict


def get_radar_chart(input_data):
    input_data = get_scaled_values(input_data)

    categories = ['Yarıçap', 'Doku', 'Çevre Uzunluğu', 'Alan',
                  'Pürüzsüzlük', 'Kompaklık',
                  'İçbükeylik', 'İçbükey Nok.',
                  'Simetri', 'Fraktal Boyut']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
            input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
            input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
            input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Ortalama Değer'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
            input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
            input_data['concave points_se'], input_data['symmetry_se'], input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standart Değer'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
            input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
            input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
            input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='En Kötü Değer'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont = dict(size=8, family = "Arial, sans-serif", color= "black", weight = "bold"),
            ),
            angularaxis = dict(
                tickfont = dict(size=18, family = "Arial, sans-serif",color = "black", weight ="bold")
            )
        ),
        showlegend=True,
        #ARKA PLANI SEFFAF YAPTIK
        plot_bgcolor = 'rgba(0, 0, 0, 0)',
        paper_bgcolor = 'rgba(0, 0, 0, 0)',
        legend = dict(
            font = dict(size = 20),
            itemsizing = 'constant'
        )



    )

    return fig



def add_predictions(input_data):
    model = pickle.load(open("model/model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))

    input_array = np.array(list(input_data.values())).reshape(1, -1)

    input_array_scaled = scaler.transform(input_array)

    prediction = model.predict(input_array_scaled)

    st.markdown("<h2 style = 'font-size:28px; font-weight:bold;'>Kanser Hücresi Durumu</h2> ", unsafe_allow_html=True)
    st.write("<p style='font-size:18px; font-weight:bold;'>Hücrenin Değerlere Göre Durumu ",unsafe_allow_html=True)

    if prediction[0] == 0:
        st.markdown("<p style='font-size:20px; font-weight:bold; color:green; background-color: #d6d6d6; "
                    "text-align:center; padding:5px ; border-radius:15px;width:fit-content; "
                    "margin:auto;margin-bottom:20px;'>İyi Huylu Hücre</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='font-size:20px; font-weight:bold; color:red; background-color: #cdb8b7; "
                    "text-align:center; padding:5px ;border-radius:15px;width:fit-content; "
                    "margin:auto;margin-bottom:20px;'>Kötü Huylu Hücre</p>", unsafe_allow_html=True)

    # st.write(prediction)

    st.markdown("<p style='font-size:18px; font-weight:bold;'>Verilen Değerlere göre İyi olma oranı: {:.2f}%</p>".format((model.predict_proba(input_array_scaled)[0][0])*100), unsafe_allow_html=True)
    st.markdown("<p style='font-size:18px; font-weight:bold;'>Verilen Değerlere göre Kötü olma oranı: {:.2f}%</p>".format(model.predict_proba(input_array_scaled)[0][1]*100), unsafe_allow_html=True)

    st.markdown(
        "<p style='font-size:18px; font-weight:bold;'>Bu uygulama tıp uzmanlarına teşhis koymada yardımcı olabilir ancak profesyonel teşhisin yerine kullanılmamalıdır.</p>",
        unsafe_allow_html=True)


if __name__ == '__main__':
    main()