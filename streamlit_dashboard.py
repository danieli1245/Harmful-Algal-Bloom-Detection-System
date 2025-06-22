import streamlit as st 
import datetime 
import requests

st.set_page_config(page_title= "HAB DETECTION SYSTEM",layout="centered")
st.title("Harmful Algal Bloom (HAB) Detection")
st.markdown("""
			Enter data manually to get a prediction from the backend API.            
			""")
st.markdown("---")
tabs = st.tabs(["Enter your details", "Pick your location on map"])
# API Endpoint URL
API_URL = f"{st.secrets.get("API_URL", "http://localhost:5000")}/predict"
with tabs[0]:
    with st.form("hab_form"):
         st.subheader("Fill the Details")
         region = st.selectbox("Region", ["Northeast","West","Midwest","South"], format_func=lambda x: x.lower())
         distance_to_water = st.number_input("Distance to Water (in meters)", min_value=0.0, step=1.0)
         lat = st.number_input("Latitude", format="%.6f")
         lon = st.number_input("Longitude", format="%.6f")

         submit = st.form_submit_button("Submit")

         payload = {
        "region": region,
        "longitude": lon,
        "latitude": lat,
        "distance_to_water_m": distance_to_water
         }
    if submit:
        if not region and not lat and not lon:
              st.error("Please fill in all the required fields.")
        else:
             try:
                  with st.spinner('Asking the model for a prediction...'):
                       response = requests.post(API_URL, json=payload)
                       response.raise_for_status()  # Raise an exception for bad status codes
                
                       prediction = response.json()
                       print(prediction)
                     # prediction = {'is_harmful': 1, 'prediction': 'Toxic'}

                       predicted_value = prediction.get("predicted_value")
                       if predicted_value:
                           st.error(f"**Status:** {prediction['predicted_label'].capitalize()}")
                       else:
                           st.success(f"**Status:** {prediction['predicted_label'].capitalize()}")
                
                       st.metric(label="Confidence", value=f"{prediction['confidence_scores'][str(predicted_value)]*100}%")
                
                       with st.expander("Show Raw API Response"):
                           st.json(prediction)
        
             except requests.exceptions.RequestException as e:
                st.error(f"**API Error:** Could not connect to the backend service. Please ensure the Docker container is running.")
                st.error(f"Details: {e}")
             except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
#adding map logic tab
with tabs[1]:
    from streamlit_folium import st_folium
    import folium
    st.subheader("Select a location on the map")
    st.write("Click on the map to pick a location. Latitude and Longitute will be shown below.")
    #centering on gulf of mexico
    gulf_map = folium.Map(location=[25.0, -90.0], zoom_start=5)
    #adding the popup that will show lat and lon when user clicks it 
    gulf_map.add_child(folium.LatLngPopup())
    #rendering map in streamlit and capturing the user click area 
    map_result = st_folium(gulf_map, width=700, height=500)
    #showing the clicked coordinates 
    if map_result and map_result.get("last_clicked"):
        lat = map_result["last_clicked"]["lat"]
        lon = map_result["last_clicked"]["lng"]
        st.success(f"Selected Latitude: {lat:.6f}")
        st.success(f"Selected Longitude: {lon:.6f}")
