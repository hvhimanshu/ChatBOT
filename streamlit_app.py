import streamlit as st
import requests
from PIL import Image
import io
from fpdf import FPDF
import datetime

# Configure page
st.set_page_config(page_title="Pneumonia Detector", page_icon="🩺")
st.title("🩺 AI-Powered Pneumonia Detection")

# API configuration
API_URL = "http://localhost:5000/predict"  # Your Render URL
# API_URL = "https://pneumoniachatbot.onrender.com/predict"  # Your Render URL
MAX_FILE_SIZE = 5  # MB

def generate_pdf_report(patient_data, analysis_result):
    """Generate PDF report with patient and analysis data"""
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Pneumonia Detection Report", 0, 1, 'C')
    pdf.ln(10)
    
    # Patient Information
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Patient Information", 0, 1)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Name: {patient_data['name']}", 0, 1)
    pdf.cell(0, 10, f"ID: {patient_data['id']}", 0, 1)
    pdf.cell(0, 10, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", 0, 1)
    pdf.ln(5)
    
    # Clinical Notes
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Clinical Notes:", 0, 1)
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 10, patient_data['notes'])
    pdf.ln(10)
    
    # Analysis Results
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Diagnosis Results", 0, 1)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Result: {analysis_result['result']}", 0, 1)
    pdf.cell(0, 10, f"Confidence: {analysis_result['confidence']*100:.2f}%", 0, 1)
    
    # Save to bytes for Streamlit
    # return pdf.output(dest='S').encode('latin1')
    return bytes(pdf.output(dest='S'))

def main():
    # Patient Information Section
    with st.expander("Patient Information", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            patient_name = st.text_input("Full Name")
        with col2:
            patient_id = st.text_input("Patient ID")
        
        clinical_notes = st.text_area("Clinical Notes", 
                                    placeholder="e.g., Persistent cough for 2 weeks, fever...")

    # Image Analysis Section
    st.markdown("## X-ray Analysis")
    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image (JPEG/PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False
    )

    if uploaded_file is not None:
        # Validate file size
        file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # MB
        if file_size > MAX_FILE_SIZE:
            st.error(f"File too large. Max size: {MAX_FILE_SIZE}MB")
            return

        # Display image
        st.image(uploaded_file, caption="Uploaded X-ray", use_container_width=True)
        
        if st.button("Analyze Image", type="primary"):
            with st.spinner("Analyzing X-ray..."):
                try:
                    files = {"file": uploaded_file.getvalue()}
                    response = requests.post(API_URL, files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.balloons()
                        
                        # Display results
                        st.markdown("## Diagnosis Results")
                        
                        if result["result"] == "Pneumonia detected":
                            st.error(f"🚨 **{result['result']}**")
                        else:
                            st.success(f"✅ **{result['result']}**")
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Confidence Level", 
                                     f"{result['confidence']*100:.2f}%")
                        with col_b:
                            st.metric("Model Used", 
                                     result.get("model", "pneumonia_classification_model.h5"))
                        
                        # Interpretation guide
                        with st.expander("Interpretation Guide"):
                            st.info("""
                            - **>85% confidence**: Strong positive indication
                            - **70-85%**: Likely positive
                            - **50-70%**: Moderate likelihood
                            - **<50%**: Unlikely
                            """)
                        
                        # Generate and download PDF
                        patient_data = {
                            "name": patient_name or "Not provided",
                            "id": patient_id or "N/A",
                            "notes": clinical_notes or "No clinical notes provided"
                        }
                        
                        pdf_bytes = generate_pdf_report(patient_data, result)
                        
                        st.download_button(
                            label="📄 Download Full Report",
                            data=pdf_bytes,
                            file_name=f"pneumonia_report_{patient_id or datetime.datetime.now().strftime('%Y%m%d')}.pdf",
                            mime="application/pdf",
                            key="pdf_download"  # Unique key to prevent duplicate widgets
                        )
                        
                    else:
                        st.error(f"API Error {response.status_code}: {response.text}")
                
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    main()