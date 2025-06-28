import gradio as gr
import pandas as pd
import numpy as np
from PIL import Image
import io
import base64
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import tempfile
import os
from datetime import datetime
import torch
import torchvision.transforms as transforms

# Mock model for demonstration (replace with your actual model)
class MockPredictionModel:
    def __init__(self):
        self.loaded = True
    
    def predict_retinal_image(self, image):
        # Simulate model prediction
        return {
            'diabetic_retinopathy_risk': np.random.uniform(0.1, 0.9),
            'glaucoma_risk': np.random.uniform(0.1, 0.8),
            'macular_degeneration_risk': np.random.uniform(0.1, 0.7),
            'confidence': np.random.uniform(0.7, 0.95)
        }
    
    def analyze_biomarkers(self, biomarker_data):
        # Simulate biomarker analysis
        return {
            'cardiovascular_risk': np.random.uniform(0.2, 0.8),
            'metabolic_score': np.random.uniform(0.1, 0.9),
            'inflammation_markers': np.random.uniform(0.1, 0.6)
        }

# Initialize model
model = MockPredictionModel()

def process_patient_data(name, age, sex, medical_history, retinal_images, biomarker_csv):
    """Process all patient data and generate predictions"""
    
    if not name or not age or not retinal_images:
        return None, "Please fill in all required fields and upload at least one retinal image."
    
    results = {
        'patient_info': {
            'name': name,
            'age': age,
            'sex': sex,
            'medical_history': medical_history,
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        'retinal_predictions': [],
        'biomarker_analysis': None
    }
    
    # Process retinal images
    for i, img_file in enumerate(retinal_images):
        if img_file is not None:
            try:
                image = Image.open(img_file)
                prediction = model.predict_retinal_image(image)
                prediction['image_name'] = f"Image_{i+1}"
                results['retinal_predictions'].append(prediction)
            except Exception as e:
                return None, f"Error processing retinal image {i+1}: {str(e)}"
    
    # Process biomarker CSV if provided
    if biomarker_csv is not None:
        try:
            df = pd.read_csv(biomarker_csv)
            biomarker_analysis = model.analyze_biomarkers(df)
            results['biomarker_analysis'] = biomarker_analysis
            results['biomarker_data'] = df.to_dict('records')[:5]  # First 5 rows for report
        except Exception as e:
            return None, f"Error processing biomarker CSV: {str(e)}"
    
    # Generate PDF report
    pdf_path = generate_pdf_report(results)
    
    # Create summary text
    summary = create_summary_text(results)
    
    return pdf_path, summary

def create_summary_text(results):
    """Create a text summary of the analysis"""
    summary = f"""
## Medical Analysis Report for {results['patient_info']['name']}

**Patient Information:**
- Age: {results['patient_info']['age']}
- Sex: {results['patient_info']['sex']}
- Analysis Date: {results['patient_info']['date']}

**Retinal Image Analysis:**
"""
    
    for pred in results['retinal_predictions']:
        summary += f"""
- **{pred['image_name']}:**
  - Diabetic Retinopathy Risk: {pred['diabetic_retinopathy_risk']:.2%}
  - Glaucoma Risk: {pred['glaucoma_risk']:.2%}
  - Macular Degeneration Risk: {pred['macular_degeneration_risk']:.2%}
  - Confidence: {pred['confidence']:.2%}
"""
    
    if results['biomarker_analysis']:
        summary += f"""
**Biomarker Analysis:**
- Cardiovascular Risk: {results['biomarker_analysis']['cardiovascular_risk']:.2%}
- Metabolic Score: {results['biomarker_analysis']['metabolic_score']:.2%}
- Inflammation Markers: {results['biomarker_analysis']['inflammation_markers']:.2%}
"""
    
    summary += """
**Recommendations:**
- Regular follow-up with ophthalmologist recommended
- Continue monitoring biomarkers if applicable
- Maintain healthy lifestyle and medication compliance

*Note: This is an AI-generated analysis and should not replace professional medical advice.*
"""
    
    return summary

def generate_pdf_report(results):
    """Generate a PDF report with the analysis results"""
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    temp_path = temp_file.name
    temp_file.close()
    
    # Create PDF document
    doc = SimpleDocTemplate(temp_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=1  # Center alignment
    )
    story.append(Paragraph("Medical Analysis Report", title_style))
    story.append(Spacer(1, 12))
    
    # Patient Information
    story.append(Paragraph("Patient Information", styles['Heading2']))
    patient_data = [
        ['Name:', results['patient_info']['name']],
        ['Age:', str(results['patient_info']['age'])],
        ['Sex:', results['patient_info']['sex']],
        ['Analysis Date:', results['patient_info']['date']]
    ]
    
    if results['patient_info']['medical_history']:
        patient_data.append(['Medical History:', results['patient_info']['medical_history']])
    
    patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
    patient_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 20))
    
    # Retinal Analysis Results
    story.append(Paragraph("Retinal Image Analysis Results", styles['Heading2']))
    
    for pred in results['retinal_predictions']:
        story.append(Paragraph(f"<b>{pred['image_name']}</b>", styles['Heading3']))
        
        retinal_data = [
            ['Risk Factor', 'Probability', 'Risk Level'],
            ['Diabetic Retinopathy', f"{pred['diabetic_retinopathy_risk']:.1%}", get_risk_level(pred['diabetic_retinopathy_risk'])],
            ['Glaucoma', f"{pred['glaucoma_risk']:.1%}", get_risk_level(pred['glaucoma_risk'])],
            ['Macular Degeneration', f"{pred['macular_degeneration_risk']:.1%}", get_risk_level(pred['macular_degeneration_risk'])],
            ['Model Confidence', f"{pred['confidence']:.1%}", '']
        ]
        
        retinal_table = Table(retinal_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
        retinal_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(retinal_table)
        story.append(Spacer(1, 12))
    
    # Biomarker Analysis (if available)
    if results['biomarker_analysis']:
        story.append(Paragraph("Biomarker Analysis", styles['Heading2']))
        
        biomarker_data = [
            ['Biomarker', 'Score', 'Interpretation'],
            ['Cardiovascular Risk', f"{results['biomarker_analysis']['cardiovascular_risk']:.1%}", get_risk_level(results['biomarker_analysis']['cardiovascular_risk'])],
            ['Metabolic Score', f"{results['biomarker_analysis']['metabolic_score']:.1%}", get_risk_level(results['biomarker_analysis']['metabolic_score'])],
            ['Inflammation Markers', f"{results['biomarker_analysis']['inflammation_markers']:.1%}", get_risk_level(results['biomarker_analysis']['inflammation_markers'])]
        ]
        
        biomarker_table = Table(biomarker_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
        biomarker_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(biomarker_table)
        story.append(Spacer(1, 20))
    
    # Recommendations
    story.append(Paragraph("Recommendations", styles['Heading2']))
    recommendations = """
    • Schedule regular follow-up appointments with your ophthalmologist
    • Monitor blood pressure and glucose levels regularly
    • Maintain a healthy diet rich in antioxidants
    • Protect eyes from UV exposure
    • Report any sudden vision changes immediately
    • Consider lifestyle modifications based on risk factors identified
    """
    story.append(Paragraph(recommendations, styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Disclaimer
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.grey
    )
    disclaimer = """
    <b>IMPORTANT DISCLAIMER:</b> This report is generated by an AI system for informational purposes only. 
    It should not be used as a substitute for professional medical advice, diagnosis, or treatment. 
    Always consult with qualified healthcare professionals regarding your medical condition.
    """
    story.append(Paragraph(disclaimer, disclaimer_style))
    
    # Build PDF
    doc.build(story)
    
    return temp_path

def get_risk_level(probability):
    """Convert probability to risk level description"""
    if probability < 0.3:
        return "Low"
    elif probability < 0.6:
        return "Moderate"
    else:
        return "High"

# Create Gradio interface
with gr.Blocks(title="Medical Prediction System", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🏥 Medical Prediction System
    
    Upload patient data, retinal images, and biomarker files to generate a comprehensive medical analysis report.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Patient Information")
            name_input = gr.Textbox(
                label="Patient Name *",
                placeholder="Enter patient's full name",
                info="Required field"
            )
            
            with gr.Row():
                age_input = gr.Number(
                    label="Age *",
                    minimum=1,
                    maximum=120,
                    value=None,
                    info="Patient age in years"
                )
                sex_input = gr.Dropdown(
                    label="Sex *",
                    choices=["Male", "Female", "Other"],
                    value=None,
                    info="Biological sex"
                )
            
            medical_history = gr.Textbox(
                label="Medical History (Optional)",
                lines=3,
                placeholder="Enter relevant medical history, current medications, etc.",
                info="Any relevant medical background"
            )
            
           
            gr.Markdown("## File Uploads")
            retinal_images = gr.File(
                label="Retinal Images *",
                file_count="multiple",
                file_types=[".jpg", ".jpeg", ".png", ".tiff"]
            )
            
            biomarker_csv = gr.File(
                label="Biomarker Data (Optional)",
                file_types=[".csv"]
            )

        with gr.Column(scale=1):
            gr.Markdown("## Analysis Results")
            
            process_btn = gr.Button(
                "🔬 Generate Analysis Report",
                variant="primary",
                size="lg"
            )
            
            status_output = gr.Textbox(
                label="Status",
                interactive=False,
                info="Processing status and messages"
            )
            
            summary_output = gr.Markdown(
                label="Analysis Summary",
                visible=True
            )
            
            pdf_output = gr.File(
                label="Download PDF Report",
                interactive=False,
                visible=True
            )
    
    # Event handlers
    process_btn.click(
        fn=process_patient_data,
        inputs=[
            name_input,
            age_input,
            sex_input,
            medical_history,
            retinal_images,
            biomarker_csv
        ],
        outputs=[pdf_output, summary_output]
    )
    
    # Example section
    gr.Markdown("""
    ## 📋 Instructions
    
    1. **Fill Patient Information**: Enter the patient's name, age, and sex (required fields marked with *)
    2. **Add Medical History**: Optionally provide relevant medical background
    3. **Upload Retinal Images**: Upload one or more retinal photographs (required)
    4. **Upload Biomarkers**: Optionally upload a CSV file containing biomarker data
    5. **Generate Report**: Click the analysis button to process the data
    6. **Download Results**: View the summary and download the comprehensive PDF report
    
    ### Supported File Formats
    - **Images**: JPG, JPEG, PNG, TIFF
    - **Biomarker Data**: CSV format
    
    ### Sample CSV Format for Biomarkers
    ```
    biomarker,value,unit,reference_range
    glucose,120,mg/dL,70-100
    hba1c,6.2,%,<5.7
    cholesterol,200,mg/dL,<200
    ```
    """)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True
    )

#http://localhost:7860    