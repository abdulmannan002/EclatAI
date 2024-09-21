import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image as PlatypusImage
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from datetime import datetime

# Create different text styles for the report
styles = getSampleStyleSheet()

# Title Style
title_style = ParagraphStyle(
    name='Title',
    fontName='Helvetica-Bold',
    fontSize=18,
    spaceAfter=12,
    alignment=TA_CENTER
)

# Body Text Style
body_style = ParagraphStyle(
    name='BodyText',
    fontName='Helvetica',
    fontSize=12,
    spaceAfter=12,
    alignment=TA_LEFT
)

# Custom style for spacing (adjusted for better layout)
custom_style = ParagraphStyle(
    'CustomStyle',
    parent=styles['BodyText'],
    spaceAfter=18,  # Adds more space after each text block
    leading=16,     # Adjusts line height (distance between lines)
    wordSpace=0.5,  # Adjusts word spacing
    alignment=0,    # Left alignment (0=left, 1=center, 2=right, 4=justified)
)

def create_pdf_report(prediction, heatmap_path, report_text, uploaded_image_path):
    """
    Creates a PDF report using ReportLab and returns it as a BytesIO buffer.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []

    # Title
    title = Paragraph("ÉclatAI: Image Classification and Explainability Report", title_style)
    elements.append(title)

    # Date/Time
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    date_time_paragraph = Paragraph(f"Date/Time: {now}", body_style)
    elements.append(date_time_paragraph)

    # Prediction
    prediction_paragraph = Paragraph(f"Prediction: {prediction}", custom_style)
    elements.append(prediction_paragraph)

    # Add uploaded image (if exists)
    if uploaded_image_path:
        img = PlatypusImage(uploaded_image_path, width=400, height=200)
        elements.append(img)

    # Add heatmap image (if exists)
    if heatmap_path:
        heatmap_img = PlatypusImage(heatmap_path, width=400, height=200)
        elements.append(heatmap_img)

    # Add report text (without paragraph-like formatting)
    report_text_paragraph = Paragraph(report_text, custom_style)
    elements.append(report_text_paragraph)

    # Build PDF document
    doc.build(elements)
    buffer.seek(0)
    
    return buffer
