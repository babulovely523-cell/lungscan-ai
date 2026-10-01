from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from datetime import datetime
import os


def generate_report(user, img_path, result_data, out_path):
    """Generate a PDF report and save to out_path."""
    c = canvas.Canvas(out_path, pagesize=A4)
    w, h = A4

    # Header
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, h - 60, "AI Lungs Health Report")
    c.setFont("Helvetica", 11)
    c.drawString(50, h - 85, f"User: {user}")
    c.drawString(50, h - 100,
                 f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Images
    y = h - 130
    if os.path.exists(img_path):
        c.drawImage(ImageReader(img_path), 50, y - 220, width=200, height=200)
    if result_data.get("heatmap_path") and os.path.exists(result_data["heatmap_path"]):
        c.drawImage(ImageReader(result_data["heatmap_path"]),
                    280, y - 220, width=200, height=200)

    # Results
    y -= 250
    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, y, "Results:")
    c.setFont("Helvetica", 11)
    y -= 20
    c.drawString(50, y,
                 f"Lungs Cancer: {result_data['cancer_result']} "
                 f"(Confidence: {result_data['cancer_conf']*100:.1f}%)")
    y -= 18
    c.drawString(50, y,
                 f"Pneumonia: {result_data['pneumonia_result']} "
                 f"(Confidence: {result_data['pneumonia_conf']*100:.1f}%)")

    # Lifestyle
    y -= 30
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Lifestyle Recommendations:")
    c.setFont("Helvetica", 10)
    for line in result_data["lifestyle"]:
        y -= 15
        c.drawString(60, y, f"- {line}")
        if y < 120:
            c.showPage()
            y = h - 60

    # Critical
    y -= 25
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Critical Recommendations:")
    c.setFont("Helvetica", 10)
    for line in result_data["critical"]:
        y -= 15
        c.drawString(60, y, f"- {line}")
        if y < 120:
            c.showPage()
            y = h - 60

    # Disclaimer
    y -= 30
    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y, "Disclaimer:")
    c.setFont("Helvetica", 9)
    y -= 15
    c.drawString(50, y,
                 "This AI result is not 100% accurate. Consult a doctor for confirmation.")
    c.save()
    return out_path
