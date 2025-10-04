import qrcode
from PIL import Image, ImageDraw, ImageFont

# URL of your Google Form for feedback
feedback_url = "https://forms.gle/TNBEeTGtd8Cc5Bi2A"

# QR Code settings
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=10,
    border=4,
)
qr.add_data(feedback_url)
qr.make(fit=True)

# Create the QR code image
qr_image = qr.make_image(fill_color="#D63384", back_color="#FFFFFF")

# Convert qr_image to RGB mode to ensure compatibility
qr_image = qr_image.convert('RGB')

# Add a label below the QR code
qr_width, qr_height = qr_image.size
label_height = 50  # Height for the label area
new_image = Image.new('RGB', (qr_width, qr_height + label_height), color="#FFFFFF")  # White background

# Paste the QR code image using a 4-item tuple
new_image.paste(qr_image, (0, 0, qr_width, qr_height))

# Create a drawing context
draw = ImageDraw.Draw(new_image)

# Use a default font (or specify a font file if you have one)
try:
    font = ImageFont.truetype("arial.ttf", 24)  # Use Arial font (Windows)
except:
    font = ImageFont.load_default()  # Fallback to default font

# Add text below the QR code
text = "Scan for Feedback"
text_bbox = draw.textbbox((0, 0), text, font=font)
text_width = text_bbox[2] - text_bbox[0]
text_x = (qr_width - text_width) // 2
text_y = qr_height + 10  # Position below the QR code
draw.text((text_x, text_y), text, fill="#333333", font=font)

# Save the QR code with the label
output_path = "feedback_only_qr_code_with_label.png"
new_image.save(output_path)

# Display the QR code
new_image.show()

print(f"QR code with label generated and saved as {output_path}")