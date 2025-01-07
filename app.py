import streamlit as st
from PIL import Image
import torch
from torch import nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b1
import base64
# Function to encode the image as Base64
def get_base64_of_bin_file(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Load and encode the background image
background_image_path = "background_1.webp"  # Replace with your image path
background_image = get_base64_of_bin_file(background_image_path)

# CSS for the background
page_bg_css = f"""
<style>
    body {{
        background-image: url("data:image/png;base64,{background_image}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .stApp {{
        background: rgba(0, 0, 0, 0.7); /* Transparency for readability */
        border-radius: 15px;
        padding: 10px;
    }}
</style>
"""
st.markdown(page_bg_css, unsafe_allow_html=True)


CLASS_LABELS = ['Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy',
            'Cherry_(including_sour)___Powdery_mildew','Cherry_(including_sour)___healthy',
            'Grape___Black_rot','Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy',
            'Peach___Bacterial_spot','Peach___healthy']

#m My custom Modeling 
class DiseaseModel(nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()
        self.model = model
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features=num_classes, bias=True)
        )
        self.pretrained()
        
    def pretrained(self):
        for param in self.model.features.parameters():
            param.requires_grad = False
        self.model.classifier = self.classifier
        
    def forward(self, x):
        return self.model(x)



# Load the model
@st.cache_resource
def load_model():
    base_model = efficientnet_b1(weights=None)  # Load base model without weights
    num_classes = len(CLASS_LABELS)  # Replace with your number of classes
    model = DiseaseModel(base_model, num_classes)
    
    try:
        model.load_state_dict(torch.load("./custom_model.pth"))
    except Exception as e:
        print("Error loading model:", e)
    
    model.eval()
    return model

model = load_model()

# Define the image transformations
def transform_image(image):
    transform_image=transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor()
    ])
    return transform_image(image).unsqueeze(0)

# Streamlit UI
st.title("🌱 Plant Disease Detection")
st.write("Upload an image of a plant leaf to identify its condition.")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image:
    # Display uploaded image (center-aligned and resized)
    col1, col2, col3 = st.columns([1, 2, 1])  # Create columns for alignment
    with col2:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=False, width=300)

    # Prediction process
    try:
        with st.spinner("Analyzing..."):
            # Transform the image and make predictions
            input_tensor = transform_image(image)
            with torch.no_grad():
                output = model(input_tensor)
                pred_idx=output.argmax(dim=1)
              
            # Display the result
        st.write(f"**Prediction:** {CLASS_LABELS[pred_idx]}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
