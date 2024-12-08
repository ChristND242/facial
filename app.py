import streamlit as st
from PIL import Image
import cv2
import numpy as np
import requests

# st.set_page_config(
#     page_title="Facial Recognition Awareness",
#     page_icon="👤",  # Face emoji to represent facial recognition
#     layout="wide",  # Optional: "wide" for a wider layout
# )


# Set up page configuration
st.set_page_config(page_title="Facial Recognition Awareness", page_icon="👤")

# Language selector
language = st.selectbox("Choose your language / Choisissez votre langue", ("English", "Français"))

# Translation dictionary
translations = {
    "English": {
        "title": "Facial Recognition Awareness Tool",
        "description": "Upload an image, and we'll show you how facial recognition systems can identify faces.",
        "upload_label": "Choose an image...",
        "uploaded_image_caption": "Uploaded Image",
        "faces_detected": "Number of faces detected",
        "error_no_faces": "No faces detected.",
        "error_api": "Could not detect faces. Please try again.",
        "error_large_image": "Image is too large. Automatically resized for processing.",
        "api_error_message": "API Error: {status_code} - {error_message}",
        "face": "Face",
        "age": "Age",
        "gender": "Gender",
        "smile": "Smile",
        "confidence": "confidence",
        "emotions": "Emotions",
        "head_pose": "Head Pose",
        "pitch": "Pitch",
        "roll": "Roll",
        "yaw": "Yaw",
        "beauty_female": "Beauty Score (Female)",
        "beauty_male": "Beauty Score (Male)",
        "ethnicity": "Ethnicity"
    },
    "Français": {
        "title": "Outil de Sensibilisation à la Reconnaissance Faciale",
        "description": "Téléchargez une image et nous vous montrerons comment les systèmes de reconnaissance faciale peuvent identifier des visages.",
        "upload_label": "Choisissez une image...",
        "uploaded_image_caption": "Image téléchargée",
        "faces_detected": "Nombre de visages détectés",
        "error_no_faces": "Aucun visage détecté.",
        "error_api": "Impossible de détecter des visages. Veuillez réessayer.",
        "error_large_image": "L'image est trop grande. Redimensionnée automatiquement pour le traitement.",
        "api_error_message": "Erreur de l'API: {status_code} - {error_message}",
        "face": "Visage",
        "age": "Âge",
        "gender": "Genre",
        "smile": "Sourire",
        "confidence": "confiance",
        "emotions": "Émotions",
        "head_pose": "Orientation de la tête",
        "pitch": "Inclinaison",
        "roll": "Roulement",
        "yaw": "Lacet",
        "beauty_female": "Score de beauté (Femmes)",
        "beauty_male": "Score de beauté (Hommes)",
        "ethnicity": "Ethnicité"
    }
}

# Get the selected language's translations
t = translations[language]

# Title and description
st.title(t["title"])
st.write(t["description"])

# Upload image widget
uploaded_file = st.file_uploader(t["upload_label"], type=["jpg", "jpeg", "png"])

# API credentials (stored in secrets.toml)
API_KEY = st.secrets["face_api"]["api_key"]
API_SECRET = st.secrets["face_api"]["api_secret"]
FACE_API_URL = "https://api-us.faceplusplus.com/facepp/v3/detect"

# Function to detect faces using the Face++ API
def detect_faces(image_bytes):
    files = {"image_file": image_bytes}
    payload = {
        "api_key": API_KEY,
        "api_secret": API_SECRET,
        "return_attributes": "age,gender,smiling,emotion,headpose,beauty,ethnicity",
    }
    
    response = requests.post(FACE_API_URL, files=files, data=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        # Log the error and display it to the user
        st.error(t["api_error_message"].format(status_code=response.status_code, error_message=response.text))
        return None

# Function to draw rectangles and smile scores on the faces
def draw_faces(image, faces_data):
    img_array = np.array(image.convert('RGB'))
    for face in faces_data["faces"]:
        rect = face["face_rectangle"]
        x, y, w, h = rect["left"], rect["top"], rect["width"], rect["height"]

        # Draw rectangle around face
        cv2.rectangle(img_array, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Draw smile score
        smile_score = face["attributes"]["smile"]["value"]
        cv2.putText(img_array, f"{t['smile']}: {smile_score:.2f}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img_array

# Function to resize image if it is too large
def resize_image(image, max_size=(1024, 1024)):
    original_size = image.size
    if original_size[0] > max_size[0] or original_size[1] > max_size[1]:
        st.warning(t["error_large_image"])
        image.thumbnail(max_size)
    return image

# Handle uploaded image
if uploaded_file is not None:
    # Load the uploaded image
    image = Image.open(uploaded_file)

    # Resize the image if too large
    resized_image = resize_image(image)

    # Display the resized image
    st.image(resized_image, caption=t["uploaded_image_caption"], use_column_width=True)

    # Convert image to bytes
    img_bytes = np.array(resized_image.convert("RGB"))
    img_bytes = cv2.imencode('.jpg', img_bytes)[1].tobytes()

    # Detect faces
    result = detect_faces(img_bytes)
    
    if result:
        faces = result.get("faces", [])
        if faces:
            st.write(f"{t['faces_detected']}: {len(faces)}")
            # Draw detected faces and display the image
            img_with_faces = draw_faces(resized_image, result)
            st.image(img_with_faces, caption=t["uploaded_image_caption"], use_column_width=True)

            # Display attributes for each face
            for i, face in enumerate(result["faces"]):
                st.write(f"{t['face']} {i+1}:")
                attributes = face["attributes"]
                
                # Display basic attributes
                st.write(f" - {t['age']}: {attributes['age']['value']}")
                st.write(f" - {t['gender']}: {attributes['gender']['value']}")
                
                # Smile detection
                st.write(f" - {t['smile']}: {attributes['smile']['value']:.2f}% {t['confidence']}")
                
                # Emotions
                st.write(f" - {t['emotions']}:")
                for emotion, value in attributes['emotion'].items():
                    st.write(f"   - {emotion.capitalize()}: {value:.2f}% {t['confidence']}")
                
                # Head pose (pitch, roll, yaw)
                st.write(f" - {t['head_pose']}:")
                st.write(f"   - {t['pitch']}: {attributes['headpose']['pitch_angle']:.2f}°")
                st.write(f"   - {t['roll']}: {attributes['headpose']['roll_angle']:.2f}°")
                st.write(f"   - {t['yaw']}: {attributes['headpose']['yaw_angle']:.2f}°")
                
                # Beauty score (if supported)
                if "beauty" in attributes:
                    st.write(f" - {t['beauty_female']}: {attributes['beauty']['female_score']:.2f}")
                    st.write(f" - {t['beauty_male']}: {attributes['beauty']['male_score']:.2f}")
                
                # Ethnicity (if supported)
                if "ethnicity" in attributes:
                    st.write(f" - {t['ethnicity']}: {attributes['ethnicity']['value']}")
        else:
            st.error(t["error_no_faces"])
    else:
        st.error(t["error_api"])
# Footer
st.markdown("""
    <footer style='text-align: center; padding: 10px; font-size: 14px; color: #ABB2B9;'>
    Created with ❤️ 
    </footer>
    <style>
        /* Hide Streamlit menu */
        #MainMenu {visibility: hidden;}

        /* Optional: Hide the "Made with Streamlit" watermark in the bottom-right corner */
        .css-1outpf7 {display: none;}
    </style>
    """, unsafe_allow_html=True)



html_code = """
<main>
  <h1>by<span>.ND</span></h1>
</main>

<style>
/* Custom font faces */
@font-face {
  font-family: "WHOA Spine Minimum";
  src: url("https://assets.codepen.io/174183/WHOA-Variable-Spine-v0.3.ttf") format("truetype");
  font-weight: normal;
  font-style: normal;
}

@font-face {
  font-family: "WHOA Top Minimum";
  src: url("https://assets.codepen.io/174183/WHOA-Variable-Top-v0.3.ttf") format("truetype");
  font-weight: normal;
  font-style: normal;
}

/* Body styling */
body {
  font-family: sans-serif;
  margin: 0;
  padding: 0;
  height: 100vh;
  width: 100vw;
  display: flex;
  justify-content: center;
  align-items: center;
  flex-direction: column;
  color: #ffffff;
  background: radial-gradient(circle, rgba(0, 255, 204, 0.2), #000000);
  animation: tunnelFlow 6s infinite ease-in-out;
  overflow: hidden;
}

/* Ensure the html and body cover the entire viewport */
html, body {
  height: 100%;
  width: 100%;
}

/* Tunnel animation */
@keyframes tunnelFlow {
  0% {
    transform: scale(1.05);
    opacity: 1;
  }
  50% {
    transform: scale(0.95);
    opacity: 0.7;
  }
  100% {
    transform: scale(1.05);
    opacity: 1;
  }
}

/* Main heading styling */
h1 {
  text-transform: uppercase;
  font-size: clamp(1.5rem, 4vw, 2rem);
  font-weight: normal;
  text-align: center;
  position: relative;
  z-index: 10;
  letter-spacing: 0.1rem;
  margin: 0;
  padding: 10px;
}

/* Span for the word "Established" with custom font and hover effect */
h1 span {
  display: block;
  cursor: pointer;
  font-family: "WHOA Top Minimum";
  font-variation-settings: "hrzn" 0, "vert" 0, "rota" 0, "zoom" 0;
  font-size: clamp(4rem, 12vw, 6rem);
  position: relative;
  line-height: 1;
  margin: 1rem auto;
}

h1 span::before {
  content: ".ND";
  font-family: "WHOA Spine Minimum";
  font-variation-settings: "hrzn" 0, "vert" 0, "rota" 0, "zoom" 0;
  font-size: 1em;
  position: absolute;
  top: 0;
  left: 0;
  z-index: -1;
}

/* Hover effect for text warping */
h1 span:hover, h1 span:hover::before {
  font-variation-settings: "hrzn" 820, "vert" -1000, "rota" 59, "zoom" 820;
  transition: all 1000ms cubic-bezier(0.42, 0, 0.11, 1.43);
}

h1 span:hover {
  -webkit-text-stroke: 1px white;
}

/* Matrix binary background effect */
#matrixCanvas {
  display: block;
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  z-index: 0;
}

/* Overlay */
.overlay {
  position: absolute;
  bottom: 0;
  left: 0;
  width: 100%;
  height: 30vh;
  z-index: 100;
  background: linear-gradient(
    0deg,
    rgba(255, 255, 255, 1) 75%,
    rgba(255, 255, 255, 0.9) 80%,
    rgba(255, 255, 255, 0.25) 95%,
    rgba(255, 255, 255, 0) 100%
  );
}

/* Text for Christ.ND */
.text {
  font-family: "Yanone Kaffeesatz";
  font-size: clamp(2rem, 8vw, 5rem);
  display: flex;
  justify-content: center;
  align-items: center;
  position: absolute;
  bottom: 15vh;
  left: 50%;
  transform: translateX(-50%);
  user-select: none;
}

.wrapper {
  padding-left: 10px;
  padding-right: 10px;
  padding-top: 10px;
}

.letter {
  transition: ease-out 1s;
  transform: translateY(40%);
}

.shadow {
  transform: scale(1, -1);
  color: #999;
  transition: ease-in 5s, ease-out 5s;
}

.wrapper:hover .letter {
  transform: translateY(-200%);
}

.wrapper:hover .shadow {
  opacity: 0;
  transform: translateY(200%);
}
</style>

<div id="matrixCanvas"></div> <!-- Move the canvas here -->

<div class="overlay"></div>

<div class="text">
  <div class="wrapper">
    <div id="C" class="letter">C</div>
    <div class="shadow">C</div>
  </div>
  <div class="wrapper">
    <div id="H" class="letter">H</div>
    <div class="shadow">H</div>
  </div>
  <div class="wrapper">
    <div id="R" class="letter">R</div>
    <div class="shadow">R</div>
  </div>
  <div class="wrapper">
    <div id="I" class="letter">I</div>
    <div class="shadow">I</div>
  </div>
  <div class="wrapper">
    <div id="S" class="letter">S</div>
    <div class="shadow">S</div>
  </div>
  <div class="wrapper">
    <div id="T" class="letter">T</div>
    <div class="shadow">T</div>
  </div>
</div>

<script>
// Binary drip-down matrix effect
const canvas = document.getElementById('matrixCanvas');
const ctx = canvas.getContext('2d');

// Set canvas size to fill the entire window
function setCanvasSize() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
}
setCanvasSize();
window.addEventListener('resize', setCanvasSize);

const fontSize = 16; // Size of binary characters
const columns = Math.floor(canvas.width / fontSize); // Number of columns for the rain
const colors = ['#28a745', '#6c757d', '#007bff', '#ffc107', '#dc3545', '#17a2b8']; // Security-related colors

// Create an array of drop positions, one for each column
const drops = Array(columns).fill(1);

// Create a function to get a random binary character
function getRandomBinary() {
    return Math.random() > 0.5 ? '1' : '0';
}

// Draw the matrix effect
function drawMatrix() {
    ctx.fillStyle = 'rgba(0, 0, 0, 0.05)'; // Slightly transparent black to create trail effect
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Set the font style for the binary rain
    ctx.font = `${fontSize}px 'Source Code Pro', monospace`;

    // Loop through each column and draw the characters
    drops.forEach((drop, i) => {
        // Set a random color from the cybersecurity palette
        ctx.fillStyle = colors[Math.floor(Math.random() * colors.length)];

        // Draw the binary character at the current drop position
        const binaryChar = getRandomBinary();
        ctx.fillText(binaryChar, i * fontSize, drop * fontSize);

        // Reset the drop to the top if it reaches the bottom or randomly
        if (drop * fontSize > canvas.height && Math.random() > 0.975) {
            drops[i] = 0;
        }

        // Increment the drop position
        drops[i]++;
    });
}

// Continuously animate the matrix binary rain effect
setInterval(drawMatrix, 50);
</script>
"""

st.components.v1.html(html_code, height=800)


