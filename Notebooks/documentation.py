import streamlit as st

def app():
    st.title("📚 Documentation")
    
    st.markdown("---")
    
    st.header("🎯 Project Overview")
    st.markdown("""
    This application uses **Deep Learning** and **Transfer Learning** to analyze medical images for two critical tasks:
    
    1. **Blood Cell Classification** - Detecting cancerous blood cells (Leukemia)
    2. **Brain Tumor Detection** - Localizing tumors in MRI scans
    
    Both models are trained on specialized medical imaging datasets and leverage state-of-the-art architectures.
    """)
    
    st.markdown("---")
    
    # Model 1: GoogLeNet
    st.header("🩸 Model 1: Blood Cell Classification (GoogLeNet)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Model Architecture")
        st.markdown("""
        - **Base Model:** GoogLeNet (Inception v1)
        - **Pre-training:** ImageNet dataset
        - **Transfer Learning:** Frozen convolutional layers
        - **Custom Classifier:**
          - FC Layer: 1024 → 512 (ReLU + Dropout 0.4)
          - FC Layer: 512 → 256 (ReLU)
          - FC Layer: 256 → 128 (ReLU)
          - Output Layer: 128 → 4 classes
        """)
        
    with col2:
        st.subheader("🔧 Technical Details")
        st.markdown("""
        - **Input Size:** 128×128 RGB images
        - **Normalization:** Mean=[0.5, 0.5, 0.5], Std=[0.5, 0.5, 0.5]
        - **Training:**
          - Optimizer: Adam (lr=0.001)
          - Loss: CrossEntropyLoss
          - Epochs: 10
          - Batch Size: 32
        - **Data Split:** 70% Train, 15% Validation, 15% Test
        """)
    
    st.subheader("🎯 Classification Classes")
    st.markdown("""
    The model classifies blood cell images into **4 categories**:
    
    1. **Benign** - Normal, healthy blood cells
    2. **Early Pre-B** - Early-stage Pre-B cell leukemia
    3. **Pre-B** - Pre-B cell leukemia
    4. **Pro-B** - Pro-B cell leukemia
    
    These classifications help in early detection and diagnosis of **Acute Lymphoblastic Leukemia (ALL)**.
    """)
    
    st.subheader("📈 Data Augmentation")
    st.markdown("""
    To balance the dataset and improve model robustness:
    - **Gaussian Blur** (radius=2)
    - **Random Noise** injection
    - **Horizontal Flip**
    - Target: 688 images per class in training set
    """)
    
    st.markdown("---")
    
    # Model 2: YOLOv8
    st.header("🧠 Model 2: Brain Tumor Detection (YOLOv8)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Model Architecture")
        st.markdown("""
        - **Base Model:** YOLOv8 (You Only Look Once v8)
        - **Task:** Object Detection
        - **Framework:** Ultralytics
        - **Backbone:** CSPDarknet with C2f modules
        - **Neck:** Path Aggregation Network (PAN)
        - **Head:** Decoupled detection head
        """)
        
    with col2:
        st.subheader("🔧 Technical Details")
        st.markdown("""
        - **Input Size:** Variable (auto-resized)
        - **Detection Threshold:** 0.25 confidence
        - **Training:**
          - Pre-trained weights: yolov8n.pt
          - Custom dataset with brain tumor annotations
          - Format: YOLO (class, x_center, y_center, width, height)
        - **Output:** Bounding boxes with confidence scores
        """)
    
    st.subheader("🎯 Detection Capability")
    st.markdown("""
    The model detects and localizes **brain tumors** in MRI scans:
    
    - **Input:** Brain MRI image (grayscale or RGB)
    - **Output:** Bounding box(es) around detected tumor(s)
    - **Confidence Score:** Percentage of model certainty
    - **Multiple Detections:** Can identify multiple tumors in one scan
    
    This helps radiologists quickly identify suspicious regions for further analysis.
    """)
    
    st.markdown("---")
    
    # User Guide
    st.header("👤 User Guide")
    
    st.subheader("🩸 How to Use Blood Cell Classification")
    st.markdown("""
    1. Navigate to **"App blood cells"** from the sidebar
    2. Click on the **"Browse files"** button
    3. Select a blood cell microscopy image (JPG, JPEG, or PNG)
    4. Wait for the image to upload and display
    5. The model will automatically classify the image
    6. **Expected Results:**
       - Predicted class name (Benign, Pre-B, Pro-B, or Early Pre-B)
       - Confidence percentage (typically 70-99%)
       - Higher confidence indicates stronger prediction
    
    **💡 Tips:**
    - Use clear, well-lit microscopy images
    - Ensure the cell is centered and in focus
    - Images should show individual or small groups of cells
    """)
    
    st.subheader("🧠 How to Use Brain Tumor Detection")
    st.markdown("""
    1. Navigate to **"App brain cells"** from the sidebar
    2. Click on the **"Browse files"** button
    3. Select a brain MRI scan image (JPG, JPEG, or PNG)
    4. Wait for the image to upload and display
    5. The model will process and detect tumors
    6. **Expected Results:**
       - Original MRI image displayed
       - Processed image with bounding box(es) around detected tumor(s)
       - Confidence score for each detection
       - Box color indicates detection confidence
    
    **💡 Tips:**
    - Use T1, T2, or FLAIR MRI sequences
    - Axial, sagittal, or coronal views are all supported
    - Clear contrast between tumor and healthy tissue improves detection
    - If no tumors detected, image shows without bounding boxes
    """)
    
    st.markdown("---")
    
    # Performance & Limitations
    st.header("⚠️ Important Notes")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("✅ Strengths")
        st.markdown("""
        - Fast inference time (<2 seconds)
        - High accuracy on similar medical images
        - Transfer learning from large datasets
        - User-friendly interface
        - No installation required
        """)
    
    with col2:
        st.subheader("⚠️ Limitations")
        st.markdown("""
        - **Not for clinical diagnosis** - For research/educational purposes only
        - Performance varies with image quality
        - Models trained on specific datasets
        - May not generalize to all imaging protocols
        - Always consult medical professionals
        """)
    
    st.markdown("---")
    
    # Technical Stack
    st.header("🛠️ Technical Stack")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🔬 Deep Learning")
        st.markdown("""
        - PyTorch
        - Torchvision
        - Ultralytics YOLOv8
        - Transfer Learning
        """)
    
    with col2:
        st.subheader("🎨 Frontend")
        st.markdown("""
        - Streamlit
        - PIL (Pillow)
        - Matplotlib
        - Seaborn
        """)
    
    with col3:
        st.subheader("📊 Data Processing")
        st.markdown("""
        - NumPy
        - Pandas
        - scikit-learn
        - split-folders
        """)
    
    st.markdown("---")
    
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h4>Made with ❤️ for Medical Image Analysis</h4>
        <p>Using Deep Learning to assist in early disease detection</p>
    </div>
    """, unsafe_allow_html=True)
