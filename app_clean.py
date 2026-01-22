"""
XHaloPathAnalyzer - Main Streamlit Application

Web-based GUI for custom image analysis on Halo digital pathology slides.
Provides interface for:
- Authentication with Halo API
- Slide selection and metadata viewing
- ROI export and image processing
- MedSAM segmentation analysis
- GeoJSON export for Halo import
Halo AI Workflow - Main Streamlit Application
Web-based GUI for digital pathology analysis with Halo API integration
"""

import streamlit as st
import asyncio
import numpy as np
from pathlib import Path
import pandas as pd
from datetime import datetime
import traceback

from config import Config
from utils.halo_api import HaloAPI
from utils.image_proc import (
    load_image_from_bytes,
    preprocess_for_medsam,
    postprocess_mask,
    overlay_mask_on_image,
    compute_mask_statistics
)
from utils.ml_models import MedSAMPredictor
from utils.geojson_utils import (
    mask_to_polygons,
    polygons_to_geojson
)
from PIL import Image
import io
import json
import logging
from typing import Optional, List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import local modules
from xhalo.api import HaloAPIClient, MockHaloAPIClient
from xhalo.ml import MedSAMPredictor, segment_tissue
from xhalo.utils import (
    load_image,
    resize_image,
    overlay_mask,
    mask_to_geojson,
    convert_to_halo_annotations,
    save_geojson
)

# Page configuration
st.set_page_config(
    page_title="XHaloPathAnalyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'local_mode' not in st.session_state:
        st.session_state.local_mode = False
    if 'api' not in st.session_state:
        st.session_state.api = None
    if 'predictor' not in st.session_state:
        st.session_state.predictor = None
    if 'selected_slide' not in st.session_state:
        st.session_state.selected_slide = None
    if 'uploaded_images' not in st.session_state:
        st.session_state.uploaded_images = []
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'current_image' not in st.session_state:
        st.session_state.current_image = None
    if 'current_mask' not in st.session_state:
        st.session_state.current_mask = None
    if 'current_image_name' not in st.session_state:
        st.session_state.current_image_name = None

init_session_state()


def authentication_page():
    """Authentication and configuration page"""
    st.markdown('<h1 class="main-header">🔬 XHaloPathAnalyzer</h1>', unsafe_allow_html=True)
    st.markdown("### Web-Based GUI for Halo Digital Pathology Analysis")
    
    st.markdown("---")
    
    # Add mode selection at the top
    st.subheader("🎯 Select Analysis Mode")
    mode = st.radio(
        "Choose how you want to work:",
        ["🔌 Halo API Mode", "📁 Local Image Upload Mode"],
        help="Halo API Mode connects to your Halo instance. Local Mode allows direct upload of images."
    )
    
    if mode == "📁 Local Image Upload Mode":
        st.info("💡 **Local Mode**: Upload images (JPG, PNG, TIFF) directly for analysis without Halo API connection")
        
        if st.button("✅ Start Local Mode", type="primary", use_container_width=True):
            st.session_state.authenticated = True
            st.session_state.local_mode = True
            st.success("✅ Local mode activated!")
            st.rerun()
            
        st.markdown("---")
        st.markdown("### 📋 Features in Local Mode")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**📤 Image Upload**")
            st.write("Upload single or multiple images for analysis")
        with col2:
            st.markdown("**🤖 AI Analysis**")
            st.write("Run MedSAM segmentation on uploaded images")
        with col3:
            st.markdown("**📥 Export Results**")
            st.write("Download segmentation masks and GeoJSON")
            
    else:
        st.subheader("🔐 Halo API Authentication")
        st.write("Connect to your Halo digital pathology instance")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            endpoint = st.text_input(
                "Halo API Endpoint",
                value=Config.HALO_API_ENDPOINT,
                placeholder="https://your-halo-instance.com/graphql",
                help="URL of your Halo GraphQL API endpoint"
            )
            
            token = st.text_input(
                "API Token",
                value=Config.HALO_API_TOKEN,
                type="password",
                placeholder="Enter your API token",
                help="API authentication token from Halo settings"
            )
            
            if st.button("🔌 Connect", type="primary", use_container_width=True):
                if not endpoint or not token:
                    st.error("❌ Please provide both endpoint and token")
                else:
                    with st.spinner("Testing connection..."):
                        try:
                            # Create API instance
                            api = HaloAPI(endpoint, token)
                            
                            # Test connection
                            success = asyncio.run(api.test_connection())
                            
                            if success:
                                st.session_state.api = api
                                st.session_state.authenticated = True
                                st.session_state.local_mode = False
                                st.success("✅ Connected successfully!")
                                st.rerun()
                            else:
                                st.error("❌ Connection test failed")
                                
                        except Exception as e:
                            st.error(f"❌ Connection failed: {str(e)}")
        
        with col2:
            st.info("""
            **How to get API token:**
            1. Log into Halo
            2. Go to Settings → API
            3. Create new token
            4. Copy and paste here
            """)
        
        st.markdown("---")
        st.markdown("### 📋 Features in Halo Mode")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**🔍 Slide Selection**")
            st.write("Browse and select slides from your Halo instance")
        with col2:
            st.markdown("**🤖 AI Analysis**")
            st.write("Run MedSAM segmentation on regions of interest")
        with col3:
            st.markdown("**📤 Export Results**")
            st.write("Generate GeoJSON annotations for Halo import")


def slide_selection_page():
    """Slide selection and browsing interface"""
    st.title("🔬 Slide Selection")
    
    if st.session_state.api is None:
        st.warning("⚠️ Please authenticate first")
        return
    
    # Fetch slides
    with st.spinner("Loading slides from Halo..."):
        try:
            slides = asyncio.run(st.session_state.api.get_slides(limit=100))
        except Exception as e:
            st.error(f"❌ Failed to fetch slides: {str(e)}")
            return
    
    if not slides:
        st.warning("⚠️ No slides found in your Halo instance")
        return
    
    st.success(f"✅ Found {len(slides)} slides")
    
    # Convert to DataFrame for display
    df = pd.DataFrame(slides)
    
    # Add filters
    st.subheader("🔍 Filter Slides")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        name_filter = st.text_input("Filter by name", "")
    with col2:
        study_filter = st.text_input("Filter by study ID", "")
    
    # Apply filters
    if name_filter:
        df = df[df['name'].str.contains(name_filter, case=False, na=False)]
    if study_filter:
        df = df[df['studyId'].str.contains(study_filter, case=False, na=False)]
    
    st.markdown("---")
    
    # Display slides table
    st.subheader("📊 Available Slides")
    
    if len(df) == 0:
        st.info("No slides match the filter criteria")
        return
    
    # Select slide from dropdown
    slide_names = df['name'].tolist()
    selected_name = st.selectbox(
        "Select a slide",
        slide_names,
        help="Choose a slide to analyze"
    )
    
    # Get selected slide data
    selected_idx = slide_names.index(selected_name)
    selected_slide = slides[selected_idx]
    
    # Display slide details
    st.markdown("---")
    st.subheader("📄 Slide Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Name", selected_slide['name'])
        st.metric("Width", f"{selected_slide['width']:,} px")
    with col2:
        st.metric("ID", selected_slide['id'][:16] + "...")
        st.metric("Height", f"{selected_slide['height']:,} px")
    with col3:
        mpp = selected_slide.get('mpp', 'N/A')
        st.metric("MPP", f"{mpp}" if mpp != 'N/A' else "N/A")
        st.metric("Format", selected_slide.get('format', 'Unknown'))
    
    # Save to session state
    if st.button("✅ Select This Slide", type="primary", use_container_width=True):
        st.session_state.selected_slide = selected_slide
        st.success(f"✅ Selected: {selected_slide['name']}")


def image_upload_page():
    """Image upload interface for local mode"""
    st.title("📤 Image Upload")
    
    st.markdown("""
    Upload one or more images (JPG, PNG, TIFF) for analysis. 
    Batch processing allows you to analyze multiple images sequentially.
    """)
    
    st.markdown("---")
    
    # File uploader
    st.subheader("📁 Select Images")
    
    uploaded_files = st.file_uploader(
        "Choose image files",
        type=["jpg", "jpeg", "png", "tiff", "tif"],
        accept_multiple_files=True,
        help="Supported formats: JPG, PNG, TIFF"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
        
        # Display uploaded files
        st.subheader("📊 Uploaded Files")
        
        file_data = []
        for uploaded_file in uploaded_files:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            file_data.append({
                "Filename": uploaded_file.name,
                "Size (MB)": f"{file_size_mb:.2f}",
                "Type": uploaded_file.type
            })
        
        df = pd.DataFrame(file_data)
        st.dataframe(df, use_container_width=True)
        
        st.markdown("---")
        
        # Select image to analyze
        st.subheader("🎯 Select Image for Analysis")
        
        selected_filename = st.selectbox(
            "Choose an image to analyze",
            [f.name for f in uploaded_files],
            help="Select which image to process"
        )
        
        # Find selected file
        selected_file = None
        for f in uploaded_files:
            if f.name == selected_filename:
                selected_file = f
                break
        
        if selected_file:
            # Display image preview
            col1, col2 = st.columns([2, 1])
            
            with col1:
                try:
                    # Load and display preview
                    image_bytes = selected_file.read()
                    selected_file.seek(0)  # Reset file pointer
                    image = load_image_from_bytes(image_bytes)
                    
                    st.image(image, caption=selected_file.name, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Error loading image: {str(e)}")
                    return
            
            with col2:
                st.metric("Filename", selected_file.name)
                st.metric("Dimensions", f"{image.shape[1]} × {image.shape[0]} px")
                st.metric("Channels", image.shape[2] if len(image.shape) > 2 else 1)
            
            # Load image button
            if st.button("✅ Load This Image for Analysis", type="primary", use_container_width=True):
                try:
                    # Load image into session state
                    image_bytes = selected_file.read()
                    selected_file.seek(0)
                    image = load_image_from_bytes(image_bytes)
                    
                    # Store in session state
                    st.session_state.current_image = image
                    st.session_state.current_image_name = selected_file.name
                    st.session_state.uploaded_images = uploaded_files
                    
                    # Create pseudo-slide object for compatibility
                    st.session_state.selected_slide = {
                        'name': selected_file.name,
                        'id': f"local_{selected_file.name}",
                        'width': image.shape[1],
                        'height': image.shape[0],
                        'mpp': None,
                        'format': selected_file.type
                    }
                    
                    st.success(f"✅ Image loaded: {selected_file.name}")
                    st.info("👉 Go to **Analysis** page to run segmentation")
                    
                except Exception as e:
                    st.error(f"❌ Error loading image: {str(e)}")
    else:
        st.info("💡 Please upload one or more images to get started")


def analysis_page():
    """Analysis interface with MedSAM segmentation"""
    st.title("🤖 AI-Powered Analysis")
    
    if st.session_state.selected_slide is None:
        if st.session_state.local_mode:
            st.warning("⚠️ Please upload and select an image first")
        else:
            st.warning("⚠️ Please select a slide first")
        return
    
    slide = st.session_state.selected_slide
    st.info(f"📊 Analyzing: **{slide['name']}**")
    
    # Check if in local mode or Halo mode
    is_local_mode = st.session_state.local_mode or slide['id'].startswith('local_')
    
    # ROI selection (only for Halo mode or if image is already loaded)
    if not is_local_mode or st.session_state.current_image is None:
        st.subheader("📍 Region of Interest (ROI)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            x = st.number_input("X coordinate (pixels)", min_value=0, max_value=slide['width'], value=0, step=100)
            width = st.number_input("Width (pixels)", min_value=1, max_value=slide['width'], value=1024, step=100)
        
        with col2:
            y = st.number_input("Y coordinate (pixels)", min_value=0, max_value=slide['height'], value=0, step=100)
            height = st.number_input("Height (pixels)", min_value=1, max_value=slide['height'], value=1024, step=100)
        
        # Validate ROI
        if x + width > slide['width']:
            st.error(f"❌ ROI extends beyond slide width ({slide['width']} px)")
            return
        if y + height > slide['height']:
            st.error(f"❌ ROI extends beyond slide height ({slide['height']} px)")
            return
    else:
        # For local mode with pre-loaded image, use full image
        x, y = 0, 0
        width, height = slide['width'], slide['height']
        st.info(f"📐 Analyzing full image: {width} × {height} pixels")
    
    st.markdown("---")
    
    # Analysis settings
    st.subheader("⚙️ Analysis Settings")
    
    use_prompts = st.checkbox("Use point/box prompts", value=False, 
                             help="Enable interactive prompts for segmentation")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
            with st.spinner("Processing..."):
                try:
                    # Get image based on mode
                    if is_local_mode and st.session_state.current_image is not None:
                        # Use pre-loaded image from local upload
                        st.info("⏳ Using uploaded image...")
                        image = st.session_state.current_image
                        
                        # If ROI is specified and not full image, crop it
                        if x > 0 or y > 0 or width < slide['width'] or height < slide['height']:
                            image = image[y:y+height, x:x+width]
                    else:
                        # Download region from Halo API
                        st.info("⏳ Downloading region from Halo... This may take a moment for large regions.")
                        region_data = st.session_state.api.download_region(
                            slide['id'], x, y, width, height
                        )
                        
                        if not region_data:
                            st.error("❌ Failed to download region - no data received")
                            return
                        
                        # Load image
                        st.info("⏳ Loading image...")
                        image = load_image_from_bytes(region_data)
                    
                    st.session_state.current_image = image
                    
                    # Initialize predictor if needed
                    if st.session_state.predictor is None:
                        st.info("⏳ Loading MedSAM model...")
                        st.session_state.predictor = MedSAMPredictor(
                            Config.MEDSAM_CHECKPOINT,
                            model_type=Config.MODEL_TYPE,
                            device=Config.DEVICE
                        )
                    
                    # Preprocess
                    st.info("⏳ Preprocessing image...")
                    preprocessed, metadata = preprocess_for_medsam(image, Config.DEFAULT_TARGET_SIZE)
                    
                    # Run inference
                    st.info("⏳ Running MedSAM segmentation...")
                    mask = st.session_state.predictor.predict(preprocessed)
                    
                    # Postprocess
                    st.info("⏳ Postprocessing results...")
                    final_mask = postprocess_mask(mask, metadata)
                    st.session_state.current_mask = final_mask
                    
                    # Compute statistics
                    mpp = slide.get('mpp')
                    stats = compute_mask_statistics(final_mask, mpp)
                    
                    # Store results
                    st.session_state.analysis_results = {
                        'image': image,
                        'mask': final_mask,
                        'roi': (x, y, width, height),
                        'statistics': stats,
                        'slide_id': slide['id'],
                        'slide_name': slide['name'],
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    st.success("✅ Analysis complete!")
                    
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    st.code(traceback.format_exc())
    
    with col2:
        if st.button("🔄 Clear Results"):
            st.session_state.analysis_results = None
            st.session_state.current_image = None
            st.session_state.current_mask = None
            st.success("✅ Cleared")
    
    # Display results
    if st.session_state.analysis_results is not None:
        st.markdown("---")
        st.subheader("📊 Results")
        
        results = st.session_state.analysis_results
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        stats = results['statistics']
        
        with col1:
            st.metric("Positive Pixels", f"{stats['num_positive_pixels']:,}")
        with col2:
            st.metric("Coverage", f"{stats['coverage_percent']:.2f}%")
        with col3:
            if 'area_um2' in stats:
                st.metric("Area", f"{stats['area_um2']:.2f} µm²")
        with col4:
            if 'area_mm2' in stats:
                st.metric("Area", f"{stats['area_mm2']:.4f} mm²")
        
        # Visualizations
        st.markdown("### 🖼️ Visualization")

