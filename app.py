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
import cv2
from pathlib import Path
import pandas as pd
from datetime import datetime
import traceback

from config import Config
from utils.halo_api import HaloAPI
from utils.image_proc import (
    load_image_from_bytes,
    overlay_mask_on_image,
    compute_mask_statistics
)
from utils.ml_models import MedSAMPredictor as UtilsMedSAMPredictor, _ensure_rgb_uint8, _compute_tissue_bbox
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

# Constants for visualization
PROMPT_BOX_COLOR = (0, 255, 0)  # Green in RGB
PROMPT_BOX_THICKNESS = 3
PROMPT_BOX_LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
PROMPT_BOX_LABEL_SCALE = 1.0
PROMPT_BOX_LABEL_THICKNESS = 2
PROMPT_BOX_LABEL_Y_OFFSET = 10
PROMPT_BOX_LABEL_MIN_Y = 20

# Import local modules
from xhalo.api import HaloAPIClient, MockHaloAPIClient
from xhalo.ml import MedSAMPredictor as XHaloMedSAMPredictor, segment_tissue
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
    # Multi-image queue support
    if 'images' not in st.session_state:
        st.session_state.images = []  # List of dicts with id, name, bytes, np_rgb_uint8, status, error, result
    if 'batch_running' not in st.session_state:
        st.session_state.batch_running = False
    if 'batch_index' not in st.session_state:
        st.session_state.batch_index = 0

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
        
        if st.button("✅ Start Local Mode", type="primary", ):
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
            
            if st.button("🔌 Connect", type="primary", ):
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
    if st.button("✅ Select This Slide", type="primary", ):
        st.session_state.selected_slide = selected_slide
        st.success(f"✅ Selected: {selected_slide['name']}")


def run_analysis_on_item(item: Dict[str, Any], prompt_mode: str = "auto_box", 
                         multimask_output: bool = False, 
                         min_area_ratio: float = 0.01,
                         morph_kernel_size: int = 5) -> Dict[str, Any]:
    """
    Run analysis on a single image item.
    
    Args:
        item: Image item dict with 'bytes' field containing raw image data
        prompt_mode: Segmentation prompt mode (auto_box, full_box, point)
        multimask_output: Whether to generate multiple mask predictions
        min_area_ratio: Minimum area ratio for tissue detection (0-1)
        morph_kernel_size: Kernel size for morphological operations (odd integer)
        
    Returns:
        Dict with the following keys:
            - image: numpy.ndarray - Original RGB image (H, W, 3)
            - mask: numpy.ndarray - Binary segmentation mask (H, W)
            - statistics: dict - Mask statistics (num_positive_pixels, coverage_percent, etc.)
            - overlay: numpy.ndarray - Image with mask overlay
            - prompt_box: numpy.ndarray or None - Bounding box used for prompt [x1, y1, x2, y2]
            - img_with_box: numpy.ndarray or None - Image with prompt box visualization
            - prompt_mode: str - Prompt mode used
            - timestamp: str - ISO format timestamp
    
    Raises:
        Exception: If image loading, segmentation, or processing fails
        
    Example:
        >>> item = {'bytes': image_bytes, 'name': 'test.png', ...}
        >>> result = run_analysis_on_item(item, prompt_mode='auto_box')
        >>> print(result['statistics']['coverage_percent'])
    """
    # Decode bytes to RGB uint8 numpy
    image = load_image_from_bytes(item['bytes'])
    item['np_rgb_uint8'] = image  # Store for future use
    
    # Initialize predictor if needed
    if st.session_state.predictor is None:
        st.session_state.predictor = UtilsMedSAMPredictor(
            Config.MEDSAM_CHECKPOINT,
            model_type=Config.MODEL_TYPE,
            device=Config.DEVICE
        )
    
    # Compute prompt box for visualization
    prompt_box = None
    if prompt_mode == "auto_box":
        image_rgb = _ensure_rgb_uint8(image)
        prompt_box = _compute_tissue_bbox(image_rgb, min_area_ratio, morph_kernel_size)
    elif prompt_mode == "full_box":
        h, w = image.shape[:2]
        prompt_box = np.array([0, 0, w - 1, h - 1])
    
    # Run segmentation
    mask = st.session_state.predictor.predict(
        image,
        prompt_mode=prompt_mode,
        multimask_output=multimask_output,
        min_area_ratio=min_area_ratio,
        morph_kernel_size=morph_kernel_size
    )
    
    # Compute statistics
    stats = compute_mask_statistics(mask, mpp=None)
    
    # Create overlay visualization
    overlay = overlay_mask_on_image(
        image,
        mask,
        color=(255, 0, 0),
        alpha=0.5
    )
    
    # Create prompt box visualization if available
    img_with_box = None
    if prompt_box is not None:
        img_with_box = image.copy()
        x1, y1, x2, y2 = prompt_box.astype(int)
        img_with_box = cv2.rectangle(img_with_box, (x1, y1), (x2, y2), 
                                    PROMPT_BOX_COLOR, PROMPT_BOX_THICKNESS)
        label_y = max(y1 - PROMPT_BOX_LABEL_Y_OFFSET, PROMPT_BOX_LABEL_MIN_Y)
        cv2.putText(img_with_box, f"Prompt: {prompt_mode}", 
                   (x1, label_y), PROMPT_BOX_LABEL_FONT, PROMPT_BOX_LABEL_SCALE,
                   PROMPT_BOX_COLOR, PROMPT_BOX_LABEL_THICKNESS)
    
    # Build and return result dict
    result = {
        'image': image,
        'mask': mask,
        'statistics': stats,
        'overlay': overlay,
        'prompt_box': prompt_box,
        'img_with_box': img_with_box,
        'prompt_mode': prompt_mode,
        'timestamp': datetime.now().isoformat()
    }
    
    return result


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
        
        # Automatically populate session_state.images from uploaded files
        # Create unique IDs based on filename and file size
        uploaded_ids = set()
        existing_ids = {img['id'] for img in st.session_state.images}  # O(1) lookup
        
        for uploaded_file in uploaded_files:
            file_id = f"{uploaded_file.name}_{uploaded_file.size}"
            uploaded_ids.add(file_id)
            
            # Check if this image is already in the list (O(1) lookup)
            if file_id not in existing_ids:
                # Read bytes
                image_bytes = uploaded_file.read()
                uploaded_file.seek(0)  # Reset file pointer
                
                # Add to images list
                st.session_state.images.append({
                    'id': file_id,
                    'name': uploaded_file.name,
                    'bytes': image_bytes,
                    'np_rgb_uint8': None,
                    'status': 'ready',  # ready, processing, done, failed, skipped
                    'error': None,
                    'result': None,
                    'include': True  # Whether to include in batch processing
                })
        
        # Remove images that are no longer in uploaded_files
        st.session_state.images = [
            img for img in st.session_state.images 
            if img['id'] in uploaded_ids
        ]
        
        # Display uploaded images with status
        st.subheader("📊 Uploaded Images")
        
        if st.session_state.images:
            # Create table data
            table_data = []
            for i, img in enumerate(st.session_state.images):
                # Preview thumbnail
                try:
                    if img['np_rgb_uint8'] is None and img['status'] != 'processing':
                        preview_img = load_image_from_bytes(img['bytes'])
                        # Create small thumbnail for preview (100px height)
                        h, w = preview_img.shape[:2]
                        thumb_h = 100
                        thumb_w = int(w * thumb_h / h)
                        thumbnail = cv2.resize(preview_img, (thumb_w, thumb_h))
                    else:
                        thumbnail = img.get('np_rgb_uint8')
                        if thumbnail is not None:
                            h, w = thumbnail.shape[:2]
                            thumb_h = 100
                            thumb_w = int(w * thumb_h / h)
                            thumbnail = cv2.resize(thumbnail, (thumb_w, thumb_h))
                except Exception as e:
                    # Handle image loading/processing errors gracefully
                    thumbnail = None
                
                # Get dimensions
                try:
                    if img['np_rgb_uint8'] is not None:
                        dims = f"{img['np_rgb_uint8'].shape[1]} × {img['np_rgb_uint8'].shape[0]}"
                    else:
                        temp_img = load_image_from_bytes(img['bytes'])
                        dims = f"{temp_img.shape[1]} × {temp_img.shape[0]}"
                except Exception as e:
                    # Handle image loading errors gracefully
                    dims = "Unknown"
                
                table_data.append({
                    'index': i,
                    'name': img['name'],
                    'dimensions': dims,
                    'status': img['status'],
                    'thumbnail': thumbnail
                })
            
            # Display table
            for row in table_data:
                col1, col2, col3, col4, col5 = st.columns([1, 3, 2, 2, 1])
                
                with col1:
                    # Include checkbox
                    include = st.checkbox(
                        "Include",
                        value=st.session_state.images[row['index']]['include'],
                        key=f"include_{row['index']}",
                        label_visibility="collapsed"
                    )
                    st.session_state.images[row['index']]['include'] = include
                
                with col2:
                    st.write(f"**{row['name']}**")
                
                with col3:
                    st.write(row['dimensions'])
                
                with col4:
                    # Status badge
                    status = row['status']
                    if status == 'ready':
                        st.write("⏳ Ready")
                    elif status == 'processing':
                        st.write("⚙️ Processing...")
                    elif status == 'done':
                        st.write("✅ Done")
                    elif status == 'failed':
                        st.write("❌ Failed")
                    elif status == 'skipped':
                        st.write("⊘ Skipped")
                
                with col5:
                    # Thumbnail preview
                    if row['thumbnail'] is not None:
                        st.image(row['thumbnail'], width=50)
            
            st.markdown("---")
            
            # Clear uploads button
            if st.button("🗑️ Clear Uploads", type="secondary"):
                st.session_state.images = []
                st.rerun()
        
        st.info("👉 Go to **Analysis** tab to process your images")
        
    else:
        st.info("💡 Please upload one or more images to get started")
        
        # If there are no uploaded files but images exist, clear them
        if st.session_state.images:
            st.session_state.images = []



def analysis_page():
    """Analysis interface with MedSAM segmentation - Multi-image queue support"""
    st.title("🤖 AI-Powered Analysis")
    
    # Check if we're in local mode and have images in queue
    is_local_mode = st.session_state.local_mode
    
    if is_local_mode:
        # Local mode with multi-image queue
        if not st.session_state.images:
            st.warning("⚠️ Please upload images first in the Image Upload tab")
            return
        
        st.info(f"📊 Image Queue: {len(st.session_state.images)} image(s)")
        
        # Analysis settings (common for all images)
        st.subheader("⚙️ Analysis Settings")
        
        col_set1, col_set2 = st.columns(2)
        
        with col_set1:
            prompt_mode = st.selectbox(
                "Prompt Mode",
                options=["auto_box", "full_box", "point"],
                index=0,
                help="auto_box: Auto-detect tissue region; full_box: Use entire image; point: Use center point"
            )
        
        with col_set2:
            multimask_output = st.checkbox(
                "Multi-mask Output",
                value=False,
                help="Generate multiple mask predictions and select the best one"
            )
        
        # Advanced settings in expander
        with st.expander("Advanced Segmentation Settings"):
            col_a, col_b = st.columns(2)
            with col_a:
                min_area_ratio = st.slider(
                    "Min Area Ratio",
                    min_value=0.001,
                    max_value=0.1,
                    value=0.01,
                    step=0.001,
                    format="%.3f",
                    help="Minimum area ratio for tissue detection (used in auto_box mode)"
                )
            with col_b:
                morph_kernel_size = st.slider(
                    "Morph Kernel Size",
                    min_value=3,
                    max_value=15,
                    value=5,
                    step=2,
                    help="Kernel size for morphological operations (used in auto_box mode)"
                )
        
        st.markdown("---")
        
        # Control buttons
        st.subheader("🎮 Controls")
        
        col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
        
        with col_btn1:
            run_next = st.button("▶️ Run Next", type="primary", disabled=st.session_state.batch_running)
        
        with col_btn2:
            run_batch = st.button("⏩ Run Batch", type="primary", disabled=st.session_state.batch_running)
        
        with col_btn3:
            if st.session_state.batch_running:
                stop_batch = st.button("⏸️ Stop Batch", type="secondary")
            else:
                stop_batch = False
        
        with col_btn4:
            clear_results = st.button("🗑️ Clear Results", type="secondary", disabled=st.session_state.batch_running)
        
        # Handle button actions
        if run_next:
            # Find next ready or skipped image that is included
            next_item = None
            for item in st.session_state.images:
                if item['include'] and item['status'] in ['ready', 'skipped']:
                    next_item = item
                    break
            
            if next_item:
                next_item['status'] = 'processing'
                st.rerun()
        
        if run_batch:
            st.session_state.batch_running = True
            st.session_state.batch_index = 0
            st.rerun()
        
        if stop_batch:
            st.session_state.batch_running = False
            # Mark any processing items as skipped
            for item in st.session_state.images:
                if item['status'] == 'processing':
                    item['status'] = 'skipped'
            st.rerun()
        
        if clear_results:
            for item in st.session_state.images:
                if item['status'] in ['done', 'failed']:
                    item['status'] = 'ready'
                    item['result'] = None
                    item['error'] = None
            st.rerun()
        
        # Batch processing logic
        if st.session_state.batch_running:
            # Find the next image to process
            processing_item = None
            for item in st.session_state.images:
                if item['include'] and item['status'] == 'processing':
                    processing_item = item
                    break
            
            if processing_item is None:
                # Find next ready item
                for item in st.session_state.images:
                    if item['include'] and item['status'] == 'ready':
                        item['status'] = 'processing'
                        processing_item = item
                        break
            
            if processing_item:
                # Process this item
                st.info(f"⏳ Processing: {processing_item['name']}")
                try:
                    result = run_analysis_on_item(
                        processing_item,
                        prompt_mode=prompt_mode,
                        multimask_output=multimask_output,
                        min_area_ratio=min_area_ratio,
                        morph_kernel_size=morph_kernel_size
                    )
                    processing_item['result'] = result
                    processing_item['status'] = 'done'
                    st.success(f"✅ Completed: {processing_item['name']}")
                except Exception as e:
                    processing_item['status'] = 'failed'
                    processing_item['error'] = str(e)
                    st.error(f"❌ Failed: {processing_item['name']}: {str(e)}")
                
                # Check if there are more items to process
                has_more = any(item['include'] and item['status'] == 'ready' for item in st.session_state.images)
                if has_more:
                    # Continue batch
                    st.rerun()
                else:
                    # Batch complete
                    st.session_state.batch_running = False
                    st.success("🎉 Batch processing complete!")
                    st.rerun()
            else:
                # No more items to process
                st.session_state.batch_running = False
                st.success("🎉 Batch processing complete!")
                st.rerun()
        
        # Single item processing (Run Next button)
        if not st.session_state.batch_running:
            processing_item = None
            for item in st.session_state.images:
                if item['status'] == 'processing':
                    processing_item = item
                    break
            
            if processing_item:
                st.info(f"⏳ Processing: {processing_item['name']}")
                try:
                    result = run_analysis_on_item(
                        processing_item,
                        prompt_mode=prompt_mode,
                        multimask_output=multimask_output,
                        min_area_ratio=min_area_ratio,
                        morph_kernel_size=morph_kernel_size
                    )
                    processing_item['result'] = result
                    processing_item['status'] = 'done'
                    st.success(f"✅ Completed: {processing_item['name']}")
                except Exception as e:
                    processing_item['status'] = 'failed'
                    processing_item['error'] = str(e)
                    st.error(f"❌ Failed: {processing_item['name']}: {str(e)}")
        
        st.markdown("---")
        
        # Display queue status
        st.subheader("📋 Queue Status")
        
        for i, item in enumerate(st.session_state.images):
            with st.expander(f"{i+1}. {item['name']} - {item['status'].upper()}", expanded=(item['status'] in ['processing', 'done'])):
                if item['status'] == 'done' and item['result']:
                    # Display results
                    result = item['result']
                    
                    # Statistics
                    st.write("**Statistics**")
                    col1, col2, col3 = st.columns(3)
                    stats = result['statistics']
                    
                    with col1:
                        st.metric("Positive Pixels", f"{stats['num_positive_pixels']:,}")
                    with col2:
                        st.metric("Coverage", f"{stats['coverage_percent']:.2f}%")
                    with col3:
                        if 'area_mm2' in stats:
                            st.metric("Area", f"{stats['area_mm2']:.4f} mm²")
                    
                    # Visualizations
                    st.write("**Visualizations**")
                    
                    # Create binary mask for visualization
                    mask = result['mask']
                    if mask.dtype == np.bool_:
                        mask_bin = mask
                    else:
                        unique_vals = np.unique(mask)
                        if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, True, False}):
                            mask_bin = mask.astype(bool)
                        else:
                            mask_bin = mask > 0.5
                    
                    mask_vis = (mask_bin.astype(np.uint8)) * 255
                    
                    if result.get('img_with_box') is not None:
                        col1, col2, col3, col4 = st.columns(4)
                        with col4:
                            st.image(result['img_with_box'], caption="Prompt Box", width=200)
                    else:
                        col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.image(result['image'], caption="Original", width=200)
                    with col2:
                        st.image(mask_vis, caption="Mask", clamp=True, width=200)
                    with col3:
                        st.image(result['overlay'], caption="Overlay", width=200)
                    
                elif item['status'] == 'failed':
                    st.error(f"**Error:** {item['error']}")
                    if st.button(f"Retry {item['name']}", key=f"retry_{i}"):
                        item['status'] = 'ready'
                        item['error'] = None
                        st.rerun()
                
                elif item['status'] == 'processing':
                    st.info("⏳ Processing in progress...")
                
                elif item['status'] == 'ready':
                    st.info("⏳ Ready for processing")
                
                elif item['status'] == 'skipped':
                    st.warning("⊘ Skipped")
        
    else:
        # Original Halo mode logic (keep existing for backward compatibility)
        if st.session_state.selected_slide is None:
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
        
        # Segmentation prompt settings
        st.write("**Segmentation Prompt Mode**")
        prompt_mode = st.selectbox(
            "Prompt Mode",
            options=["auto_box", "full_box", "point"],
            index=0,
            help="auto_box: Auto-detect tissue region; full_box: Use entire image; point: Use center point"
        )
        
        # Advanced settings in expander
        with st.expander("Advanced Segmentation Settings"):
            col_a, col_b = st.columns(2)
            with col_a:
                min_area_ratio = st.slider(
                    "Min Area Ratio",
                    min_value=0.001,
                    max_value=0.1,
                    value=0.01,
                    step=0.001,
                    format="%.3f",
                    help="Minimum area ratio for tissue detection (used in auto_box mode)"
                )
                morph_kernel_size = st.slider(
                    "Morph Kernel Size",
                    min_value=3,
                    max_value=15,
                    value=5,
                    step=2,
                    help="Kernel size for morphological operations (used in auto_box mode)"
                )
            with col_b:
                multimask_output = st.checkbox(
                    "Multi-mask Output",
                    value=False,
                    help="Generate multiple mask predictions and select the best one"
                )
        
        use_prompts = st.checkbox("Use point/box prompts", value=False, 
                                 help="Enable interactive prompts for segmentation")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.button("🚀 Run Analysis", type="primary"):
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
                            st.session_state.predictor = UtilsMedSAMPredictor(
                                Config.MEDSAM_CHECKPOINT,
                                model_type=Config.MODEL_TYPE,
                                device=Config.DEVICE
                            )
                        
                        # Run inference directly on original image (no preprocessing)
                        st.info(f"⏳ Running MedSAM segmentation with {prompt_mode} prompt...")
                        
                        # Compute prompt box for visualization
                        prompt_box = None
                        if prompt_mode == "auto_box":
                            image_rgb = _ensure_rgb_uint8(image)
                            prompt_box = _compute_tissue_bbox(image_rgb, min_area_ratio, morph_kernel_size)
                            st.info(f"📦 Detected tissue box: {prompt_box}")
                        elif prompt_mode == "full_box":
                            h, w = image.shape[:2]
                            prompt_box = np.array([0, 0, w - 1, h - 1])
                            st.info(f"📦 Using full image box: {prompt_box}")
                        
                        mask = st.session_state.predictor.predict(
                            image,
                            prompt_mode=prompt_mode,
                            multimask_output=multimask_output,
                            min_area_ratio=min_area_ratio,
                            morph_kernel_size=morph_kernel_size
                        )
                        
                        # Store mask directly (already at original image size)
                        st.session_state.current_mask = mask
                        
                        # Compute statistics
                        mpp = slide.get('mpp')
                        stats = compute_mask_statistics(mask, mpp)
                        
                        # Store results
                        st.session_state.analysis_results = {
                            'image': image,
                            'mask': mask,
                            'roi': (x, y, width, height),
                            'statistics': stats,
                            'slide_id': slide['id'],
                            'slide_name': slide['name'],
                            'timestamp': datetime.now().isoformat(),
                            'prompt_box': prompt_box,
                            'prompt_mode': prompt_mode
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
            
            # Debug information for mask
            mask = results['mask']
            st.write("**Debug Info:**")
            st.write(f"Prompt mode: {results.get('prompt_mode', 'unknown')}")
            if results.get('prompt_box') is not None:
                prompt_box = results['prompt_box']
                st.write(f"Prompt box: [{prompt_box[0]}, {prompt_box[1]}, {prompt_box[2]}, {prompt_box[3]}]")
                box_area = (prompt_box[2] - prompt_box[0]) * (prompt_box[3] - prompt_box[1])
                img_area = mask.shape[0] * mask.shape[1]
                st.write(f"Prompt box area: {box_area:,} pixels ({100*box_area/img_area:.1f}% of image)")
            st.write(f"Mask dtype: {mask.dtype}, shape: {mask.shape}")
            st.write(f"Mask min/max: {float(np.min(mask))}, {float(np.max(mask))}")
            
            # Create binary mask - handle both boolean and numeric masks
            # For boolean masks, use directly; for numeric, threshold at 0.5
            if mask.dtype == np.bool_:
                mask_bin = mask
            else:
                # Check if already binary (0/1) or needs thresholding
                unique_vals = np.unique(mask)
                if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, True, False}):
                    mask_bin = mask.astype(bool)  # Already binary
                else:
                    mask_bin = mask > 0.5  # Threshold probabilistic output
            
            st.write(f"Binary mask unique values: {np.unique(mask_bin)}")
            st.write(f"Binary mask sum (positive pixels): {int(mask_bin.sum())}")
            
            # Create prompt box overlay for visualization if available
            if results.get('prompt_box') is not None:
                prompt_box = results['prompt_box']
                img_with_box = results['image'].copy()
                # Draw rectangle on image
                x1, y1, x2, y2 = prompt_box.astype(int)
                img_with_box = cv2.rectangle(img_with_box, (x1, y1), (x2, y2), 
                                            PROMPT_BOX_COLOR, PROMPT_BOX_THICKNESS)
                # Add text label
                label_y = max(y1 - PROMPT_BOX_LABEL_Y_OFFSET, PROMPT_BOX_LABEL_MIN_Y)
                cv2.putText(img_with_box, f"Prompt: {results.get('prompt_mode', 'box')}", 
                           (x1, label_y), PROMPT_BOX_LABEL_FONT, PROMPT_BOX_LABEL_SCALE,
                           PROMPT_BOX_COLOR, PROMPT_BOX_LABEL_THICKNESS)
            
            # Display images in columns
            if results.get('prompt_box') is not None:
                col1, col2, col3, col4 = st.columns(4)
            else:
                col1, col2, col3 = st.columns(3)
            
            with col1:
                st.image(results['image'], caption="Original Image")
            
            with col2:
                # Display binary mask properly - convert to uint8 for visualization
                # Use explicit parentheses for clarity
                mask_vis = (mask_bin.astype(np.uint8)) * 255
                st.image(mask_vis, caption="Segmentation Mask (binary)", clamp=True)
            
            with col3:
                overlay = overlay_mask_on_image(
                    results['image'],
                    results['mask'],
                    color=(255, 0, 0),
                    alpha=0.5
                )
                st.image(overlay, caption="Overlay")
            
            if results.get('prompt_box') is not None:
                with col4:
                    st.image(img_with_box, caption="Prompt Box")


def export_page():
    """Export results to GeoJSON format"""
    st.title("📤 Export Results")
    
    if st.session_state.analysis_results is None:
        st.warning("⚠️ No analysis results to export. Please run analysis first.")
        return
    
    results = st.session_state.analysis_results
    
    st.success(f"✅ Results ready for export from: **{results['slide_name']}**")
    
    # Export settings
    st.subheader("⚙️ Export Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        classification = st.text_input(
            "Classification Label",
            value="tissue_segmentation",
            help="Label for the segmented regions"
        )
        
        min_area = st.number_input(
            "Minimum Polygon Area (pixels)",
            min_value=1,
            value=Config.MIN_POLYGON_AREA,
            help="Filter out small polygons"
        )
    
    with col2:
        simplify = st.checkbox(
            "Simplify Polygons",
            value=True,
            help="Reduce polygon complexity"
        )
        
        if simplify:
            tolerance = st.slider(
                "Simplification Tolerance",
                min_value=0.1,
                max_value=5.0,
                value=Config.SIMPLIFY_TOLERANCE,
                help="Higher = more simplified"
            )
        else:
            tolerance = 1.0
    
    st.markdown("---")
    
    if st.button("🔄 Generate GeoJSON", type="primary", ):
        with st.spinner("Converting mask to GeoJSON..."):
            try:
                # Convert mask to polygons
                st.info("⏳ Extracting polygons from mask...")
                polygons = mask_to_polygons(results['mask'], min_area=min_area)
                
                if len(polygons) == 0:
                    st.warning("⚠️ No polygons found. Try reducing minimum area.")
                    return
                
                # Create GeoJSON
                st.info(f"⏳ Creating GeoJSON with {len(polygons)} features...")
                geojson = polygons_to_geojson(
                    polygons,
                    properties={"classification": classification}
                )
                
                # Save to file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"annotations_{results['slide_name']}_{timestamp}.geojson"
                output_path = Config.get_temp_path(filename)
                
                with open(output_path, 'w') as f:
                    json.dump(geojson, f, indent=2)
                
                # Store in session state
                st.session_state.geojson = geojson
                st.session_state.geojson_path = output_path
                
                st.success(f"✅ Exported {len(polygons)} polygons to GeoJSON!")
                
            except Exception as e:
                st.error(f"❌ Export failed: {str(e)}")
                st.code(traceback.format_exc())
    
    # Display and download
    if 'geojson' in st.session_state:
        st.markdown("---")
        st.subheader("📄 GeoJSON Preview")
        
        # Statistics
        num_features = len(st.session_state.geojson['features'])
        st.metric("Number of Features", num_features)
        
        # Preview
        with st.expander("View GeoJSON"):
            st.json(st.session_state.geojson)
        
        # Download button
        with open(st.session_state.geojson_path, 'r') as f:
            geojson_str = f.read()
        
        st.download_button(
            label="💾 Download GeoJSON",
            data=geojson_str,
            file_name=st.session_state.geojson_path.name,
            mime="application/json"
        )
        
        st.info("""
        **Next Steps:**
        1. Download the GeoJSON file
        2. Open your slide in Halo (or view in GIS software)
        3. Import the GeoJSON as annotations
        4. Visualize and refine results
        """)


def import_page():
    """Import annotations to Halo (optional feature)"""
    st.title("📥 Import to Halo")
    
    st.info("🚧 This feature requires additional Halo API permissions")
    
    st.markdown("""
    ### Manual Import Instructions
    
    1. **Download GeoJSON** from the Export page
    2. **Open Halo** and navigate to your slide
    3. **Import Annotations**: 
       - File → Import → GeoJSON
       - Select the downloaded file
    4. **Review and Save** the imported annotations
    
    ### Programmatic Import (Coming Soon)
    
    Automatic upload of annotations via API will be available in a future release.
    """)


def main():
    """Main application"""
    st.sidebar.title("🔬 XHaloPathAnalyzer")
    st.sidebar.markdown("---")
    
    if not st.session_state.authenticated:
        # Show only authentication
        authentication_page()
    else:
        # Show connection status
        if st.session_state.local_mode:
            st.sidebar.success("✅ Local Mode Active")
        else:
            st.sidebar.success("✅ Connected to Halo")
        
        # Determine navigation options based on mode
        if st.session_state.local_mode:
            nav_options = [
                "📤 Image Upload",
                "🤖 Analysis",
                "📥 Export",
                "⚙️ Settings"
            ]
        else:
            nav_options = [
                "🔬 Slide Selection",
                "🤖 Analysis",
                "📤 Export",
                "📥 Import",
                "⚙️ Settings"
            ]
        
        page = st.sidebar.radio("Navigation", nav_options)
        
        st.sidebar.markdown("---")
        
        # Show current slide/image info
        if st.session_state.selected_slide:
            st.sidebar.info(f"**Current {'Image' if st.session_state.local_mode else 'Slide'}:**\n{st.session_state.selected_slide['name']}")
        
        # Logout/Exit button
        if st.sidebar.button("🚪 Exit to Start"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        # Route to pages
        if page == "📤 Image Upload":
            image_upload_page()
        elif page == "🔬 Slide Selection":
            slide_selection_page()
        elif page == "🤖 Analysis":
            analysis_page()
        elif page == "📤 Export" or page == "📥 Export":
            export_page()
        elif page == "📥 Import":
            import_page()
        elif page == "⚙️ Settings":
            st.title("⚙️ Settings")
            st.info("Configuration settings coming soon")
            st.write(f"**Device:** {Config.DEVICE}")
            st.write(f"**Model:** {Config.MODEL_TYPE}")
            st.write(f"**Checkpoint:** {Config.MEDSAM_CHECKPOINT}")
            st.write(f"**Mode:** {'Local' if st.session_state.local_mode else 'Halo API'}")
            
            # Halo Link Debug Section
            st.markdown("---")
            st.subheader("🔗 Halo Link Integration")
            
            if st.button("Run Halo Link Smoke Test"):
                with st.spinner("Running Halo Link smoke test..."):
                    try:
                        from xhalo.halolink.smoketest import run_smoke_test
                        
                        # Run smoke test
                        results = run_smoke_test(verbose=False)
                        
                        if results["success"]:
                            st.success("✓ Halo Link smoke test passed!")
                        else:
                            st.error(f"✗ Halo Link smoke test failed: {results.get('error', 'Unknown error')}")
                        
                        # Display step-by-step results
                        st.markdown("#### Test Results")
                        for step_name, step_result in results.get("steps", {}).items():
                            if step_result.get("success"):
                                if step_result.get("skipped"):
                                    st.info(f"⊘ {step_name}: {step_result.get('reason', 'Skipped')}")
                                else:
                                    st.success(f"✓ {step_name}")
                            else:
                                st.error(f"✗ {step_name}: {step_result.get('error', 'Failed')}")
                        
                        # Show detailed results in expander
                        with st.expander("View Detailed Results"):
                            st.json(results)
                            
                    except ImportError as e:
                        st.error(f"Halo Link module not available: {e}")
                    except Exception as e:
                        st.error(f"Error running smoke test: {e}")
                        logger.exception("Halo Link smoke test error")
            
            # Show current configuration
            st.markdown("#### Current Configuration")
            halolink_config = {
                "HALOLINK_BASE_URL": Config.HALOLINK_BASE_URL or "(not set)",
                "HALOLINK_GRAPHQL_URL": Config.HALOLINK_GRAPHQL_URL or "(not set)",
                "HALOLINK_GRAPHQL_PATH": Config.HALOLINK_GRAPHQL_PATH or "(not set)",
                "HALOLINK_CLIENT_ID": "***" if Config.HALOLINK_CLIENT_ID else "(not set)",
                "HALOLINK_CLIENT_SECRET": "***" if Config.HALOLINK_CLIENT_SECRET else "(not set)",
                "HALOLINK_SCOPE": Config.HALOLINK_SCOPE or "(not set)",
            }
            
            for key, value in halolink_config.items():
                st.text(f"{key}: {value}")



if __name__ == "__main__":
    main()
