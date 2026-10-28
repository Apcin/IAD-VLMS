"""
Evaluation Results Viewer with Image Display
Builds image paths from directories - no need to re-run evaluations!
Run: streamlit run results_viewer.py
"""
import streamlit as st
import json
import pandas as pd
import plotly.express as px
from pathlib import Path
from PIL import Image
import os
import random

# Page Config
st.set_page_config(layout="wide", page_title="Model Evaluation Results")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Server paths (update if running locally)
RESULTS_DIR = Path("/bulk/aacudad/Training")
REALIAD_IMAGES = Path("/bulk/aacudad/reasoning_traces_gen/data/Real-IAD/images")
MMAD_IMAGES = Path("/bulk/aacudad/reasoning_traces_gen/data/MMAD")

# For local Windows, uncomment and update:
# RESULTS_DIR = Path("C:/Users/Gebruiker/Desktop/Results")
# REALIAD_IMAGES = Path("C:/Users/Gebruiker/Desktop/images/Real-IAD/images")
# MMAD_IMAGES = Path("C:/Users/Gebruiker/Desktop/images/MMAD")

FILES = {
    "MCQ Only": "evaluate_mcq_only_results.json",
    "MCQ Extended": "evaluate_mcq_extended_results.json",
    "One-Shot 20k": "evaluate_oneshot_20k_results.json",
    "Reasoning 20k": "evaluate_20k_results.json"
}

# =============================================================================
# IMAGE MAPPING - Build from directories
# =============================================================================

@st.cache_data
def build_product_image_map():
    """
    Build a mapping: product_name -> list of (image_path, ref_image_path) tuples
    This scans the image directories to find images for each product.
    """
    product_images = {}
    
    # Real-IAD: Structure is images/{product}/{good or defect_type}/image.jpg
    if REALIAD_IMAGES.exists():
        for product_dir in REALIAD_IMAGES.iterdir():
            if product_dir.is_dir():
                product_name = product_dir.name.lower()
                product_images[product_name] = {'source': 'realiad', 'good': [], 'defect': []}
                
                for subdir in product_dir.iterdir():
                    if subdir.is_dir():
                        images = list(subdir.glob('*.jpg')) + list(subdir.glob('*.png')) + \
                                 list(subdir.glob('*.JPG')) + list(subdir.glob('*.PNG'))
                        
                        if subdir.name.lower() == 'good':
                            product_images[product_name]['good'].extend([str(p) for p in images])
                        else:
                            product_images[product_name]['defect'].extend([str(p) for p in images])
    
    # MMAD: Structure is {category}/{product}/...
    if MMAD_IMAGES.exists():
        for category_dir in MMAD_IMAGES.iterdir():
            if category_dir.is_dir() and category_dir.name not in ['annotations', '.git']:
                for product_dir in category_dir.iterdir():
                    if product_dir.is_dir():
                        product_name = product_dir.name.lower()
                        if product_name not in product_images:
                            product_images[product_name] = {'source': 'mmad', 'good': [], 'defect': []}
                        
                        # Look for good/test directories
                        good_dir = product_dir / 'good'
                        test_dir = product_dir / 'test'
                        
                        if good_dir.exists():
                            for img in good_dir.rglob('*'):
                                if img.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                                    product_images[product_name]['good'].append(str(img))
                        
                        if test_dir.exists():
                            for img in test_dir.rglob('*'):
                                if img.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                                    product_images[product_name]['defect'].append(str(img))
    
    return product_images

def get_sample_images(product_name, source, product_map, seed=None):
    """Get a sample reference and target image for a product."""
    product_key = product_name.lower()
    
    # Try exact match first
    if product_key not in product_map:
        # Try partial match
        for key in product_map:
            if product_key in key or key in product_key:
                product_key = key
                break
    
    if product_key not in product_map:
        return None, None
    
    data = product_map[product_key]
    
    ref_image = None
    target_image = None
    
    if seed:
        random.seed(seed)
    
    if data['good']:
        ref_image = random.choice(data['good'])
    
    if data['defect']:
        target_image = random.choice(data['defect'])
    elif data['good'] and len(data['good']) > 1:
        # If no defect images, use another good image
        target_image = random.choice([g for g in data['good'] if g != ref_image] or data['good'])
    
    return ref_image, target_image

def load_image_safe(path):
    """Safely load an image."""
    if path and os.path.exists(path):
        try:
            return Image.open(path)
        except Exception:
            pass
    return None

# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def load_results(filename):
    path = RESULTS_DIR / filename
    if not path.exists():
        return None
    with open(path, 'r') as f:
        return json.load(f)

def extract_results_list(data_node):
    """Traverse the dictionary to find the list of result objects."""
    if not isinstance(data_node, dict):
        return []
    
    if 'results' in data_node and isinstance(data_node['results'], list):
        return data_node['results']
    
    if 'zeroshot_mcq' in data_node:
        return extract_results_list(data_node['zeroshot_mcq'])
    
    if 'results' in data_node and isinstance(data_node['results'], dict):
        if 'is_defect' in data_node['results']:
            return extract_results_list(data_node['results']['is_defect'])
    
    return []

def calculate_stats(results):
    if not results:
        return None
    
    df = pd.DataFrame(results)
    
    cols = ['source', 'product', 'correct']
    for c in cols:
        if c not in df.columns:
            return None
    
    acc = df['correct'].mean()
    source_acc = df.groupby('source')['correct'].mean().to_dict()
    product_acc = df.groupby('product')['correct'].mean().to_dict()
    
    return {
        "accuracy": acc,
        "source_accuracy": source_acc,
        "product_accuracy": product_acc,
        "total": len(df),
        "df": df
    }

# =============================================================================
# MAIN APP
# =============================================================================

def main():
    st.title("🔍 Evaluation Results Analysis")
    
    # Build image map
    with st.spinner("Building image map from directories..."):
        product_map = build_product_image_map()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        show_images = st.checkbox("Show Images", value=True)
        num_examples = st.slider("Examples to show", 1, 20, 5)
        st.markdown("---")
        st.info(f"📁 Found {len(product_map)} products with images")
    
    # Load all models
    all_data = {}
    
    for display_name, fname in FILES.items():
        raw = load_results(fname)
        if raw:
            all_data[display_name] = {}
            
            for k in ['baseline', 'finetuned']:
                if k in raw:
                    res_list = extract_results_list(raw[k])
                    if res_list:
                        stats = calculate_stats(res_list)
                        if stats:
                            all_data[display_name][k.capitalize()] = stats
            
            if not all_data[display_name]:
                st.warning(f"Could not extract results for {display_name}")
        else:
            st.warning(f"File not found: {fname}")

    if not all_data:
        st.error("No data loaded.")
        return

    # ==========================================================================
    # COMPARISON TABLE
    # ==========================================================================
    st.header("📊 Overall Performance Comparison")
    
    comp_rows = []
    for model_name, sub_models in all_data.items():
        for sub_name, stats in sub_models.items():
            row = {
                "Model": model_name,
                "Config": sub_name,
                "Overall Accuracy": stats['accuracy'],
                "Total Samples": stats['total']
            }
            for src, acc in stats['source_accuracy'].items():
                row[f"{src.upper()}"] = acc
            comp_rows.append(row)
    
    comp_df = pd.DataFrame(comp_rows)
    if not comp_df.empty:
        format_dict = {"Overall Accuracy": "{:.2%}", "Total Samples": "{:,}"}
        if "MMAD" in comp_df.columns:
            format_dict["MMAD"] = "{:.2%}"
        if "REALIAD" in comp_df.columns:
            format_dict["REALIAD"] = "{:.2%}"
        
        st.dataframe(
            comp_df.style.format(format_dict).background_gradient(
                subset=["Overall Accuracy"], cmap="RdYlGn", vmin=0.4, vmax=1.0
            ),
            use_container_width=True
        )
        
        # Bar Chart
        st.subheader("📈 Accuracy Comparison")
        
        chart_data = []
        for model_name, sub_models in all_data.items():
            for sub_name, stats in sub_models.items():
                chart_data.append({
                    "Model": f"{model_name}",
                    "Config": sub_name,
                    "Accuracy": stats['accuracy']
                })
        
        chart_df = pd.DataFrame(chart_data)
        fig = px.bar(
            chart_df, x="Model", y="Accuracy", color="Config", 
            barmode="group", text_auto='.1%',
            color_discrete_map={"Baseline": "#636EFA", "Finetuned": "#00CC96"}
        )
        fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

    # ==========================================================================
    # DETAILED ANALYSIS
    # ==========================================================================
    st.header("🔬 Detailed Analysis")
    
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        selected_model = st.selectbox("Select Model", list(all_data.keys()))
    with col_sel2:
        selected_config = st.selectbox("Select Config", list(all_data[selected_model].keys()))
    
    stats = all_data[selected_model][selected_config]
    df = stats['df']
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall Accuracy", f"{stats['accuracy']:.2%}")
    with col2:
        incorrect_count = len(df[~df['correct']])
        st.metric("Incorrect", f"{incorrect_count} / {len(df)}")
    with col3:
        st.metric("Total Samples", len(df))
    
    # Product accuracy chart
    st.subheader("Accuracy by Product")
    prod_acc = df.groupby('product')['correct'].agg(['mean', 'count']).sort_values('mean')
    prod_acc.columns = ['Accuracy', 'Count']
    
    fig_prod = px.bar(
        prod_acc.reset_index(), x='product', y='Accuracy',
        color='Accuracy', color_continuous_scale='RdYlGn',
        hover_data=['Count']
    )
    fig_prod.update_layout(xaxis_tickangle=-45, yaxis_tickformat=".0%")
    st.plotly_chart(fig_prod, use_container_width=True)

    # ==========================================================================
    # EXAMPLE VIEWER WITH IMAGES
    # ==========================================================================
    st.header("🖼️ Example Viewer")
    
    col_filter1, col_filter2, col_filter3 = st.columns(3)
    with col_filter1:
        viz_type = st.radio("Show", ["All", "Correct Only", "Incorrect Only"], horizontal=True)
    with col_filter2:
        source_filter = st.multiselect(
            "Filter by Source", 
            options=df['source'].unique().tolist(),
            default=df['source'].unique().tolist()
        )
    with col_filter3:
        product_filter = st.multiselect(
            "Filter by Product",
            options=sorted(df['product'].unique().tolist()),
            default=[]
        )
    
    # Apply filters
    filtered_df = df[df['source'].isin(source_filter)]
    if product_filter:
        filtered_df = filtered_df[filtered_df['product'].isin(product_filter)]
    
    if viz_type == "Correct Only":
        filtered_df = filtered_df[filtered_df['correct'] == True]
    elif viz_type == "Incorrect Only":
        filtered_df = filtered_df[filtered_df['correct'] == False]
    
    if filtered_df.empty:
        st.info("No samples match the current filters.")
    else:
        sample_df = filtered_df.sample(min(num_examples, len(filtered_df)), random_state=42)
        
        for idx, row in sample_df.iterrows():
            with st.container():
                result_emoji = "✅" if row['correct'] else "❌"
                st.markdown(f"### {result_emoji} **{row['product']}** ({row['source'].upper()})")
                
                if show_images:
                    # Get images from product map
                    ref_path, target_path = get_sample_images(
                        row['product'], row['source'], product_map, seed=idx
                    )
                    
                    img_col1, img_col2 = st.columns(2)
                    
                    with img_col1:
                        st.caption("Reference (Normal)")
                        ref_img = load_image_safe(ref_path)
                        if ref_img:
                            st.image(ref_img, use_container_width=True)
                        else:
                            st.warning(f"No reference image found for {row['product']}")
                    
                    with img_col2:
                        st.caption("Target (Test)")
                        target_img = load_image_safe(target_path)
                        if target_img:
                            st.image(target_img, use_container_width=True)
                        else:
                            st.warning(f"No target image found for {row['product']}")
                
                # Prediction
                pred_color = "green" if row['correct'] else "red"
                st.markdown(f"**Predicted:** :{pred_color}[{row['predicted']}] | **Actual:** {row['actual']}")
                
                with st.expander("View Model Response"):
                    st.text(row.get('response', 'N/A'))
                
                st.markdown("---")

    # ==========================================================================
    # EXPORT
    # ==========================================================================
    st.header("📥 Export")
    
    if st.button("Export Filtered Results to CSV"):
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"{selected_model}_{selected_config}_results.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
