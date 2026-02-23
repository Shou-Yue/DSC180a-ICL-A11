import streamlit as st
import os
import sys
import torch
import google.genai as genai
from dotenv import load_dotenv
import subprocess
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from llm_providers import get_provider

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../icl_reproduction'))
from training import BinaryClassificationDataset

# 1. Load Environment Variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# 2. Configure Page Layout
st.set_page_config(page_title="Binary Classification Task Solver", layout="wide")

# --- RESULTS TRACKING SETUP ---
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "accuracy_results.jsonl"

def save_result(d, N, R, flip_prob, prediction, true_label, correct, accuracy_type="in-context"):
    """Save individual prediction result to JSONL file"""
    result = {
        "timestamp": datetime.now().isoformat(),
        "d": d,
        "N": N,
        "R": R,
        "flip_prob": flip_prob,
        "prediction": int(prediction),
        "true_label": int(true_label),
        "correct": bool(correct),
        "accuracy_type": accuracy_type
    }
    with open(RESULTS_FILE, 'a') as f:
        f.write(json.dumps(result) + '\n')

def load_results():
    """Load all results from JSONL file"""
    if not RESULTS_FILE.exists():
        return pd.DataFrame()
    
    results = []
    with open(RESULTS_FILE, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    return pd.DataFrame(results)

def calculate_accuracy_by_config(df):
    """Calculate accuracy grouped by dataset configuration"""
    if df.empty:
        return pd.DataFrame()
    
    return df.groupby(['d', 'N', 'R', 'flip_prob']).agg({
        'correct': ['sum', 'count', 'mean']
    }).round(4)

# 3. Sidebar for API Key Check
with st.sidebar:
    st.header("Status")
    
    # Provider selection
    provider = st.radio("Select LLM Provider", ["gemini", "claude", "gpt"])
    
    required_key = f"{provider.upper()}_API_KEY"
    api_key = os.getenv(required_key)
    
    if api_key:
        st.success(f"✅ {provider.upper()} API Key Detected")
    else:
        st.error(f"❌ {required_key} missing in .env file!")
        st.stop()
    
    # Tab selection
    selected_mode = st.radio("Select Mode", ["Single Test", "Batch Testing", "Results Dashboard"])

# Create main tabs
main_tabs = st.tabs(["Introduction", "Test with Commercial LLM"])

# --- HELPER FUNCTIONS ---
def format_dataset_for_prompt(context_x, context_y, query_x):
    """Format the dataset as a readable string for Gemini"""
    B, d = context_x.shape
    
    # Format context examples
    dataset_str = "Labeled Context Examples:\n"
    for i in range(B):
        features = [f"{context_x[i, j].item():.4f}" for j in range(d)]
        label = int(context_y[i].item())
        dataset_str += f"Example {i+1}: Features = [{', '.join(features)}], Label = {label}\n"
    
    # Format query point
    dataset_str += "\nQuery Point (unlabeled):\n"
    query_features = [f"{query_x[j].item():.4f}" for j in range(d)]
    dataset_str += f"Features = [{', '.join(query_features)}]\n"
    
    return dataset_str

def predict_on_dataset(dataset_str, provider_name="gemini"):
    """Send dataset to LLM and get prediction for query point"""
    
    prompt = f"""
    You are a binary classification model performing in-context learning.
    
    You will be given a dataset with labeled examples followed by a single unlabeled datapoint.
    Your task is to learn from the labeled examples and make a prediction on the query point.
    
    DATASET:
    {dataset_str}
    
    INSTRUCTIONS:
    1. Use the labeled examples to infer the classification pattern.
    2. Apply this pattern to predict the label of the query point.
    3. Output ONLY the predicted label (0 or 1).
    4. Do not include any explanation, reasoning, or additional text.
    5. Your response should be a single value: either 0 or 1.
    
    PREDICTION:
    """
    
    try:
        api_key = os.getenv(f"{provider_name.upper()}_API_KEY")
        if not api_key:
            raise ValueError(f"API key for {provider_name} not found in .env")
        
        provider = get_provider(provider_name, api_key)
        print(f"Consulting {provider_name.upper()} for prediction...")
        return provider.predict(prompt)
    except Exception as e:
        print(f"Error with {provider_name}: {str(e)}")
        raise

def run_single_test(d, N, R, flip_prob, provider_name="gemini"):
    """Run a single test and return result"""
    try:
        dataset = BinaryClassificationDataset(d=d, N=N, num_tasks=1, R=R, flip_prob=flip_prob)
        task = dataset[0]
        
        context_x = task['context_x']
        context_y = task['context_y']
        query_x = task['query_x']
        query_y = task['query_y']
        
        dataset_str = format_dataset_for_prompt(context_x, context_y, query_x)
        prediction = predict_on_dataset(dataset_str, provider_name=provider_name)
        
        true_label = int(query_y.item())
        if prediction in ['0', '1']:
            pred_label = int(prediction)
            is_correct = pred_label == true_label
            save_result(d, N, R, flip_prob, pred_label, true_label, is_correct, accuracy_type=provider_name)
            return {
                "d": d,
                "N": N,
                "R": R,
                "flip_prob": flip_prob,
                "prediction": pred_label,
                "true_label": true_label,
                "correct": is_correct,
                "provider": provider_name,
                "status": "success"
            }
        else:
            return {
                "d": d,
                "N": N,
                "R": R,
                "flip_prob": flip_prob,
                "status": "invalid_prediction",
                "prediction": prediction,
                "provider": provider_name
            }
    except Exception as e:
        return {
            "d": d,
            "N": N,
            "R": R,
            "flip_prob": flip_prob,
            "status": "error",
            "error": str(e),
            "provider": provider_name
        }

# ============================================
# INTRODUCTION TAB
# ============================================
with main_tabs[0]:
    st.title("🤖 Binary Classification Task Solver")
    
    st.header("Project Overview")
    st.markdown("""
    This application demonstrates **In-Context Learning (ICL)** with commercial Large Language Models (LLMs).
    
    ### What is In-Context Learning?
    In-context learning is the ability of language models to learn new tasks from a small number of examples 
    provided in the input prompt, without any parameter updates. This project evaluates how well commercial LLMs 
    perform binary classification tasks through in-context learning.
    
    ### How It Works
    1. **Generate Dataset**: Create synthetic binary classification datasets with configurable parameters:
       - **d**: Feature dimension (number of features)
       - **N**: Number of context examples (labeled examples shown to the LLM)
       - **R**: Distribution scale (affects the signal strength)
       - **flip_prob**: Label noise probability (probability of noisy labels)
    
    2. **In-Context Learning**: The LLM is given labeled context examples and asked to predict the label 
       of a query point based on the patterns learned from these examples.
    
    3. **Evaluation**: Measure the accuracy of different LLMs across various dataset configurations 
       to understand how different parameters affect ICL performance.
    
    ### Features
    - **Single Test Mode**: Test one configuration at a time with detailed feedback
    - **Batch Testing Mode**: Run parameter sweeps in parallel for comprehensive evaluation
    - **Results Dashboard**: Track and visualize accuracy metrics across configurations
    """)
    
    st.divider()
    st.header("Key Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **Feature Dimension (d)**
        - Controls the complexity of the classification task
        - Higher dimensions typically make the task harder
        """)
    
    with col2:
        st.info("""
        **Context Examples (N)**
        - Number of labeled examples for in-context learning
        - More examples generally improve prediction accuracy
        """)
    
    with col3:
        st.info("""
        **Signal Strength (R)**
        - Controls how separable the classes are
        - Higher R values indicate clearer patterns
        """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.warning("""
        **Label Noise (flip_prob)**
        - Probability that a label is flipped (incorrect)
        - Simulates real-world noisy data
        """)
    
    with col2:
        st.success("""
        **Start Testing**
        - Navigate to the "Test with Commercial LLM" tab to begin experiments
        - Select your preferred LLM provider in the sidebar
        """)

# ============================================
# TEST WITH COMMERCIAL LLM TAB
# ============================================
with main_tabs[1]:
    st.title("🤖 Binary Classification Task Solver")
    st.markdown("Using **in-context learning** to predict binary classification labels.")
    
    # ============================================
    # SINGLE TEST MODE
    # ============================================
    if selected_mode == "Single Test":
        st.header("Single Test Mode")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Configuration")
            d = st.slider("Feature Dimension (d):", 2, 50, 20)
            N = st.slider("Context Examples (N):", 5, 100, 40)
            R = st.slider("Distribution Scale (R):", 0.1, 5.0, 1.0)
            flip_prob = st.slider("Label Noise Probability:", 0.0, 0.5, 0.2, step=0.05)
            
            if st.button("🔄 Generate New Dataset"):
                st.session_state.dataset = BinaryClassificationDataset(
                    d=d, N=N, num_tasks=1, R=R, flip_prob=flip_prob
                )
                st.session_state.task_idx = 0
                st.success("Dataset generated!")
        
        # Initialize dataset if not exists
        if 'dataset' not in st.session_state:
            st.session_state.dataset = BinaryClassificationDataset(d=20, N=40, num_tasks=1)
            st.session_state.task_idx = 0
        
        # Get current task
        task = st.session_state.dataset[st.session_state.task_idx]
        context_x = task['context_x']
        context_y = task['context_y']
        query_x = task['query_x']
        query_y = task['query_y']
        
        # Display Dataset
        st.subheader("Binary Classification Dataset")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Context Examples:**")
            context_df = f"Features: shape {tuple(context_x.shape)}\nLabels: {context_y.tolist()}"
            st.code(context_df, language="text")
        
        with col2:
            st.write("**Query Point:**")
            query_info = f"Features: shape {tuple(query_x.shape)}\nTrue Label: {int(query_y.item())}"
            st.code(query_info, language="text")
        
        # Main Execution
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button(f"🚀 Get Prediction from {provider.upper()}", type="primary"):
                with st.spinner(f"{provider.upper()} is making a prediction..."):
                    try:
                        dataset_str = format_dataset_for_prompt(context_x, context_y, query_x)
                        prediction = predict_on_dataset(dataset_str, provider_name=provider)
                        
                        st.success("Prediction Complete!")
                        st.markdown(f"## {provider.upper()}'s Prediction")
                        st.markdown(f"### {prediction}")
                        
                        true_label = int(query_y.item())
                        if prediction in ['0', '1']:
                            pred_label = int(prediction)
                            is_correct = pred_label == true_label
                            
                            save_result(d, N, R, flip_prob, pred_label, true_label, is_correct, accuracy_type=provider)
                            
                            if is_correct:
                                st.success(f"✅ Correct! True label was {true_label}")
                            else:
                                st.error(f"❌ Incorrect. True label was {true_label}")
                        else:
                            st.warning(f"⚠️ Invalid prediction: {prediction}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        with col2:
            if st.button("📊 Show Full Dataset Prompt"):
                dataset_str = format_dataset_for_prompt(context_x, context_y, query_x)
                st.text_area(f"Dataset for {provider.upper()}:", value=dataset_str, height=250)

    # ============================================
    # BATCH TESTING MODE
    # ============================================
    elif selected_mode == "Batch Testing":
        st.header("⚡ Batch Testing Mode")
        st.markdown(f"Run multiple tests with parameter sweeps in parallel using **{provider.upper()}**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dimension (d) Range")
            d_min = st.number_input("Min d:", min_value=2, max_value=50, value=5)
            d_max = st.number_input("Max d:", min_value=2, max_value=50, value=20)
            d_step = st.number_input("Step d:", min_value=1, max_value=10, value=5)
        
        with col2:
            st.subheader("Context Examples (N) Range")
            N_min = st.number_input("Min N:", min_value=5, max_value=100, value=10)
            N_max = st.number_input("Max N:", min_value=5, max_value=100, value=50)
            N_step = st.number_input("Step N:", min_value=1, max_value=20, value=10)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Signal Strength (R) Range")
            R_min = st.number_input("Min R:", min_value=0.1, max_value=5.0, value=0.5)
            R_max = st.number_input("Max R:", min_value=0.1, max_value=5.0, value=2.0)
            R_step = st.number_input("Step R:", min_value=0.1, max_value=1.0, value=0.5)
        
        with col2:
            st.subheader("Noise (flip_prob) Range")
            flip_min = st.number_input("Min flip_prob:", min_value=0.0, max_value=0.5, value=0.0, step=0.05)
            flip_max = st.number_input("Max flip_prob:", min_value=0.0, max_value=0.5, value=0.2, step=0.05)
            flip_step = st.number_input("Step flip_prob:", min_value=0.05, max_value=0.5, value=0.1, step=0.05)
        
        col1, col2 = st.columns(2)
        
        with col1:
            num_workers = st.number_input("Number of Parallel Workers:", min_value=1, max_value=8, value=2)
        
        with col2:
            tests_per_config = st.number_input("Tests per Configuration:", min_value=1, max_value=10, value=1)
        
        # Generate parameter grid
        d_values = list(range(int(d_min), int(d_max) + 1, int(d_step)))
        N_values = list(range(int(N_min), int(N_max) + 1, int(N_step)))
        R_values = [round(r, 1) for r in [R_min + i * R_step for i in range(int((R_max - R_min) / R_step) + 1)]]
        flip_values = [round(f, 2) for f in [flip_min + i * flip_step for i in range(int((flip_max - flip_min) / flip_step) + 1)]]
        
        total_configs = len(d_values) * len(N_values) * len(R_values) * len(flip_values) * tests_per_config
        
        st.info(f"📋 Total configurations to test: **{total_configs}**")
        
        if st.button("🚀 Start Batch Testing", type="primary"):
            # Create parameter list
            test_params = []
            for d in d_values:
                for N in N_values:
                    for R in R_values:
                        for flip_prob in flip_values:
                            for _ in range(int(tests_per_config)):
                                test_params.append((d, N, R, flip_prob))
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            results_container = st.container()
            
            successful_tests = 0
            failed_tests = 0
            results_list = []
            
            start_time = time.time()
            
            # Run tests in parallel
            with ThreadPoolExecutor(max_workers=int(num_workers)) as executor:
                future_to_params = {
                    executor.submit(run_single_test, d, N, R, flip_prob, provider_name=provider): (d, N, R, flip_prob)
                    for d, N, R, flip_prob in test_params
                }
                
                for i, future in enumerate(as_completed(future_to_params)):
                    result = future.result()
                    results_list.append(result)
                    
                    if result['status'] == 'success':
                        successful_tests += 1
                    else:
                        failed_tests += 1
                    
                    # Update progress
                    progress = (i + 1) / len(test_params)
                    progress_bar.progress(progress)
                    
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    remaining = (len(test_params) - i - 1) / rate if rate > 0 else 0
                    
                    status_text.text(
                        f"Progress: {i + 1}/{len(test_params)} | "
                        f"✅ {successful_tests} | ❌ {failed_tests} | "
                        f"⏱️ {remaining:.0f}s remaining"
                    )
            
            # Display results
            elapsed_total = time.time() - start_time
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Tests", len(test_params))
            col2.metric("Successful", successful_tests)
            col3.metric("Failed", failed_tests)
            col4.metric("Time (min)", f"{elapsed_total / 60:.1f}")
            
            # Calculate batch accuracy
            successful_results = [r for r in results_list if r['status'] == 'success']
            if successful_results:
                batch_accuracy = sum(1 for r in successful_results if r['correct']) / len(successful_results) * 100
                st.success(f"✨ Batch Accuracy: **{batch_accuracy:.2f}%** ({sum(1 for r in successful_results if r['correct'])}/{len(successful_results)})")
            
            st.divider()
            st.subheader("Detailed Results")
            results_df = pd.DataFrame(successful_results)
            if not results_df.empty:
                st.dataframe(results_df)
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Batch Results (CSV)",
                    data=csv,
                    file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

    # ============================================
    # RESULTS DASHBOARD
    # ============================================
    elif selected_mode == "Results Dashboard":
        st.header("📊 Results Dashboard")
        
        results_df = load_results()
        
        if not results_df.empty:
            # Overall metrics
            col1, col2, col3, col4 = st.columns(4)
            total_tests = len(results_df)
            total_correct = results_df['correct'].sum()
            overall_accuracy = (total_correct / total_tests * 100) if total_tests > 0 else 0
            
            col1.metric("Total Predictions", total_tests)
            col2.metric("Correct Predictions", int(total_correct))
            col3.metric("Overall Accuracy", f"{overall_accuracy:.2f}%")
            col4.metric("Incorrect Predictions", int(total_tests - total_correct))
            
            st.divider()
            
            # Accuracy by configuration
            st.subheader("Accuracy by Configuration")
            accuracy_by_config = calculate_accuracy_by_config(results_df)
            if not accuracy_by_config.empty:
                st.dataframe(accuracy_by_config.style.highlight_max(axis=0))
            
            st.divider()
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Accuracy vs Context Examples (N)")
                n_accuracy = results_df.groupby('N')['correct'].agg(['sum', 'count'])
                n_accuracy['accuracy'] = (n_accuracy['sum'] / n_accuracy['count'] * 100)
                st.line_chart(n_accuracy['accuracy'])
            
            with col2:
                st.subheader("Accuracy vs Signal Strength (R)")
                r_accuracy = results_df.groupby('R')['correct'].agg(['sum', 'count'])
                r_accuracy['accuracy'] = (r_accuracy['sum'] / r_accuracy['count'] * 100)
                st.line_chart(r_accuracy['accuracy'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Accuracy vs Dimension (d)")
                d_accuracy = results_df.groupby('d')['correct'].agg(['sum', 'count'])
                d_accuracy['accuracy'] = (d_accuracy['sum'] / d_accuracy['count'] * 100)
                st.line_chart(d_accuracy['accuracy'])
            
            with col2:
                st.subheader("Accuracy vs Noise (flip_prob)")
                flip_accuracy = results_df.groupby('flip_prob')['correct'].agg(['sum', 'count'])
                flip_accuracy['accuracy'] = (flip_accuracy['sum'] / flip_accuracy['count'] * 100)
                st.line_chart(flip_accuracy['accuracy'])
            
            st.divider()
            
            # Export and clear
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download All Results (CSV)",
                    data=csv,
                    file_name=f"all_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                if st.button("🗑️ Clear All Results"):
                    RESULTS_FILE.unlink(missing_ok=True)
                    st.success("Results cleared!")
                    st.rerun()
            
            with col3:
                st.info(f"Last test: {results_df['timestamp'].max()}")
        else:
            st.info("No predictions recorded yet. Go to Single Test or Batch Testing to start!")
