import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sympy as sp
import re

# --- Page Configuration ---
st.set_page_config(
    page_title="Gradient & Steepest Ascent Visualizer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. Math Input Preprocessing (The "Smart" Logic) ---
def preprocess_input(equation_str):
    """
    Converts standard math notation (x², x^2) into Python syntax (x**2).
    """
    if not equation_str:
        return ""
    
    # 1. Handle Unicode superscripts (what you wanted: x², y³)
    equation_str = equation_str.replace("²", "**2")
    equation_str = equation_str.replace("³", "**3")
    
    # 2. Handle standard latex-style caret (x^2 -> x**2)
    equation_str = equation_str.replace("^", "**")
    
    # 3. Handle 'e' for exponential (e^x -> exp(x)) roughly, mostly relies on user typing exp
    # But let's ensure 'e' is treated as Euler's number if used in standard context
    equation_str = equation_str.replace("e**", "exp") 
    
    return equation_str

def calculate_gradient(func_str, x_val, y_val):
    x, y = sp.symbols('x y')
    try:
        # Convert user's "pretty" math to python math
        clean_str = preprocess_input(func_str)
        
        # Parse the function
        f = sp.sympify(clean_str)
        
        # Calculate partial derivatives
        fx = sp.diff(f, x)
        fy = sp.diff(f, y)
        
        # Evaluate
        # Use simple eval for points to ensure standard float types
        z_val = float(f.subs({x: x_val, y: y_val}))
        fx_val = float(fx.subs({x: x_val, y: y_val}))
        fy_val = float(fy.subs({x: x_val, y: y_val}))
        
        return f, fx, fy, z_val, fx_val, fy_val, None
    except Exception as e:
        return None, None, None, None, None, None, str(e)

# --- Sidebar: Controls ---
st.sidebar.header("1. Input Configuration")
st.sidebar.markdown("Define your surface function $f(x, y)$.")

st.sidebar.info("💡 **Tip:** You can input standard math like `x² + y²` or `x^2 - y^2`. No need for python code!")

# Input for the function
func_input = st.sidebar.text_input(
    "Enter function f(x, y):", 
    value="10 - x² - y²"  # Default using your preferred notation
)

st.sidebar.markdown("---")
st.sidebar.header("2. Current Position $(x_0, y_0)$")
x0 = st.sidebar.slider("x coordinate", -5.0, 5.0, 1.0, 0.5)
y0 = st.sidebar.slider("y coordinate", -5.0, 5.0, 2.0, 0.5)

# --- Main Page Content ---
st.title("Interactive Calculus: Gradient & Steepest Ascent")
st.markdown("### A Visual Exploration for MAT201")

# Tabs
tab1, tab2, tab3 = st.tabs(["📘 Math & Theory", "📊 3D Visualization", "🌍 Real-World Application"])

# --- Calculation ---
f_sym, fx_sym, fy_sym, z0, grad_x, grad_y, error = calculate_gradient(func_input, x0, y0)

if error:
    st.error(f"⚠️ **Input Error:** Could not parse the function. Please use standard variables 'x' and 'y'. Error details: {error}")
    st.stop()

# --- Tab 1: Theory ---
with tab1:
    st.header("1. Mathematical Derivation")
    st.markdown("Here we calculate the gradient vector $\\nabla f$, which points in the direction of steepest ascent.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Function & Partials")
        st.write("Target Function:")
        st.latex(f"f(x, y) = {sp.latex(f_sym)}")
        
        st.write("Partial Derivatives (Rates of Change):")
        st.latex(r"\frac{\partial f}{\partial x} = " + sp.latex(fx_sym))
        st.latex(r"\frac{\partial f}{\partial y} = " + sp.latex(fy_sym))
        
    with col2:
        st.subheader("The Gradient Vector")
        st.latex(r"\nabla f = \left\langle \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y} \right\rangle")
        st.write(f"At point $({x0}, {y0})$:")
        st.latex(f"\\nabla f({x0}, {y0}) = \\langle {grad_x:.3f}, {grad_y:.3f} \\rangle")
        
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        st.write(f"**Steepest Ascent Rate (Magnitude):** {magnitude:.4f}")

# --- Tab 2: Visualization ---
with tab2:
    st.header("2. Interactive 3D Surface")
    
    # Grid for plotting
    grid_size = 5
    x_range = np.linspace(-grid_size, grid_size, 50)
    y_range = np.linspace(-grid_size, grid_size, 50)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Safe numerical evaluation
    # We convert sympy expression to a numpy-ready lambda function
    try:
        f_func = sp.lambdify((sp.Symbol('x'), sp.Symbol('y')), f_sym, 'numpy')
        Z = f_func(X, Y)
        # Handle constant functions (lambdify might return a scalar)
        if np.isscalar(Z): 
            Z = np.full_like(X, Z)
    except Exception as e:
        st.error("Could not plot surface.")
        st.stop()

    fig = go.Figure()

    # Surface
    fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', opacity=0.8, name='f(x,y)'))

    # Point
    fig.add_trace(go.Scatter3d(
        x=[x0], y=[y0], z=[z0],
        mode='markers', marker=dict(size=5, color='red'),
        name='Current Point'
    ))

    # Gradient Vector (Arrow)
    # Scale arrow for visibility
    scale = 1.0
    fig.add_trace(go.Scatter3d(
        x=[x0, x0 + grad_x * scale],
        y=[y0, y0 + grad_y * scale],
        z=[z0, z0 + magnitude * scale], # Visualizing ascent in Z
        mode='lines+markers',
        line=dict(color='orange', width=6),
        marker=dict(size=4, color='orange'),
        name='Gradient (Steepest Ascent)'
    ))

    fig.update_layout(
        title='Gradient Vector Visualization',
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        margin=dict(l=0, r=0, b=0, t=30),
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

# --- Tab 3: Real World ---
with tab3:
    st.header("3. Real-World Application: Heat Source Navigation")
    
    st.markdown("""
    ### Scenario: Autonomous Firefighting Drone
    
    **The Problem:**
    A drone is flying in a smoke-filled forest. It needs to find the heart of the fire (the hottest point) to drop water, but it cannot see visually.
    
    **The Math:**
    Let Temperature $T(x, y)$ be the function defined on the terrain.
    The drone measures the temperature at its current location and slightly around it to calculate the **Gradient**:
    
    $$ \\nabla T(x, y) = \langle T_x, T_y \\rangle $$
    
    **The Strategy (Steepest Ascent):**
    1. The drone calculates $\\nabla T$.
    2. It moves in the direction of this vector because this is the direction where temperature increases **most rapidly**.
    3. By repeating this, the drone automatically finds the maximum temperature (the fire source).
    
    **Why this matters:**
    This same algorithm (Gradient Ascent/Descent) is used to train **Neural Networks** (like the AI generating this code!). The AI tries to find the "bottom of the valley" (minimum error) by following the negative gradient.
    """)
