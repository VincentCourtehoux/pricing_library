import streamlit as st
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

st.set_page_config(layout="wide", page_title="Option Pricing Calculator")

st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem;
    }

    .main > div {
        padding-top: 0;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .sidebar .sidebar-content {
        background-color: #f8f9fa;
        padding: 0.5rem 1rem;
        padding-top: 1rem;
        height: 100vh;
        overflow-y: auto;
    }

    .css-1d391kg {
        padding-top: 1rem;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1rem;
    }

    .model-category {
        background-color: #e9ecef;
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 5px;
        font-weight: bold;
    }

    .model-option {
        margin-left: 1rem;
        margin: 0.3rem 0;
    }

    .stRadio > label {
        font-size: 14px;
    }

    .main-content {
        margin-left: 0;
        padding: 1rem;
    }

    .parameter-section {
        background-color: #ffffff;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .results-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 10px;
        border-left: 4px solid #007bff;
    }

    .model-selection-title {
        font-size: 26px;
        font-weight: 700;
        color: #343a40;
        margin-bottom: 0rem;
        margin-top: 0;
    }

    .sidebar .model-selection-title,
    section[data-testid="stSidebar"] .model-selection-title,
    div[data-testid="stSidebar"] .model-selection-title {
        font-size: 26px;
        font-weight: 700;
        color: #343a40;
    }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<h3 class="model-selection-title">Model Selection</h3>', unsafe_allow_html=True)

    def clear_others(selected_group):
        if selected_group == "european":
            st.session_state.american_radio = None
            st.session_state.barrier_radio = None
        elif selected_group == "american":
            st.session_state.european_radio = None
            st.session_state.barrier_radio = None
        elif selected_group == "barrier":
            st.session_state.european_radio = None
            st.session_state.american_radio = None

    st.markdown('<div class="model-category"> 🍦 Vanilla Options</div>', unsafe_allow_html=True)

    st.markdown("**European**")
    european_models = st.radio(
        "",
        ["Black-Scholes", "Monte Carlo"],
        key="european_radio",
        index=None,
        label_visibility="collapsed",
        on_change=clear_others,
        args=("european",)
    )

    st.markdown("**American**")
    american_models = st.radio(
        "",
        ["Binomial Tree", "Longstaff-Schwartz"],
        key="american_radio",
        index=None,
        label_visibility="collapsed",
        on_change=clear_others,
        args=("american",)
    )

    st.markdown('<div class="model-category"> 🌟 Exotic Options</div>', unsafe_allow_html=True)

    st.markdown("**Barrier**")
    barrier_models = st.radio(
        "",
        ["Black Scholes", "Monte Carlo"],
        key="barrier_radio",
        index=None,
        label_visibility="collapsed",
        on_change=clear_others,
        args=("barrier",)
    )
    
    selected_models = {
    "european": st.session_state.get("european_radio"),
    "american": st.session_state.get("american_radio"),
    "barrier": st.session_state.get("barrier_radio")
    }

    active_model = None
    option_style = None
    pricing_model = None

    for category, model in selected_models.items():
        if model is not None:
            active_model = model
            if category == "european":
                option_style = "European"
                pricing_model = model
            elif category == "american":
                option_style = "American"
                pricing_model = model
            elif category == "barrier":
                option_style = "Barrier"
                pricing_model = model
            break

st.markdown("# Multi-Model Option Pricing Calculator")
st.markdown("Configure your option parameters and calculate prices using advanced pricing models")

if active_model:
    st.subheader("📋 Contract Specifications")
    
    contract_col1, contract_col2 = st.columns(2)
    
    with contract_col1:
        option_type = st.selectbox(
            "Option Type", 
            ["Call", "Put"],
            help="Call: right to buy, Put: right to sell"
        ).lower()
        
        K = st.number_input(
            "Strike Price (K)", 
            min_value=0.01, 
            value=100.0, 
            step=5.0,
            help="Exercise price of the option"
        )
    
    with contract_col2:
        T = st.number_input(
            "Time to Maturity (years)", 
            min_value=0.001, 
            value=1.00, 
            step=0.25,
            format="%.3f",
            help="Time remaining until option expiration"
        )
    
    if option_style == "Barrier":
        barrier_col1, barrier_col2, barrier_col3 = st.columns(3)
        with barrier_col1:
            barrier_type = st.selectbox(
                "Barrier Type",
                ["Up-In", "Up-Out", "Down-In", "Down-Out"],
                help="Type of barrier condition"
            ).lower()
        with barrier_col2:
            barrier = st.number_input(
                "Barrier Level", 
                min_value=0.01, 
                value=110.0, 
                step=5.0,
                help="Barrier price level"
            )
        with barrier_col3:
            rebate = st.number_input(
                "Rebate Amount", 
                min_value=0.0, 
                value=0.0, 
                step=1.0,
                help="Rebate amount if barrier is breached"
            )
    st.subheader("📈 Market Data")
    
    market_col1, market_col2, market_col3 = st.columns(3)
    
    with market_col1:
        S = st.number_input(
            "Current Asset Price (S)", 
            min_value=0.01, 
            value=100.0, 
            step=5.0,
            help="Current market price of the underlying asset"
        )
    
    with market_col2:
        r = st.number_input(
            "Risk-Free Rate", 
            min_value=0.0, 
            max_value=1.0,
            value=0.05, 
            step=0.005,
            format="%.3f",
            help="Annualized risk-free interest rate"
        )
    
    with market_col3:
        sigma = st.number_input(
            "Volatility", 
            min_value=0.001, 
            max_value=5.0,
            value=0.20, 
            step=0.01,
            format="%.3f",
            help="Annualized volatility of the underlying asset"
        )
    
    dividend_col1, dividend_col2 = st.columns([1, 2])
    
    with dividend_col1:
        q = st.number_input(
            "Dividend Yield", 
            min_value=0.0, 
            max_value=1.0,
            value=0.000, 
            step=0.005,
            format="%.3f",
            help="Continuous dividend yield"
        )
    
    if pricing_model in ["Monte Carlo", "Longstaff-Schwartz", "Binomial Tree"]:
        st.subheader("⚙️ Simulation Parameters")
        
        num_col1, num_col2 = st.columns(2)
        
        with num_col1:
            N = st.number_input(
                "Time Steps", 
                min_value=1, 
                max_value=1000,
                value=100, 
                step=10,
                help="Number of time steps for discretization"
            )
        
        with num_col2:
            nb_paths = st.number_input(
                "Simulation Paths", 
                min_value=1, 
                max_value=100000,
                value=10000, 
                step=1000,
                help="Number of Monte Carlo simulation paths"
            )
    
    st.subheader("🎯 Results")
    
    if st.button("Calculate Option Price", type="primary", use_container_width=True):
        try:
            if option_style == "European" and pricing_model == "Black-Scholes":
                from core.models.vanilla.european.black_scholes.pricing_scalar import BlackScholesScalar
                bs = BlackScholesScalar()
                price = bs.premium(S, K, T, r, sigma, q, option_type)
            elif option_style == "European" and pricing_model == "Monte Carlo":
                from core.models.vanilla.european.monte_carlo import price_european_option_mc
                price = price_european_option_mc(S, K, T, r, sigma, q, N, nb_paths, option_type, seed=44)
            elif option_style == "American" and pricing_model == "Longstaff-Schwartz":
                from core.models.vanilla.american.longstaff_schwartz.pricing import longstaff_schwartz_american
                price = longstaff_schwartz_american(S, K, r, sigma, T, q, N, nb_paths, option_type, degree=2, seed=42)
            elif option_style == "American" and pricing_model == "Binomial Tree":
                from core.models.vanilla.american.binomial_tree import BinomialTreeAmerican
                bt = BinomialTreeAmerican(S, K, T, r, sigma, N, option_type)
                price = bt.price_option()
            elif option_style == "Barrier" and pricing_model == "Monte Carlo":
                from core.models.exotic.barrier.monte_carlo_pricing import mc_barrier_premium
                price = mc_barrier_premium(S, K, T, r, sigma, q, barrier, N, nb_paths, option_type, barrier_type, seed=42)
            elif option_style == "Barrier" and pricing_model == "Black Scholes":
                from core.models.exotic.barrier.bs_barrier_pricing import bs_barrier_premium
                price = bs_barrier_premium(S, K, T, r, sigma, q, barrier, option_type, barrier_type, rebate)               
            
            st.success(f"**{option_type.capitalize()} Option Price ({pricing_model}): ${price:.4f}**")
            
            met_col1, met_col2, met_col3 = st.columns(3)
            
            with met_col1:
                moneyness = S / K
                if moneyness > 1:
                    money_status = "ITM" if option_type == "call" else "OTM"
                elif moneyness < 1:
                    money_status = "OTM" if option_type == "call" else "ITM"
                else:
                    money_status = "ATM"
                
                st.metric("Moneyness", f"{moneyness:.3f}", money_status)
            
            with met_col2:
                if option_style == "Barrier":
                    if barrier_type in ["up-in", "down-in"] and (S > barrier if barrier_type == "down-in" else S < barrier):
                        intrinsic_value = 0
                    elif barrier_type in ["up-out", "down-out"] and (S < barrier if barrier_type == "down-out" else S > barrier):
                        intrinsic_value = 0
                    else: intrinsic_value = max(0, S - K) if option_type == "call" else max(0, K - S)
                else : 
                    intrinsic_value = max(0, S - K) if option_type == "call" else max(0, K - S)
                st.metric("Intrinsic Value", f"${intrinsic_value:.4f}")
            
            with met_col3:
                time_value = price - intrinsic_value
                st.metric("Time Value", f"${time_value:.4f}")
                
        except Exception as e:
            st.error(f"Error calculating option price: {str(e)}")
            st.info("Please check your input parameters and try again.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    with st.expander("📚 Model Information"):
        if pricing_model == "Black-Scholes":
            st.markdown("""
            **Black-Scholes Model**
            
            The classic option pricing model developed by Fischer Black, Myron Scholes, and Robert Merton.
            
            **Key Features:**
            - Analytical closed-form solution
            - Assumes constant volatility and interest rates
            - European-style exercise only
            - No dividends or constant dividend yield
            
            **Best for:** Liquid European options with known parameters
            """)
        elif pricing_model == "Binomial Tree":
            st.markdown("""
            **Binomial Tree Model**
            
            A discrete-time model that builds a tree of possible asset price movements.
            
            **Key Features:**
            - Can handle American-style early exercise
            - Flexible for different payoff structures
            - Convergence to Black-Scholes with more steps
            
            **Best for:** American options and complex payoff structures
            """)
        elif pricing_model == "Monte Carlo":
            st.markdown("""
            **Monte Carlo Simulation**
            
            Uses random sampling to simulate many possible price paths.
            
            **Key Features:**
            - Handles complex payoffs and multiple assets
            - Path-dependent options (Asian, Barrier)
            - Flexible but computationally intensive
            
            **Best for:** Exotic options and multi-asset derivatives
            """)
        elif pricing_model == "Longstaff-Schwartz":
            st.markdown("""
            **Longstaff-Schwartz Method**
            
            A Monte Carlo approach for pricing American options using least squares regression.
            
            **Key Features:**
            - Handles American-style early exercise
            - Uses regression to estimate continuation values
            - Flexible for complex payoffs
            
            **Best for:** American options with complex features
            """)

else:
    st.info("👈 Please select a pricing model from the sidebar to configure parameters")

st.markdown("---")
st.markdown("© 2025 - Développé par Vincent Courtehoux avec [Streamlit](https://streamlit.io/)")
