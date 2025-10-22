import os
import io
import json
import time
import yaml
import pandas as pd
import streamlit as st
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

# ============================================================================
# THEME CONFIGURATIONS - 20 Beautiful Themes
# ============================================================================
THEMES = {
    "Streamlit Default": {
        "primaryColor": "#FF4B4B",
        "backgroundColor": "#FFFFFF",
        "secondaryBackgroundColor": "#F0F2F6",
        "textColor": "#31333F",
        "font": "sans serif"
    },
    "Ocean Blue": {
        "primaryColor": "#0066CC",
        "backgroundColor": "#F0F8FF",
        "secondaryBackgroundColor": "#E6F3FF",
        "textColor": "#1A1A1A",
        "font": "sans serif"
    },
    "Forest Green": {
        "primaryColor": "#2E7D32",
        "backgroundColor": "#F1F8F4",
        "secondaryBackgroundColor": "#E8F5E9",
        "textColor": "#1B5E20",
        "font": "sans serif"
    },
    "Sunset Orange": {
        "primaryColor": "#FF6B35",
        "backgroundColor": "#FFF8F3",
        "secondaryBackgroundColor": "#FFE5D9",
        "textColor": "#2D2D2D",
        "font": "sans serif"
    },
    "Royal Purple": {
        "primaryColor": "#7B2CBF",
        "backgroundColor": "#FAF5FF",
        "secondaryBackgroundColor": "#F3E5F5",
        "textColor": "#2D2D2D",
        "font": "sans serif"
    },
    "Cherry Blossom": {
        "primaryColor": "#E91E63",
        "backgroundColor": "#FFF0F5",
        "secondaryBackgroundColor": "#FCE4EC",
        "textColor": "#2D2D2D",
        "font": "sans serif"
    },
    "Midnight Dark": {
        "primaryColor": "#00D9FF",
        "backgroundColor": "#0E1117",
        "secondaryBackgroundColor": "#1E2130",
        "textColor": "#FAFAFA",
        "font": "monospace"
    },
    "Nord": {
        "primaryColor": "#88C0D0",
        "backgroundColor": "#2E3440",
        "secondaryBackgroundColor": "#3B4252",
        "textColor": "#ECEFF4",
        "font": "sans serif"
    },
    "Solarized Dark": {
        "primaryColor": "#268BD2",
        "backgroundColor": "#002B36",
        "secondaryBackgroundColor": "#073642",
        "textColor": "#839496",
        "font": "monospace"
    },
    "Dracula": {
        "primaryColor": "#BD93F9",
        "backgroundColor": "#282A36",
        "secondaryBackgroundColor": "#44475A",
        "textColor": "#F8F8F2",
        "font": "monospace"
    },
    "Monokai": {
        "primaryColor": "#F92672",
        "backgroundColor": "#272822",
        "secondaryBackgroundColor": "#3E3D32",
        "textColor": "#F8F8F2",
        "font": "monospace"
    },
    "Cyberpunk": {
        "primaryColor": "#FF00FF",
        "backgroundColor": "#0A0E27",
        "secondaryBackgroundColor": "#1A1F3A",
        "textColor": "#00FFFF",
        "font": "monospace"
    },
    "Mint Fresh": {
        "primaryColor": "#00BFA5",
        "backgroundColor": "#F0FFF4",
        "secondaryBackgroundColor": "#E0F2E9",
        "textColor": "#1B4D3E",
        "font": "sans serif"
    },
    "Coffee": {
        "primaryColor": "#8B4513",
        "backgroundColor": "#FFF8DC",
        "secondaryBackgroundColor": "#F5E6D3",
        "textColor": "#3E2723",
        "font": "serif"
    },
    "Lavender": {
        "primaryColor": "#9C27B0",
        "backgroundColor": "#F3E5F5",
        "secondaryBackgroundColor": "#E1BEE7",
        "textColor": "#4A148C",
        "font": "sans serif"
    },
    "Arctic": {
        "primaryColor": "#00ACC1",
        "backgroundColor": "#E0F7FA",
        "secondaryBackgroundColor": "#B2EBF2",
        "textColor": "#006064",
        "font": "sans serif"
    },
    "Rose Gold": {
        "primaryColor": "#B76E79",
        "backgroundColor": "#FFF5F7",
        "secondaryBackgroundColor": "#FFE4E8",
        "textColor": "#5D4037",
        "font": "serif"
    },
    "Emerald": {
        "primaryColor": "#00897B",
        "backgroundColor": "#E0F2F1",
        "secondaryBackgroundColor": "#B2DFDB",
        "textColor": "#004D40",
        "font": "sans serif"
    },
    "Amber": {
        "primaryColor": "#FF8F00",
        "backgroundColor": "#FFF8E1",
        "secondaryBackgroundColor": "#FFECB3",
        "textColor": "#FF6F00",
        "font": "sans serif"
    },
    "Steel Gray": {
        "primaryColor": "#546E7A",
        "backgroundColor": "#ECEFF1",
        "secondaryBackgroundColor": "#CFD8DC",
        "textColor": "#263238",
        "font": "sans serif"
    }
}

APP_TITLE = "🚀 Agentic Dataset Workbench Pro"
APP_DESC = "Multi-theme, multi-provider agentic dataset processing with OpenAI, Gemini, and Grok"

DEFAULT_SAMPLE_JSON = [
    {"entity":"Documentation Level","title":"Risk-based documentation level","context":"Determines Basic vs Enhanced software documentation based on hazard of death/serious injury prior to risk controls.","keywords":["Basic","Enhanced","risk","hazardous situation"]},
    {"entity":"Basic Documentation Level","title":"Lower-risk software documentation","context":"Applies when failures would not present probable risk of death/serious injury before risk controls.","keywords":["risk","software","510k"]},
    {"entity":"Enhanced Documentation Level","title":"Higher-risk software documentation","context":"Applies when failures could present probable risk of death/serious injury before risk controls; includes SDS and detailed tests.","keywords":["high risk","SDS","unit/integration tests"]},
    {"entity":"Device software function","title":"Software that is a medical device","context":"A software function meeting FD&C Act 201(h).","keywords":["function","device","FDCA"]},
    {"entity":"Off-the-Shelf Software","title":"OTS software components","context":"Software for which manufacturer lacks full lifecycle control (OS, libraries).","keywords":["OTS","COTS","libraries"]},
]

PROVIDERS = {
    "openai": {
        "env_key": "OPENAI_API_KEY",
        "models": ["gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
    },
    "gemini": {
        "env_key": "GEMINI_API_KEY",
        "models": ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"]
    },
    "grok": {
        "env_key": "GROK_API_KEY",
        "models": ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"]
    }
}

DEFAULT_AGENTS_YAML = """agents:
  - name: Summarizer
    provider: openai
    model: gpt-4o-mini
    description: Summarize dataset entries into a concise executive overview.
    system_prompt: |
      You are a helpful data analyst. Provide a concise, accurate summary of the provided data.
    user_prompt: |
      Summarize the following dataset. Focus on key entities and themes.
      Dataset preview (first 5 rows):
      {dataset_preview}
    params:
      temperature: 0.2
      max_tokens: 800
      top_p: 1.0
      force_json: false

  - name: JSON Converter
    provider: gemini
    model: gemini-1.5-flash
    description: Converts the summary into a structured JSON object.
    system_prompt: |
      You are a helpful NLP assistant. Your task is to convert the user's text into a structured JSON format.
    user_prompt: |
      Please convert the following summary into a JSON object with the keys "main_theme", "key_entities", and "estimated_risk_level".
      Summary:
      {previous_agent_output}
    params:
      temperature: 0.1
      max_tokens: 2048
      top_p: 0.9
      force_json: true

  - name: Final Reviewer
    provider: grok
    model: llama3-8b-8192
    description: Reviews the structured JSON and provides a final comment.
    system_prompt: |
      You are Grok, an intelligent quality assurance agent.
      Review the provided JSON and give a brief, one-sentence comment on its quality.
    user_prompt: |
      Review this JSON and provide a final quality comment.
      JSON for review:
      {previous_agent_output}
    params:
      temperature: 0.3
      max_tokens: 4096
      top_p: 0.95
      force_json: false
"""

# ============================================================================
# SESSION STATE MANAGEMENT
# ============================================================================

def ensure_session_state():
    """Initialize all session state variables"""
    defaults = {
        "dataset_df": pd.DataFrame(DEFAULT_SAMPLE_JSON),
        "agents_cfg": load_or_default_agents(),
        "keys": {
            "openai": os.getenv(PROVIDERS["openai"]["env_key"]),
            "gemini": os.getenv(PROVIDERS["gemini"]["env_key"]),
            "grok": os.getenv(PROVIDERS["grok"]["env_key"]),
        },
        "connected": {"openai": False, "gemini": False, "grok": False},
        "run_history": [],
        "current_theme": "Ocean Blue",
        "auto_save": True,
        "show_advanced": False,
        "configured_agents": [],
        "pipeline_step": 0,
        "pipeline_outputs": {},
        "pipeline_results": [],
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def load_or_default_agents() -> Dict[str, Any]:
    """Load agents from agents.yaml or return defaults"""
    path = "agents.yaml"
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict) and "agents" in data:
                return data
        except Exception as e:
            st.warning(f"Failed to load agents.yaml: {e}")
    return yaml.safe_load(DEFAULT_AGENTS_YAML)

def save_agents_cfg(cfg: Dict[str, Any]):
    """Save agents configuration to file"""
    try:
        with open("agents.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        st.toast("✅ Saved agents.yaml", icon="✅")
    except Exception as e:
        st.toast(f"⚠️ Could not save: {e}", icon="⚠️")

# ============================================================================
# DATA LOADING & PROCESSING
# ============================================================================

def parse_text_as_records(text: str) -> pd.DataFrame:
    """Parse various text formats into DataFrame"""
    text = text.strip()
    try:
        obj = json.loads(text)
        return pd.json_normalize(obj if isinstance(obj, list) else [obj])
    except: pass
    try:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        if rows: return pd.json_normalize(rows)
    except: pass
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines: return pd.DataFrame({"text": lines})
    return pd.DataFrame()

def load_file_to_df(uploaded_file) -> pd.DataFrame:
    """Load uploaded file into DataFrame with smart detection"""
    try:
        content = uploaded_file.getvalue()
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(io.BytesIO(content))
        elif uploaded_file.name.endswith(('.json', '.jsonl')):
            return parse_text_as_records(content.decode("utf-8"))
        else:
            return parse_text_as_records(content.decode("utf-8"))
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return pd.DataFrame()

# ============================================================================
# LLM PROVIDER INTEGRATIONS
# ============================================================================

def ensure_openai_client(api_key: str):
    from openai import OpenAI
    return OpenAI(api_key=api_key)

def ensure_gemini_client(api_key: str):
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    return genai

def ensure_grok_client(api_key: str):
    from openai import OpenAI
    return OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")

def call_openai(model: str, system_prompt: str, user_prompt: str, params: Dict[str, Any], api_key: str) -> str:
    client = ensure_openai_client(api_key)
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt or "You are a helpful assistant."},
            {"role": "user", "content": user_prompt or ""},
        ],
        "temperature": params.get("temperature", 0.2),
        "max_tokens": params.get("max_tokens", 1024),
        "top_p": params.get("top_p", 1.0),
    }
    if params.get("force_json"):
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content

def call_gemini(model: str, system_prompt: str, user_prompt: str, params: Dict[str, Any], api_key: str) -> str:
    genai = ensure_gemini_client(api_key)
    gen_config = {
        "temperature": params.get("temperature", 0.2),
        "top_p": params.get("top_p", 0.95),
        "max_output_tokens": params.get("max_tokens", 1024),
    }
    if params.get("force_json"):
        gen_config["response_mime_type"] = "application/json"
    
    gmodel = genai.GenerativeModel(model_name=model, system_instruction=system_prompt)
    resp = gmodel.generate_content(user_prompt or "", generation_config=gen_config)
    return resp.text or ""

def call_grok(model: str, system_prompt: str, user_prompt: str, params: Dict[str, Any], api_key: str) -> str:
    client = ensure_grok_client(api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt or "You are Grok, a helpful AI."},
            {"role": "user", "content": user_prompt or ""},
        ],
        temperature=params.get("temperature", 0.2),
        max_tokens=params.get("max_tokens", 1024),
    )
    return resp.choices[0].message.content

def call_llm(provider: str, model: str, system_prompt: str, user_prompt: str, 
             params: Dict[str, Any], api_key: str) -> Tuple[bool, str]:
    """Unified LLM calling interface"""
    try:
        if provider == "openai":
            out = call_openai(model, system_prompt, user_prompt, params, api_key)
        elif provider == "gemini":
            out = call_gemini(model, system_prompt, user_prompt, params, api_key)
        elif provider == "grok":
            out = call_grok(model, system_prompt, user_prompt, params, api_key)
        else:
            return False, f"Unknown provider: {provider}"
        return True, out
    except Exception as e:
        return False, f"API Error ({provider}): {str(e)}"

# ============================================================================
# PROMPT TEMPLATING
# ============================================================================

def build_user_prompt(template: str, df: pd.DataFrame, previous_output: Optional[str] = None) -> str:
    """Build user prompt with dataset and pipeline templating"""
    def to_json(df_in: pd.DataFrame):
        return json.dumps(df_in.to_dict(orient="records"), ensure_ascii=False, indent=2)

    mapping = {
        "dataset_preview": to_json(df.head(5)),
        "dataset_sample_200": to_json(df.head(200)),
        "dataset_first_50": to_json(df.head(50)),
        "full_dataset_json": to_json(df),
        "columns": json.dumps(list(df.columns)),
        "row_count": str(len(df)),
        "previous_agent_output": previous_output or ""
    }
    
    class SafeDict(dict):
        def __missing__(self, key):
            return "{" + key + "}"
    return template.format_map(SafeDict(mapping))

# ============================================================================
# UI RENDERING FUNCTIONS
# ============================================================================

def apply_custom_theme(theme_name: str):
    """Apply custom theme styling"""
    theme = THEMES.get(theme_name, THEMES["Ocean Blue"])
    # This is a simplified approach. For full theme application, Streamlit's config.toml is preferred.
    # We will use st.markdown to inject CSS as a dynamic alternative.
    st.markdown(f"<style>[data-testid='stAppViewContainer'] {{ background-color: {theme['backgroundColor']}; }}</style>", unsafe_allow_html=True)

def render_header():
    st.set_page_config(page_title=APP_TITLE, page_icon="🚀", layout="wide")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title(APP_TITLE)
        st.caption(APP_DESC)
    with col2:
        theme_name = st.selectbox("🎨 Theme", options=list(THEMES.keys()), key="current_theme")
    apply_custom_theme(st.session_state.current_theme)

def provider_badge(provider: str, connected: bool) -> str:
    icon = "🟢" if connected else "🔴"
    return f"{icon} **{provider.upper()}**"

def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Settings")
        df = st.session_state.dataset_df
        st.metric("Dataset Rows", len(df))
        st.metric("Columns", len(df.columns))
        st.metric("Total Runs", len(st.session_state.run_history))
        st.divider()
        st.session_state.auto_save = st.checkbox("💾 Auto-save results", value=st.session_state.get("auto_save", True))
        st.session_state.show_advanced = st.checkbox("🔧 Show advanced options", value=st.session_state.get("show_advanced", False))
        st.divider()
        st.subheader("⚡ Quick Actions")
        if st.button("🔄 Reload Agents", use_container_width=True):
            st.session_state.agents_cfg = load_or_default_agents()
            st.toast("Agents reloaded!", icon="✅")
        if st.button("🗑️ Clear Pipeline", use_container_width=True):
            st.session_state.pipeline_step = 0
            st.session_state.pipeline_outputs = {}
            st.session_state.pipeline_results = []
            st.toast("Pipeline results cleared!", icon="🗑️")
        if st.button("📋 Reset Dataset", use_container_width=True):
            st.session_state.dataset_df = pd.DataFrame(DEFAULT_SAMPLE_JSON)
            st.toast("Dataset reset to default!", icon="📋")

def render_provider_keys():
    st.subheader("🔐 API Connections")
    cols = st.columns(len(PROVIDERS))
    for idx, (provider, config) in enumerate(PROVIDERS.items()):
        with cols[idx]:
            is_connected = bool(st.session_state.keys.get(provider))
            st.markdown(provider_badge(provider, is_connected))
            if not is_connected:
                key = st.text_input(f"{provider.upper()} API Key", type="password", key=f"{provider}_key_input")
                if key:
                    st.session_state.keys[provider] = key
                    st.toast(f"✅ {provider.upper()} key accepted!", icon="🔐")
                    st.rerun()
            else:
                st.success("✓ Key loaded", icon="🔒")
            st.session_state.connected[provider] = bool(st.session_state.keys.get(provider))

def render_dataset_tab():
    st.subheader("📊 Dataset Management")
    col1, col2 = st.columns(2)
    with col1:
        uploaded = st.file_uploader("Upload File (.csv, .json, .jsonl, .txt)", type=["csv", "json", "jsonl", "txt"])
        if uploaded:
            df = load_file_to_df(uploaded)
            if not df.empty:
                st.session_state.dataset_df = df.reset_index(drop=True)
                st.success(f"✅ Loaded {len(df)} rows")
    with col2:
        raw_data = st.text_area("Or Paste Data", placeholder='[{"col1": "data1"}]')
        if st.button("Parse Data", use_container_width=True):
            if raw_data.strip():
                df = parse_text_as_records(raw_data)
                if not df.empty:
                    st.session_state.dataset_df = df.reset_index(drop=True)
                    st.success(f"✅ Parsed {len(df)} rows")
    st.markdown("#### Dataset Preview")
    st.dataframe(st.session_state.dataset_df.head(10), use_container_width=True)

def render_agents_tab():
    st.subheader("🤖 Agent Configuration")
    with st.expander("📝 Edit agents.yaml"):
        yaml_text = st.text_area("YAML Configuration", value=yaml.dump(st.session_state.agents_cfg, sort_keys=False), height=300)
        c1, c2 = st.columns(2)
        if c1.button("🔄 Apply from Text", use_container_width=True):
            try:
                st.session_state.agents_cfg = yaml.safe_load(yaml_text)
                st.success("✅ Configuration updated")
            except Exception as e:
                st.error(f"❌ Invalid YAML: {e}")
        if c2.button("💾 Save to File", use_container_width=True):
            save_agents_cfg(st.session_state.agents_cfg)
    
    st.divider()
    
    agents = st.session_state.agents_cfg.get("agents", [])
    if not agents:
        st.warning("No agents defined in configuration.")
        return

    st.markdown("#### Select and Configure Agents for Pipeline")
    agent_names = [a.get("name", f"Agent {i}") for i, a in enumerate(agents)]
    selected_names = st.multiselect("Choose agents to run in order", agent_names, default=agent_names if agent_names else [])
    
    configured_agents = []
    for name in selected_names:
        agent = next((a for a in agents if a.get("name") == name), None)
        if not agent: continue

        with st.expander(f"⚙️ Configure: {agent['name']}", expanded=True):
            st.markdown(f"**Description:** {agent.get('description', 'N/A')}")
            c1, c2 = st.columns(2)
            provider = c1.selectbox("Provider", list(PROVIDERS.keys()), index=list(PROVIDERS.keys()).index(agent.get("provider", "openai")), key=f"provider_{name}")
            model_list = PROVIDERS[provider]["models"]
            default_model = agent.get("model", model_list[0])
            model_idx = model_list.index(default_model) if default_model in model_list else 0
            model = c2.selectbox("Model", model_list, index=model_idx, key=f"model_{name}")
            
            system_prompt = st.text_area("System Prompt", value=agent.get("system_prompt", ""), height=100, key=f"sys_{name}")
            user_prompt = st.text_area("User Prompt Template", value=agent.get("user_prompt", ""), height=150, key=f"user_{name}", help="Use {previous_agent_output} to chain agents.")
            
            params = agent.get("params", {})
            if st.session_state.show_advanced:
                st.markdown("**Advanced Parameters**")
                pc1, pc2, pc3, pc4 = st.columns(4)
                temperature = pc1.slider("Temperature", 0.0, 2.0, float(params.get("temperature", 0.2)), 0.05, key=f"temp_{name}")
                top_p = pc2.slider("Top P", 0.0, 1.0, float(params.get("top_p", 1.0)), 0.05, key=f"topp_{name}")
                max_tokens = pc3.number_input("Max Tokens", 16, 32768, int(params.get("max_tokens", 1024)), key=f"maxtok_{name}")
                force_json = pc4.checkbox("Force JSON", value=bool(params.get("force_json", False)), key=f"json_{name}")
            else:
                temperature, top_p, max_tokens, force_json = (params.get("temperature", 0.2), params.get("top_p", 1.0), params.get("max_tokens", 1024), params.get("force_json", False))

            configured_agents.append({
                "name": agent.get("name"), "provider": provider, "model": model,
                "system_prompt": system_prompt, "user_prompt": user_prompt,
                "params": {"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens, "force_json": force_json}
            })
    
    st.session_state.configured_agents = configured_agents
    if configured_agents:
        st.success(f"{len(configured_agents)} agents selected and configured for the pipeline.")

def render_run_tab():
    st.subheader("🚀 Execute Agent Pipeline")

    agents_to_run = st.session_state.get("configured_agents", [])
    if not agents_to_run:
        st.warning("⚠️ No agents configured. Go to the 'Agents' tab to select and set up agents for the pipeline.")
        return

    step = st.session_state.pipeline_step
    
    c1, c2 = st.columns([3,1])
    with c1:
        st.progress((step) / len(agents_to_run), text=f"Step {step} of {len(agents_to_run)} complete")
    if c2.button("🔄 Start/Reset Pipeline", use_container_width=True):
        st.session_state.pipeline_step = 0
        st.session_state.pipeline_outputs = {}
        st.session_state.pipeline_results = []
        st.rerun()

    st.divider()

    if step >= len(agents_to_run):
        st.success("✅ Pipeline Finished!")
        st.balloons()
        render_results()
        return

    current_agent = agents_to_run[step]
    agent_name = current_agent['name']
    
    st.header(f"Step {step + 1}/{len(agents_to_run)}: {agent_name}")
    st.markdown(f"**Provider:** `{current_agent['provider']}` | **Model:** `{current_agent['model']}`")

    previous_output = ""
    if step > 0:
        prev_agent_name = agents_to_run[step - 1]['name']
        previous_output = st.session_state.pipeline_outputs.get(prev_agent_name, "")
        with st.expander("Input from previous step (read-only)"):
            st.text(previous_output)

    run_placeholder = st.empty()
    if run_placeholder.button(f"▶️ Execute: {agent_name}", type="primary", use_container_width=True):
        api_key = st.session_state.keys.get(current_agent["provider"])
        if not api_key:
            st.error(f"Missing API key for {current_agent['provider'].upper()}. Please set it in the Dataset tab.")
            return

        full_prompt = build_user_prompt(current_agent["user_prompt"], st.session_state.dataset_df, previous_output)
        
        with st.spinner(f"Running {agent_name}..."):
            start_time = time.time()
            ok, output = call_llm(
                current_agent["provider"], current_agent["model"],
                current_agent["system_prompt"], full_prompt,
                current_agent["params"], api_key
            )
            elapsed = time.time() - start_time

        result = {
            "agent": agent_name, "provider": current_agent["provider"], "model": current_agent["model"],
            "ok": ok, "output": output if ok else "", "error": output if not ok else "",
            "elapsed": elapsed, "timestamp": datetime.now().isoformat()
        }
        st.session_state.pipeline_results.append(result)

        if ok:
            st.session_state.pipeline_outputs[agent_name] = output
            run_placeholder.empty() # Remove run button after execution
            st.rerun()
        else:
            st.error(f"Agent failed: {output}")

    # Display output editor and confirmation button
    if agent_name in st.session_state.pipeline_outputs:
        st.success(f"✅ {agent_name} completed.")
        output = st.session_state.pipeline_outputs[agent_name]
        
        modified_output = st.text_area(
            "📝 Agent Output (Editable)",
            value=output,
            height=250,
            key=f"output_editor_{agent_name}",
            help="Modify this output before it's passed to the next agent."
        )
        
        # Update the stored output with any user modifications
        st.session_state.pipeline_outputs[agent_name] = modified_output

        if st.button("Confirm and Continue to Next Step →", use_container_width=True):
            st.session_state.pipeline_step += 1
            st.rerun()

def render_results():
    """Render execution results after pipeline is finished"""
    st.markdown("### 📊 Final Results")
    results = st.session_state.pipeline_results
    if not results:
        st.info("No results to display yet.")
        return

    for idx, result in enumerate(results):
        status_icon = "✅" if result["ok"] else "❌"
        with st.expander(f"{status_icon} {result['agent']} ({result.get('elapsed', 0):.2f}s)", expanded=True):
            if result["ok"]:
                st.text_area("Output", value=result["output"], height=200, key=f"final_output_{idx}", disabled=True)
            else:
                st.error(f"**Error:** {result.get('error', 'Unknown error')}")

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application entry point"""
    ensure_session_state()
    render_header()
    render_sidebar()
    
    tabs = st.tabs(["📊 Dataset", "🤖 Agents", "🚀 Run & Results"])
    
    with tabs[0]:
        render_provider_keys()
        st.divider()
        render_dataset_tab()
    
    with tabs[1]:
        render_agents_tab()
    
    with tabs[2]:
        render_run_tab()

if __name__ == "__main__":
    main()
