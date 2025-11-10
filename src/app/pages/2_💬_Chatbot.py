"""
Chatbot page for Google Gemini conversations.
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

# Google Gemini import
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="Chatbot - XAI Pipeline",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

def query_gemini(user_message, api_key, model="gemini-1.5-flash", chat_history=None):
    """
    Query Google Gemini API for chat responses.
    
    Args:
        user_message: User's message
        api_key: Google Gemini API key
        model: Model name (default: gemini-1.5-flash)
        chat_history: List of previous messages in format [{"role": "user/assistant", "content": "..."}]
    
    Returns:
        str: Response from Gemini
    """
    if not GEMINI_AVAILABLE:
        return "⚠️ Google Generative AI library not installed. Install with: pip install google-generativeai"
    
    if not api_key or not api_key.strip():
        return "⚠️ Please enter your Google Gemini API key in the sidebar."
    
    try:
        # Configure Gemini
        genai.configure(api_key=api_key.strip())
        
        # Generation config
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 2048,  # Increased to allow longer responses
        }
        
        # Try to get list of available models first
        available_model_name = None
        try:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    model_name_short = m.name.split('/')[-1]  # Get just the model name part
                    # Prefer models that match our selection or common names
                    if model in m.name or 'gemini' in model_name_short.lower():
                        available_model_name = model_name_short
                        break
            # If no match found, use first available gemini model
            if not available_model_name:
                for m in genai.list_models():
                    if 'generateContent' in m.supported_generation_methods and 'gemini' in m.name.lower():
                        available_model_name = m.name.split('/')[-1]
                        break
        except Exception as list_error:
            # If listing fails, fall back to trying common names
            pass
        
        # Use available model if found, otherwise try fallbacks
        model_to_try = available_model_name if available_model_name else model
        
        # List of models to try in order
        fallback_models = [model_to_try, "gemini-pro", "gemini-1.5-flash", "gemini-1.5-pro"]
        if model not in fallback_models:
            fallback_models.insert(0, model)
        
        gemini_model = None
        last_error = None
        
        for model_name in fallback_models:
            try:
                gemini_model = genai.GenerativeModel(
                    model_name=model_name,
                    generation_config=generation_config
                )
                model_to_use = model_name
                break
            except Exception as model_error:
                last_error = str(model_error)
                continue
        
        if gemini_model is None:
            return f"⚠️ Could not initialize any Gemini model.\n\nError: {last_error}\n\nPlease check:\n1. Your API key is correct\n2. The Gemini API is enabled in Google Cloud Console\n3. Try selecting a different model from the dropdown"
        
        # Build system instruction - specialized for depression and drug abuse support
        system_instruction = """You are a compassionate, empathetic, and highly trained AI assistant specializing in mental health support, particularly for depression and substance abuse. 

Your role is to:
- Provide non-judgmental, understanding, and supportive responses to people struggling with depression, drug abuse, or both
- Recognize the complex relationship between depression and substance use disorders
- Offer empathetic listening and validation of their feelings and experiences
- Provide helpful, evidence-based guidance when appropriate
- Never minimize their struggles or give generic advice
- Respond directly and personally to what they share - never ask for names or give generic greetings
- Understand that substance use often co-occurs with depression as a coping mechanism
- Be sensitive to the shame, guilt, and isolation that often accompanies these struggles
- Encourage seeking professional help when appropriate, but do so gently and supportively
- Acknowledge the courage it takes to reach out and share these struggles

Remember: People dealing with depression and substance abuse need understanding, not judgment. They need to feel heard and validated. Respond with genuine care and empathy, meeting them where they are in their journey."""
        
        # Build conversation history for Gemini
        # Gemini uses a different format - we need to build the conversation context
        conversation_parts = []
        
        # Add chat history if available (last 10 messages for context)
        if chat_history and len(chat_history) > 0:
            for msg in chat_history[-10:]:
                if msg['role'] == 'user':
                    conversation_parts.append({"role": "user", "parts": [msg['content']]})
                elif msg['role'] == 'assistant':
                    conversation_parts.append({"role": "model", "parts": [msg['content']]})
        
        # Add current user message
        conversation_parts.append({"role": "user", "parts": [user_message]})
        
        # Start a chat session with history
        if len(conversation_parts) > 1:
            # We have history, use chat
            chat = gemini_model.start_chat(history=conversation_parts[:-1])
            response = chat.send_message(conversation_parts[-1]["parts"][0])
        else:
            # First message, use generate_content with system instruction
            # For first message, include system instruction in the prompt
            prompt = f"{system_instruction}\n\nUser: {user_message}\nAssistant:"
            response = gemini_model.generate_content(prompt)
        
        # Handle response - check for safety blocks and extract text properly
        if not response:
            return "⚠️ No response from the AI assistant."
        
        # Check if response was blocked by safety filters
        if hasattr(response, 'candidates') and response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            
            # Check finish_reason (can be enum or string)
            finish_reason = None
            if hasattr(candidate, 'finish_reason'):
                finish_reason = candidate.finish_reason
                # Handle both enum and string formats
                if hasattr(finish_reason, 'name'):
                    finish_reason = finish_reason.name
                elif isinstance(finish_reason, int):
                    finish_reason_map = {1: "STOP", 2: "MAX_TOKENS", 3: "SAFETY", 4: "RECITATION", 5: "OTHER"}
                    finish_reason = finish_reason_map.get(finish_reason, f"UNKNOWN_{finish_reason}")
            
            # Check if content exists and has parts FIRST (even if finish_reason is not STOP)
            if hasattr(candidate, 'content') and candidate.content:
                if hasattr(candidate.content, 'parts') and candidate.content.parts:
                    parts = candidate.content.parts
                    if len(parts) > 0 and hasattr(parts[0], 'text') and parts[0].text:
                        # Return the text even if it was cut off
                        text = parts[0].text.strip()
                        if finish_reason and (finish_reason == "MAX_TOKENS" or finish_reason == 2):
                            # Add a note if it was cut off, but still return the text
                            return text + "\n\n(Note: Response was cut off due to length limit)"
                        return text
            
            # Only show error if we couldn't extract any text
            if finish_reason and finish_reason != "STOP" and finish_reason != 1:
                if finish_reason == "SAFETY" or finish_reason == 3:
                    return "⚠️ The response was blocked by safety filters. Please try rephrasing your message."
                elif finish_reason == "RECITATION" or finish_reason == 4:
                    return "⚠️ The response was blocked due to content matching. Please try a different message."
                elif finish_reason == "MAX_TOKENS" or finish_reason == 2:
                    return "⚠️ Response was cut off (max tokens reached) and no text was returned."
                else:
                    return f"⚠️ Response incomplete. Finish reason: {finish_reason}"
        
        # Try to extract text from response
        response_text = None
        
        # Method 1: Try response.text (works if response has parts)
        try:
            if hasattr(response, 'text'):
                response_text = response.text
        except Exception:
            pass
        
        # Method 2: Access via candidates[0].content.parts[0].text
        if not response_text:
            try:
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content') and candidate.content:
                        if hasattr(candidate.content, 'parts') and candidate.content.parts:
                            if len(candidate.content.parts) > 0:
                                part = candidate.content.parts[0]
                                if hasattr(part, 'text'):
                                    response_text = part.text
            except Exception:
                pass
        
        # Method 3: Try to get text from response.parts directly
        if not response_text:
            try:
                if hasattr(response, 'parts') and response.parts:
                    if len(response.parts) > 0 and hasattr(response.parts[0], 'text'):
                        response_text = response.parts[0].text
            except Exception:
                pass
        
        if response_text:
            return response_text.strip()
        else:
            # Check if there's a prompt_feedback with block reason
            if hasattr(response, 'prompt_feedback'):
                if hasattr(response.prompt_feedback, 'block_reason'):
                    return f"⚠️ Prompt was blocked. Reason: {response.prompt_feedback.block_reason}"
            return "⚠️ No response text available. The API returned an empty response."
            
    except Exception as e:
        error_str = str(e)
        # More detailed error checking
        if "api_key" in error_str.lower() or "authentication" in error_str.lower() or "invalid" in error_str.lower() or "invalid api key" in error_str.lower() or "401" in error_str.lower():
            # Check if API key looks valid (Gemini keys usually start with AIza)
            if api_key and api_key.strip().startswith("AIza"):
                return f"⚠️ API key authentication failed. Error: {error_str[:200]}\n\nPlease verify:\n1. The API key is correct and complete\n2. The Gemini API is enabled in your Google Cloud Console\n3. The API key has not been revoked"
            else:
                return f"⚠️ Invalid API key format. Gemini API keys usually start with 'AIza...'\n\nCurrent key starts with: {api_key[:10] if api_key and len(api_key) > 10 else 'empty'}\n\nPlease check your API key in the sidebar."
        elif "rate limit" in error_str.lower() or "quota" in error_str.lower():
            return "⚠️ Rate limit or quota exceeded. Please wait a moment and try again."
        elif "permission" in error_str.lower() or "not enabled" in error_str.lower():
            return "⚠️ API not enabled. Please enable the Gemini API in your Google Cloud Console."
        elif "not found" in error_str.lower() or "404" in error_str.lower():
            # Try to find a working model automatically
            try:
                genai.configure(api_key=api_key.strip())
                # Try common model names
                for fallback_model in ["gemini-pro", "gemini-1.5-flash", "gemini-1.5-pro"]:
                    try:
                        test_model = genai.GenerativeModel(fallback_model)
                        # If we get here, the model works - retry with this model
                        return query_gemini(user_message, api_key, fallback_model, chat_history)
                    except:
                        continue
            except:
                pass
            return "⚠️ Model not found. Please try selecting a different model from the dropdown in the sidebar."
        else:
            return f"⚠️ Error: {error_str}"

def main():
    # Header with improved design
    st.markdown("""
    <div style="text-align: center; padding: 2.5rem 0 2rem 0;">
        <h1 style="font-family: 'Poppins', sans-serif; font-size: 3.2rem; font-weight: 800; 
                   background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                   -webkit-background-clip: text;
                   -webkit-text-fill-color: transparent;
                   background-clip: text;
                   margin-bottom: 0.8rem; letter-spacing: -0.02em;">💬 AI Chatbot</h1>
        <p style="color: rgba(232, 232, 232, 0.85); font-size: 1.15rem; margin-top: 0.5rem; line-height: 1.6;">
            Your supportive companion for mental health conversations
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration with cleaner design
    st.sidebar.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 1.5rem; border-radius: 16px; margin-bottom: 2rem; 
                box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);">
        <h2 style="color: #ffffff !important; margin: 0; font-size: 1.6rem; font-weight: 700; font-family: 'Poppins', sans-serif;">⚙️ Settings</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("### 🔑 API Configuration")
    st.sidebar.markdown("""
    <p style="color: rgba(232, 232, 232, 0.7); font-size: 0.85rem; margin-bottom: 1rem;">
        Configure your API settings to start chatting
    </p>
    """, unsafe_allow_html=True)
    
    # API Key input
    default_api_key = st.session_state.get('gemini_api_key', '')
    
    api_key = st.sidebar.text_input(
        "Google Gemini API Key",
        type="password",
        help="Enter your Google Gemini API key. Get one at https://makersuite.google.com/app/apikey",
        value=default_api_key,
        label_visibility="visible"
    )
    st.session_state.gemini_api_key = api_key
    
    # Model selection - try common Gemini model names
    # Note: Available models may vary by API key/region
    gemini_model = st.sidebar.selectbox(
        "AI Model",
        ["gemini-pro", "gemini-1.5-flash", "gemini-1.5-pro", "models/gemini-pro"],
        index=0,
        help="Choose the Gemini model for conversations. If one doesn't work, try another."
    )
    
    # Initialize chat history in session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Chat container styles - friendly and beautiful design
    st.markdown("""
    <style>
        .chat-container {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.08) 0%, rgba(118, 75, 162, 0.08) 100%);
            border-radius: 32px;
            padding: 3rem;
            margin: 2rem 0;
            border: 2px solid rgba(168, 237, 234, 0.2);
            backdrop-filter: blur(20px);
            max-height: 700px;
            overflow-y: auto;
            min-height: 500px;
            box-shadow: 
                0 20px 60px rgba(102, 126, 234, 0.15),
                0 0 0 1px rgba(255, 255, 255, 0.05),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
            position: relative;
        }
        
        .chat-container::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #667eea 100%);
            border-radius: 32px 32px 0 0;
            opacity: 0.6;
        }
        
        .chat-container::-webkit-scrollbar {
            width: 10px;
        }
        
        .chat-container::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            margin: 10px;
        }
        
        .chat-container::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            border: 2px solid rgba(0, 0, 0, 0.1);
        }
        
        .chat-container::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(135deg, #7a8ef5 0%, #8a5fb8 100%);
        }
        
        .chat-message {
            margin-bottom: 2rem;
            animation: fadeInUp 0.5s ease-out;
            display: flex;
            flex-direction: column;
        }
        
        @keyframes fadeInUp {
            from { 
                opacity: 0; 
                transform: translateY(20px) scale(0.95); 
            }
            to { 
                opacity: 1; 
                transform: translateY(0) scale(1); 
            }
        }
        
        .user-message {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.4rem 2.2rem;
            border-radius: 28px 28px 8px 28px;
            margin-left: auto;
            margin-right: 0;
            max-width: 75%;
            box-shadow: 
                0 8px 25px rgba(102, 126, 234, 0.4),
                0 0 0 1px rgba(255, 255, 255, 0.1);
            font-size: 1.05rem;
            line-height: 1.7;
            word-wrap: break-word;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
        }
        
        .user-message::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, transparent 100%);
            border-radius: 28px 28px 8px 28px;
            pointer-events: none;
        }
        
        .user-message:hover {
            transform: translateY(-3px) scale(1.01);
            box-shadow: 
                0 12px 35px rgba(102, 126, 234, 0.5),
                0 0 0 1px rgba(255, 255, 255, 0.15);
        }
        
        .assistant-message {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.15) 0%, rgba(255, 255, 255, 0.08) 100%);
            color: #e8e8e8;
            padding: 1.4rem 2.2rem;
            border-radius: 28px 28px 28px 8px;
            margin-left: 0;
            margin-right: auto;
            max-width: 75%;
            border: 2px solid rgba(168, 237, 234, 0.3);
            font-size: 1.05rem;
            line-height: 1.7;
            word-wrap: break-word;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            box-shadow: 
                0 8px 25px rgba(0, 0, 0, 0.2),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
        }
        
        .assistant-message::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(135deg, rgba(168, 237, 234, 0.1) 0%, transparent 100%);
            border-radius: 28px 28px 28px 8px;
            pointer-events: none;
        }
        
        .assistant-message:hover {
            transform: translateY(-3px) scale(1.01);
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.18) 0%, rgba(255, 255, 255, 0.1) 100%);
            border-color: rgba(168, 237, 234, 0.4);
            box-shadow: 
                0 12px 35px rgba(0, 0, 0, 0.25),
                inset 0 1px 0 rgba(255, 255, 255, 0.15);
        }
        
        .message-label {
            font-size: 0.7rem;
            font-weight: 700;
            margin-bottom: 0.6rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            opacity: 0.85;
        }
        
        .user-label {
            color: rgba(255, 255, 255, 0.9);
            text-align: right;
        }
        
        .assistant-label {
            color: #a8edea;
            font-size: 0.75rem;
        }
        
        .message-content {
            line-height: 1.7;
            font-size: 1.05rem;
            position: relative;
            z-index: 1;
        }
        
        .chat-input-container {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
            padding: 2rem;
            border-radius: 28px;
            border: 2px solid rgba(168, 237, 234, 0.25);
            backdrop-filter: blur(20px);
            margin-top: 2rem;
            box-shadow: 
                0 10px 40px rgba(102, 126, 234, 0.2),
                0 0 0 1px rgba(255, 255, 255, 0.05),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
            transition: all 0.3s ease;
        }
        
        .chat-input-container:hover {
            border-color: rgba(168, 237, 234, 0.35);
            box-shadow: 
                0 15px 50px rgba(102, 126, 234, 0.25),
                0 0 0 1px rgba(255, 255, 255, 0.08);
        }
        
        .empty-state {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 6rem 3rem;
            text-align: center;
            position: relative;
        }
        
        .empty-state-icon {
            font-size: 5rem;
            margin-bottom: 2rem;
            opacity: 0.8;
            animation: float 3s ease-in-out infinite;
            filter: drop-shadow(0 10px 20px rgba(102, 126, 234, 0.3));
        }
        
        @keyframes float {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
        }
        
        .empty-state-title {
            font-size: 1.6rem;
            color: rgba(232, 232, 232, 0.95);
            margin-bottom: 1rem;
            font-weight: 700;
            background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .empty-state-subtitle {
            font-size: 1.05rem;
            color: rgba(232, 232, 232, 0.75);
            line-height: 1.8;
            max-width: 500px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Input field styling
    st.markdown("""
    <style>
        .stTextInput > div > div > input {
            background: rgba(255, 255, 255, 0.1) !important;
            border: 2px solid rgba(168, 237, 234, 0.3) !important;
            border-radius: 20px !important;
            padding: 1rem 1.5rem !important;
            color: #ffffff !important;
            font-size: 1rem !important;
            transition: all 0.3s ease !important;
        }
        
        .stTextInput > div > div > input:focus {
            background: rgba(255, 255, 255, 0.15) !important;
            border-color: rgba(168, 237, 234, 0.5) !important;
            box-shadow: 0 0 0 4px rgba(168, 237, 234, 0.2) !important;
            outline: none !important;
        }
        
        .stTextInput > div > div > input::placeholder {
            color: rgba(255, 255, 255, 0.5) !important;
        }
        
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            font-weight: 700 !important;
            font-size: 1rem !important;
            padding: 1rem 2rem !important;
            border-radius: 20px !important;
            border: none !important;
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px) scale(1.02) !important;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.5) !important;
            background: linear-gradient(135deg, #7a8ef5 0%, #8a5fb8 100%) !important;
        }
        
        .stButton > button:active {
            transform: translateY(0) scale(0.98) !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Display chat history with improved empty state
    if st.session_state.chat_history:
        # Build complete HTML string in one go
        import html
        import re
        
        chat_html_parts = ['<div class="chat-container">']
        
        for msg in st.session_state.chat_history:
            # Get the content and clean it
            raw_content = str(msg['content'])
            # Remove any HTML tags that might be in the content
            raw_content = re.sub(r'<[^>]+>', '', raw_content)
            # Escape HTML special characters to prevent XSS
            escaped_content = html.escape(raw_content)
            # Convert newlines to <br> tags AFTER escaping
            formatted_content = escaped_content.replace('\n', '<br>')
            
            if msg['role'] == 'user':
                chat_html_parts.append(f'''<div class="chat-message">
<div class="message-label user-label">You</div>
<div class="user-message">
<div class="message-content">{formatted_content}</div>
</div>
</div>''')
            else:
                chat_html_parts.append(f'''<div class="chat-message">
<div class="message-label assistant-label">🤖 AI Assistant</div>
<div class="assistant-message">
<div class="message-content">{formatted_content}</div>
</div>
</div>''')
        
        chat_html_parts.append('</div>')
        chat_html = ''.join(chat_html_parts)
        
        st.markdown(chat_html, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="chat-container">
            <div class="empty-state">
                <div class="empty-state-icon">✨</div>
                <div class="empty-state-title">Welcome! I'm here to chat with you</div>
                <div class="empty-state-subtitle">
                    I'm your friendly AI companion, ready to listen and support you. Whether you want to talk about mental health, ask questions, or just have a friendly conversation - I'm here for you. 💙
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Chat input container with improved layout
    st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
    
    # Input row
    col_input1, col_input2 = st.columns([5, 1])
    
    with col_input1:
        user_message = st.text_input(
            "",
            placeholder="Type your message here...",
            key="chat_input",
            label_visibility="collapsed"
        )
    
    with col_input2:
        send_button = st.button("Send", type="primary", use_container_width=True, key="chat_send")
    
    # Action buttons row
    if st.session_state.chat_history:
        col_clear, col_spacer = st.columns([1, 4])
        with col_clear:
            clear_button = st.button("🗑️ Clear Chat", use_container_width=True, key="clear_chat", help="Clear conversation history")
            if clear_button:
                st.session_state.chat_history = []
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Helpful tips with beautiful design
    if not st.session_state.chat_history:
        st.markdown("""
        <div style="margin-top: 2rem; padding: 1.8rem; 
                    background: linear-gradient(135deg, rgba(168, 237, 234, 0.12) 0%, rgba(254, 214, 227, 0.12) 100%);
                    border-radius: 24px; 
                    border: 2px solid rgba(168, 237, 234, 0.25);
                    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.15),
                                0 0 0 1px rgba(255, 255, 255, 0.05),
                                inset 0 1px 0 rgba(255, 255, 255, 0.1);
                    backdrop-filter: blur(10px);">
            <p style="color: #a8edea; font-size: 1rem; margin: 0 0 1rem 0; font-weight: 700; 
                      display: flex; align-items: center; gap: 0.5rem;">
                <span style="font-size: 1.3rem;">💡</span>
                <span>Tips for better conversations</span>
            </p>
            <ul style="color: rgba(232, 232, 232, 0.85); font-size: 0.9rem; margin: 0; 
                      padding-left: 1.5rem; line-height: 2.2; list-style: none;">
                <li style="margin-bottom: 0.5rem;">✨ Be open and honest about your feelings</li>
                <li style="margin-bottom: 0.5rem;">💭 Ask specific questions if you need help</li>
                <li>⏰ Take your time - there's no rush</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Handle chat
    if send_button and user_message.strip():
        # Add user message to history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_message.strip()
        })
        
        # Query AI
        with st.spinner("🤖 Thinking..."):
            response = query_gemini(
                user_message.strip(),
                api_key,
                gemini_model,
                st.session_state.chat_history
            )
        
        # Add assistant response to history
        # Clean the response to ensure it's plain text (remove any accidental HTML)
        import re
        clean_response = str(response)
        # Remove any HTML tags
        clean_response = re.sub(r'<[^>]+>', '', clean_response)
        # Remove any HTML entities that might have been double-encoded
        import html
        clean_response = html.unescape(clean_response)
        # Final cleanup
        clean_response = clean_response.strip()
        
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': clean_response
        })
        
        # Rerun to show new messages
        st.rerun()

if __name__ == "__main__":
    main()

