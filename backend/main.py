from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import subprocess
import json
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserMessage(BaseModel):
    message: str
    mood: Optional[int] = None  # Mood scale 1-10
    conversation_history: Optional[list] = None  # For context

risky_keywords = ["alcohol", "drink", "smoke", "drugs", "weed", "cocaine",
                  "anxious", "stress", "depressed", "sad", "overwhelmed"]

def analyze_risk(user_input: str):
    detected = [w for w in risky_keywords if w in user_input.lower()]
    score = min(len(detected) * 0.4, 1.0)
    if score <= 0.1:
        level = "Green (Low Risk)"
    elif score <= 0.6:
        level = "Yellow (Medium Risk)"
    else:
        level = "Red (High Risk)"
    return {"keywords_detected": detected, "risk_score": score, "risk_level": level}

def fix_generic_response(response: str, user_message: str) -> str:
    """Fix generic greetings and make responses more specific to user input."""
    user_lower = user_message.lower().strip()
    response_lower = response.lower()
    
    # Check if user shared feelings/problems but got generic greeting
    feeling_keywords = ["stressed", "depressed", "sad", "anxious", "worried", "overwhelmed", "tired", "exhausted", "hurt", "angry", "frustrated", "down", "upset"]
    generic_greetings = ["nice to meet you", "what should i call you", "what's your name", "thanks for trusting me with your name", "thanks for trusting me"]
    name_request_patterns = ["what's your name", "what should i call", "tell me your name"]
    
    has_feelings = any(keyword in user_lower for keyword in feeling_keywords)
    has_generic = any(greeting in response_lower for greeting in generic_greetings)
    has_name_request = any(pattern in response_lower for pattern in name_request_patterns)
    
    # CRITICAL: If response treats user message as a name (e.g., "Hello im stressed!"), replace it
    # Check if response starts with greeting + user message treated as name
    if has_feelings and (user_lower in response_lower[:100] and ("hello" in response_lower[:30] or "hey" in response_lower[:30] or "hi" in response_lower[:30])):
        detected_feeling = None
        for keyword in feeling_keywords:
            if keyword in user_lower:
                detected_feeling = keyword
                break
        
        if detected_feeling:
            if "depressed" in user_lower or "sad" in user_lower:
                return f"I hear that you're feeling {detected_feeling}, and I want you to know that your feelings are valid. That sounds really difficult. Can you tell me more about what's been going on?"
            elif "stressed" in user_lower or "anxious" in user_lower or "worried" in user_lower:
                return f"I'm sorry to hear you're feeling {detected_feeling}. That can be really overwhelming. What's been causing you to feel this way?"
            else:
                return f"I hear that you're feeling {detected_feeling}. That must be really hard. Can you tell me more about what's going on?"
    
    # If user shared feelings but got generic response or name request
    if has_feelings and (has_generic or has_name_request):
        detected_feeling = None
        for keyword in feeling_keywords:
            if keyword in user_lower:
                detected_feeling = keyword
                break
        
        if detected_feeling:
            if "depressed" in user_lower or "sad" in user_lower:
                return f"I hear that you're feeling {detected_feeling}, and I want you to know that your feelings are valid. That sounds really difficult. Can you tell me more about what's been going on?"
            elif "stressed" in user_lower or "anxious" in user_lower or "worried" in user_lower:
                return f"I'm sorry to hear you're feeling {detected_feeling}. That can be really overwhelming. What's been causing you to feel this way?"
            else:
                return f"I hear that you're feeling {detected_feeling}. That must be really hard. Can you tell me more about what's going on?"
    
    # If response asks for name when user just greeted
    if has_name_request and (user_lower in ["hi", "hello", "hey", "yo", "sup"] or len(user_lower.split()) <= 2):
        return "Hi, I'm Luma. How are you feeling today?"
    
    # If response starts with generic greeting but user just said "hi", that's okay but keep it brief
    if "hi" in user_lower or "yo" in user_lower or "hey" in user_lower:
        if len(user_lower.split()) <= 2:  # Simple greeting
            if has_name_request or (len(response) > 120 and has_generic):
                return "Hi, I'm Luma. How are you feeling today?"
    
    return response

def get_llm_response(user_message: str, mood: Optional[int] = None, conversation_history: Optional[list] = None):
    try:
        # Build system prompt for mental health assistant - CRITICAL: Must respond directly to user input
        system_prompt = """You are Luma, a compassionate mental health assistant.

ABSOLUTE CRITICAL RULES - FOLLOW THESE EXACTLY:
1. NEVER ask for the user's name. NEVER say "What's your name?" or "What should I call you?"
2. NEVER treat emotional statements as names. If user says "im stressed" or "I'm depressed", they are sharing feelings, NOT giving you a name.
3. NEVER say "Nice to meet you [name]" or "Thanks for trusting me with your name" - these are WRONG.
4. ALWAYS respond directly to what the user actually says. If they share a feeling, acknowledge that feeling immediately.
5. If user says "hi", "yo", "hello" - respond briefly: "Hi, I'm Luma. How are you feeling today?"
6. If user shares feelings like "stressed", "depressed", "sad", "anxious" - immediately acknowledge: "I hear that you're feeling [feeling]. That sounds really difficult. Can you tell me more?"

CORRECT EXAMPLES:
User: "yo" → Luma: "Hi, I'm Luma. How are you feeling today?"
User: "im stressed" → Luma: "I'm sorry to hear you're feeling stressed. That can be really overwhelming. What's been causing you stress?"
User: "I'm depressed" → Luma: "I hear that you're feeling depressed, and I want you to know that your feelings are valid. Can you tell me more about what's been going on?"
User: "hi" → Luma: "Hi, I'm Luma. How are you feeling today?"

WRONG EXAMPLES (DO NOT DO THIS):
User: "im stressed" → WRONG: "Hello im stressed! Thanks for trusting me with your name..."
User: "yo" → WRONG: "Hey! What's your name?"
User: "hi" → WRONG: "Hi there! What should I call you?"

Remember: The user is sharing their emotional state, NOT giving you a name. Respond to their feelings, not with name requests."""

        # Build context from conversation history
        context_parts = []
        if conversation_history and len(conversation_history) > 0:
            context_parts.append("\nPrevious conversation:")
            for msg in conversation_history[-6:]:  # Last 6 messages for context
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    context_parts.append(f"User: {content}")
                else:
                    context_parts.append(f"Luma: {content}")
        
        # Add mood context if available
        if mood is not None:
            if mood <= 3:
                context_parts.append(f"\n[User's mood is very low: {mood}/10. Be extra supportive and empathetic.]")
            elif mood <= 5:
                context_parts.append(f"\n[User's mood is moderate-low: {mood}/10. They may be feeling down or stressed.]")
            elif mood <= 7:
                context_parts.append(f"\n[User's mood is moderate: {mood}/10.]")
            else:
                context_parts.append(f"\n[User's mood is positive: {mood}/10.]")
        
        # Build the full prompt with clear instructions
        context_text = "\n".join(context_parts) if context_parts else ""
        
        # Use a more direct prompt format that works better with Ollama
        full_prompt = f"""{system_prompt}

{context_text}

Current user message: {user_message}

Now respond as Luma (remember: respond directly to what they said, no generic greetings):"""
        
        # Try using Ollama API first (more reliable)
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3",
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                    }
                },
                timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "").strip()
                # Post-process to fix generic responses
                response_text = fix_generic_response(response_text, user_message)
                return response_text
        except Exception as api_error:
            print(f"API error, falling back to subprocess: {api_error}")
        
        # Fallback to subprocess if API fails
        result = subprocess.run(
            ["ollama", "run", "llama3", full_prompt],
            capture_output=True, text=True, check=True, timeout=30
        )
        response_text = result.stdout.strip()
        
        # Post-process to fix generic responses
        response_text = fix_generic_response(response_text, user_message)
        return response_text
    except subprocess.TimeoutExpired:
        print("LLM timeout")
        return "I'm taking a moment to think about how best to respond. Could you tell me more about what you're experiencing?"
    except Exception as e:
        print(f"LLM error: {e}")
        return "Sorry, I couldn't generate a response right now. Please try again."

@app.post("/chat")
async def chat(user_msg: UserMessage):
    risk_info = analyze_risk(user_msg.message)
    
    # Get LLM response with mood and conversation context
    llm_text = get_llm_response(
        user_msg.message,
        mood=user_msg.mood,
        conversation_history=user_msg.conversation_history
    )

    # Add followup suggestions based on risk level
    followup = ""
    if risk_info["risk_level"].startswith("Yellow"):
        followup = " You might consider mindfulness exercises, journaling, or light physical activity."
    elif risk_info["risk_level"].startswith("Red"):
        followup = " If you're in immediate danger or having thoughts of self-harm, please contact a crisis helpline or emergency services right away. For ongoing support, consider speaking with a mental health professional."

    full_response = f"{llm_text}{followup}".strip()

    return {
        "response": full_response,
        "risk_level": risk_info["risk_level"],
        "explanation": risk_info
    }
