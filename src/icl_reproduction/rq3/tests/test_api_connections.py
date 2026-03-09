#!/usr/bin/env python
"""Test API connections for Gemini, Claude, and GPT"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from parent directory
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

print("="*80)
print("🔌 Testing API Connections")
print("="*80)

# Test Gemini
print("\n📍 Testing GEMINI...")
try:
    from google import genai
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        print("   ❌ GEMINI_API_KEY not found in .env")
    else:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents='Say "hello" in one word only'
        )
        print(f"   ✅ SUCCESS! Response: {response.text.strip()}")
except Exception as e:
    print(f"   ❌ FAILED: {type(e).__name__}: {str(e)[:100]}")

# Test Claude
print("\n📍 Testing CLAUDE...")
try:
    from anthropic import Anthropic
    api_key = os.getenv('CLAUDE_API_KEY')
    
    if not api_key:
        print("   ❌ CLAUDE_API_KEY not found in .env")
    else:
        client = Anthropic(api_key=api_key)
        message = client.messages.create(
            model='claude-opus-4-1-20250805',
            max_tokens=20,
            messages=[{'role': 'user', 'content': 'Say "hello" in one word only'}]
        )
        print(f"   ✅ SUCCESS! Response: {message.content[0].text.strip()}")
except Exception as e:
    print(f"   ❌ FAILED: {type(e).__name__}: {str(e)[:100]}")

# Test GPT
print("\n📍 Testing GPT...")
try:
    from openai import OpenAI
    api_key = os.getenv('GPT_API_KEY')
    
    if not api_key:
        print("   ❌ GPT_API_KEY not found in .env")
    else:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model='gpt-4o-mini',
            max_tokens=20,
            messages=[{'role': 'user', 'content': 'Say "hello" in one word only'}]
        )
        print(f"   ✅ SUCCESS! Response: {response.choices[0].message.content.strip()}")
except Exception as e:
    print(f"   ❌ FAILED: {type(e).__name__}: {str(e)[:100]}")

print("\n" + "="*80)
print("Test complete!")
print("="*80)
