#!/usr/bin/env python
"""Test different Claude model names to find what's available"""

import os
from pathlib import Path
from dotenv import load_dotenv
from anthropic import Anthropic

# Load environment variables from parent directory
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

api_key = os.getenv('CLAUDE_API_KEY')

if not api_key:
    print("❌ CLAUDE_API_KEY not found in .env")
    exit(1)

# List of Claude models to try (in order of preference)
models_to_try = [
    'claude-3-5-sonnet-20241022',      # Most recent Sonnet
    'claude-3-5-sonnet-latest',         # Latest alias
    'claude-opus-4-1-20250805',        # Opus variant (if available)
    'claude-3-opus-20250729',          # Newer Opus
    'claude-3-5-haiku-20241022',       # Haiku (lighter)
    'claude-opus-4-turbo-20250514',    # Turbo variant
    'claude-3-haiku-20240307',         # Older Haiku
]

print("="*80)
print("🔍 Testing Available Claude Models")
print("="*80)

client = Anthropic(api_key=api_key)
working_model = None

for model_name in models_to_try:
    try:
        print(f"\n📍 Trying: {model_name}...", end=" ")
        message = client.messages.create(
            model=model_name,
            max_tokens=10,
            messages=[{'role': 'user', 'content': 'Hi'}]
        )
        print(f"✅ WORKS!")
        print(f"   Response: {message.content[0].text.strip()}")
        working_model = model_name
        break
    except Exception as e:
        error_msg = str(e)
        if '404' in error_msg or 'not found' in error_msg.lower():
            print(f"❌ Not found")
        elif 'invalid_request_error' in error_msg.lower():
            print(f"❌ Invalid")
        else:
            print(f"❌ {type(e).__name__}")

print("\n" + "="*80)
if working_model:
    print(f"✅ RECOMMENDED MODEL: {working_model}")
    print("\nUpdate your code with:")
    print(f'    model="{working_model}",')
else:
    print("❌ No working Claude model found!")
    print("\nTroubleshooting:")
    print("1. Check your CLAUDE_API_KEY is valid")
    print("2. Check your account has access to Claude models")
    print("3. Try visiting https://console.anthropic.com to verify your account")
print("="*80)
