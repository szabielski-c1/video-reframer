#!/usr/bin/env python3
"""
Test script to try Vertex AI approach with permissive safety settings
"""

import asyncio
import sys
import os
import json
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_vertex_ai_approach():
    """Test Vertex AI with permissive safety settings"""
    print("\n🔧 Testing Vertex AI approach with safety settings...")

    try:
        from vertexai.generative_models import (
            GenerativeModel,
            HarmCategory,
            HarmBlockThreshold,
            Part,
            SafetySetting,
            GenerationConfig
        )
        print("✓ Vertex AI imports successful")
    except ImportError as e:
        print(f"❌ Vertex AI not available: {e}")
        print("Installing google-cloud-aiplatform...")

        import subprocess
        result = subprocess.run(
            ["pip", "install", "google-cloud-aiplatform"],
            capture_output=True, text=True
        )

        if result.returncode == 0:
            print("✓ google-cloud-aiplatform installed")
            try:
                from vertexai.generative_models import (
                    GenerativeModel,
                    HarmCategory,
                    HarmBlockThreshold,
                    Part,
                    SafetySetting,
                    GenerationConfig
                )
                print("✓ Vertex AI imports successful after installation")
            except ImportError as e2:
                print(f"❌ Still can't import Vertex AI: {e2}")
                return False
        else:
            print(f"❌ Failed to install google-cloud-aiplatform: {result.stderr}")
            return False

    try:
        # Initialize Vertex AI (this might require project setup)
        import vertexai

        # Try to initialize with a project (you may need to set this)
        # For now, let's try without explicit project initialization

        # Safety config with BLOCK_NONE for all categories
        safety_config = [
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=HarmBlockThreshold.BLOCK_NONE,
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=HarmBlockThreshold.BLOCK_NONE,
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=HarmBlockThreshold.BLOCK_NONE,
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=HarmBlockThreshold.BLOCK_NONE,
            ),
        ]

        print("✓ Safety settings configured with BLOCK_NONE")

        # Test with different models
        models_to_test = ["gemini-1.5-flash", "gemini-2.5-flash", "gemini-1.5-pro"]

        for model_name in models_to_test:
            print(f"\n  Testing model: {model_name}")

            try:
                model = GenerativeModel(model_name)

                # Simple test prompt
                test_prompt = "Analyze this video content and describe what you see. Focus on visual elements, people, and activities. Respond with: 'Test successful - no safety issues detected.'"

                response = model.generate_content(
                    test_prompt,
                    stream=False,
                    safety_settings=safety_config
                )

                if response and response.text:
                    print(f"    ✅ Success: {response.text[:100]}...")

                    # Check finish reason
                    if hasattr(response, 'candidates') and response.candidates:
                        candidate = response.candidates[0]
                        if hasattr(candidate, 'finish_reason'):
                            print(f"    Finish reason: {candidate.finish_reason}")
                        else:
                            print(f"    No finish_reason (good sign)")

                    return True
                else:
                    print(f"    ⚠️ No response text")

            except Exception as e:
                error_str = str(e)
                print(f"    ❌ Error: {error_str}")

                if "project" in error_str.lower():
                    print(f"    💡 Hint: You may need to set up a Google Cloud project")
                elif "authentication" in error_str.lower():
                    print(f"    💡 Hint: You may need to authenticate with Google Cloud")

        return False

    except Exception as e:
        print(f"❌ Vertex AI test failed: {str(e)}")
        return False

async def test_google_generativeai_with_safety():
    """Test the regular google-generativeai with explicit safety settings"""
    print("\n🔧 Testing google-generativeai with explicit safety settings...")

    try:
        import google.generativeai as genai

        # Configure with API key
        genai.configure(api_key=settings.GEMINI_API_KEY)

        # Safety settings for google-generativeai
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        models_to_test = ["gemini-2.5-flash", "gemini-1.5-flash"]

        for model_name in models_to_test:
            print(f"\n  Testing model: {model_name}")

            try:
                model = genai.GenerativeModel(model_name)

                test_prompt = "Analyze video content for technical reframing purposes. Respond with: 'Test successful - no safety issues detected.'"

                response = model.generate_content(
                    test_prompt,
                    safety_settings=safety_settings
                )

                if response and response.text:
                    print(f"    ✅ Success: {response.text[:100]}...")

                    # Check finish reason
                    if hasattr(response, 'candidates') and response.candidates:
                        candidate = response.candidates[0]
                        if hasattr(candidate, 'finish_reason'):
                            finish_reason = candidate.finish_reason
                            print(f"    Finish reason: {finish_reason}")
                            if finish_reason == 2:  # SAFETY
                                print(f"    ⚠️ Still blocked by safety filters")
                            else:
                                print(f"    ✅ No safety blocks")
                                return True
                        else:
                            print(f"    ✅ No finish_reason (good)")
                            return True
                else:
                    print(f"    ⚠️ No response text")

            except Exception as e:
                print(f"    ❌ Error: {str(e)}")

        return False

    except Exception as e:
        print(f"❌ google-generativeai test failed: {str(e)}")
        return False

async def main():
    """Run all tests"""
    print("🧪 Testing alternative approaches to bypass safety filters")
    print(f"📋 Using API key: {settings.GEMINI_API_KEY[:8]}...{settings.GEMINI_API_KEY[-8:]}")

    # Test 1: Vertex AI approach
    vertex_works = await test_vertex_ai_approach()

    # Test 2: google-generativeai with explicit safety settings
    genai_works = await test_google_generativeai_with_safety()

    # Summary
    print("\n📊 TEST SUMMARY:")
    print(f"   Vertex AI Approach: {'✅' if vertex_works else '❌'}")
    print(f"   google-generativeai with safety settings: {'✅' if genai_works else '❌'}")

    if vertex_works:
        print("\n🎉 Vertex AI approach successful! This can bypass safety filters.")
    elif genai_works:
        print("\n🎉 google-generativeai with explicit safety settings worked!")
    else:
        print("\n😞 Both approaches still hitting safety filters.")
        print("You may need to:")
        print("  1. Set up Google Cloud project for Vertex AI")
        print("  2. Use different prompt phrasing")
        print("  3. Contact Google for API access review")

if __name__ == "__main__":
    asyncio.run(main())