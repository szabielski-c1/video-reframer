#!/usr/bin/env python3
"""
Test script to verify new Gemini API key functionality and check for safety filter issues
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

from app.services.gemini_service import GeminiService
from app.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_basic_connectivity():
    """Test basic API connectivity with simple text prompt"""
    print("\n🔍 Testing basic Gemini API connectivity...")

    service = GeminiService()

    try:
        health_result = await service.health_check()
        print(f"✓ Health check result: {json.dumps(health_result, indent=2)}")

        if health_result['status'] == 'healthy':
            print("✅ Basic connectivity test PASSED")
            return True
        else:
            print(f"⚠️ Basic connectivity test shows issues: {health_result['status']}")
            return False

    except Exception as e:
        print(f"❌ Basic connectivity test FAILED: {str(e)}")
        return False

async def test_video_content_analysis():
    """Test video analysis prompts that might trigger safety filters"""
    print("\n🎥 Testing video analysis prompts for safety filter issues...")

    service = GeminiService()

    # Test prompts that previously caused issues
    test_prompts = [
        {
            "name": "Basic video analysis",
            "prompt": "Analyze this video content and describe what you see. Focus on visual elements, people, and activities."
        },
        {
            "name": "Shot detection",
            "prompt": "Detect shot boundaries and scene changes in this video. Identify cuts, transitions, and camera movements."
        },
        {
            "name": "Subject tracking",
            "prompt": "Identify and track the main subjects in this video. Focus on people, faces, and moving objects."
        },
        {
            "name": "Reframing analysis",
            "prompt": "Analyze this video for reframing from 16:9 to 9:16 format. Identify the best crop positions to keep important content visible."
        }
    ]

    results = []

    for test in test_prompts:
        print(f"\n  Testing: {test['name']}")
        try:
            # Test with text-only model first (no video upload needed)
            response = await asyncio.to_thread(
                service.model.generate_content,
                f"{test['prompt']}\n\nNote: This is a test prompt. Please respond with: 'Test successful - no safety issues detected.'"
            )

            if response and response.text:
                print(f"    ✓ Response received: {response.text[:100]}...")

                # Check for safety blocks
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'finish_reason'):
                        finish_reason = candidate.finish_reason
                        if finish_reason == 2:  # SAFETY
                            print(f"    ⚠️ SAFETY FILTER TRIGGERED")
                            safety_ratings = getattr(candidate, 'safety_ratings', [])
                            print(f"    Safety ratings: {safety_ratings}")
                            results.append({"test": test['name'], "status": "blocked", "reason": "safety_filter"})
                        elif finish_reason == 3:  # RECITATION
                            print(f"    ⚠️ RECITATION FILTER TRIGGERED")
                            results.append({"test": test['name'], "status": "blocked", "reason": "recitation"})
                        else:
                            print(f"    ✅ No safety issues (finish_reason: {finish_reason})")
                            results.append({"test": test['name'], "status": "passed"})
                    else:
                        print(f"    ✅ No safety issues (no finish_reason)")
                        results.append({"test": test['name'], "status": "passed"})
                else:
                    print(f"    ✅ Response successful")
                    results.append({"test": test['name'], "status": "passed"})
            else:
                print(f"    ❌ No response received")
                results.append({"test": test['name'], "status": "failed", "reason": "no_response"})

        except Exception as e:
            error_str = str(e)
            print(f"    ❌ Error: {error_str}")

            if "safety" in error_str.lower() or "finish_reason" in error_str.lower():
                results.append({"test": test['name'], "status": "blocked", "reason": "safety_exception", "error": error_str})
            else:
                results.append({"test": test['name'], "status": "failed", "reason": "exception", "error": error_str})

    return results

async def test_api_key_validity():
    """Test if the new API key is valid and working"""
    print("\n🔑 Testing API key validity...")

    try:
        import google.generativeai as genai
        genai.configure(api_key=settings.GEMINI_API_KEY)

        # List available models to test authentication
        models = list(genai.list_models())
        print(f"✓ API key authenticated successfully")
        print(f"✓ Available models: {len(models)}")

        # Check if our configured model is available
        model_names = [m.name for m in models]
        configured_model = settings.GEMINI_MODEL

        if any(configured_model in name for name in model_names):
            print(f"✓ Configured model '{configured_model}' is available")
            return True
        else:
            print(f"⚠️ Configured model '{configured_model}' not found")
            print(f"Available models: {[name.split('/')[-1] for name in model_names[:5]]}")
            return False

    except Exception as e:
        print(f"❌ API key validation failed: {str(e)}")
        return False

async def main():
    """Run all tests"""
    print("🧪 Starting Gemini API tests with new API key")
    print(f"📋 Configuration:")
    print(f"   API Key: {settings.GEMINI_API_KEY[:8]}...{settings.GEMINI_API_KEY[-8:]}")
    print(f"   Model: {settings.GEMINI_MODEL}")

    # Test 1: API key validity
    key_valid = await test_api_key_validity()

    # Test 2: Basic connectivity
    basic_works = await test_basic_connectivity() if key_valid else False

    # Test 3: Video analysis prompts for safety issues
    safety_results = await test_video_content_analysis() if basic_works else []

    # Summary
    print("\n📊 TEST SUMMARY:")
    print(f"   API Key Valid: {'✅' if key_valid else '❌'}")
    print(f"   Basic Connectivity: {'✅' if basic_works else '❌'}")

    if safety_results:
        passed = sum(1 for r in safety_results if r['status'] == 'passed')
        blocked = sum(1 for r in safety_results if r['status'] == 'blocked')
        failed = sum(1 for r in safety_results if r['status'] == 'failed')

        print(f"   Safety Filter Tests:")
        print(f"     ✅ Passed: {passed}/{len(safety_results)}")
        print(f"     ⚠️ Blocked: {blocked}/{len(safety_results)}")
        print(f"     ❌ Failed: {failed}/{len(safety_results)}")

        if blocked > 0:
            print(f"\n⚠️ SAFETY FILTER ISSUES DETECTED:")
            for result in safety_results:
                if result['status'] == 'blocked':
                    print(f"   - {result['test']}: {result.get('reason', 'unknown')}")
        else:
            print(f"\n✅ NO SAFETY FILTER ISSUES DETECTED")

    print(f"\n🏁 Tests completed!")

if __name__ == "__main__":
    asyncio.run(main())