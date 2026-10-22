import asyncio
import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from generate_test_audio import create_synthetic_wav
from riddle_diary_agent import RiddleDiaryAgent, RiddleDiaryState, ATMOSPHERIC_FALLBACK

async def run_direct_agent_test():
    print("\n" + "=" * 60)
    print("STEP 1: Testing Direct RiddleDiaryAgent Capabilities")
    print("=" * 60)
    
    test_audio = "test_audio.wav"
    if not os.path.exists(test_audio):
        create_synthetic_wav(test_audio)

    agent = RiddleDiaryAgent()

    # Test 1: Run with valid audio
    print("\n--- Sub-test 1.1: Audio Transcription & Pipeline Execution ---")
    state = RiddleDiaryState(
        session_id="test_session_001",
        raw_audio_path=test_audio
    )
    result = await agent.run(state)
    print(f"Result Status: {result.get('status')}")
    print(f"Transcript: '{result.get('user_transcript')}'")
    print(f"Tom Riddle Response: '{result.get('riddle_response')}'")
    print(f"Manifest Lines: {result.get('texture_manifest', {}).get('lines')}")

    # Test 2: In-character response with text thought
    print("\n--- Sub-test 1.2: Direct Whisper / Thought Inscription ---")
    state_text = RiddleDiaryState(
        session_id="test_session_001",
        user_transcript="Tom, I found your diary hidden in the girls bathroom. Who are you?",
        conversation_history=result.get("conversation_history", [])
    )
    state_text = await agent.generate_riddle_reply_node(state_text)
    result_text = await agent.clean_and_dispatch_node(state_text)
    print(f"Tom Riddle Response: '{result_text.get('riddle_response')}'")
    print(f"Updated History Length: {len(result_text.get('conversation_history', []))}")

    # Test 3: Robust Exception / Fallback Verification
    print("\n--- Sub-test 1.3: Non-existent Audio Exception Handling ---")
    faulty_state = RiddleDiaryState(
        session_id="test_session_err",
        raw_audio_path="non_existent_audio_path.wav"
    )
    faulty_result = await agent.run(faulty_state)
    print(f"Expected Fallback Response: '{faulty_result.get('riddle_response')}'")
    assert faulty_result.get("riddle_response") == ATMOSPHERIC_FALLBACK, "Fallback mismatch!"

    print("\n" + "=" * 60)
    print("DIRECT AGENT CAPABILITIES TEST PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(run_direct_agent_test())
