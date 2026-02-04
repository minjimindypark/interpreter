import os
import time
import queue
import threading
import sys
import numpy as np
import pyaudiowpatch as pa
from scipy import signal
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator

# ==================================================================
# 🔧 시스템 설정
# ==================================================================
MODEL_SIZE = "tiny.en"     # 속도 최적화
TARGET_SR = 16000
VAD_THRESHOLD = 0.015      # 잡음 무시 임계값
UPDATE_INTERVAL = 0.3      # 0.3초마다 화면 갱신 (반응속도 UP)
SILENCE_TIMEOUT = 0.8      # 0.8초 조용하면 문장 확정
# ==================================================================

# 윈도우/맥에 따른 화면 지우기 명령어 설정
CLEAR_CMD = 'cls' if os.name == 'nt' else 'clear'

print(f"🔄 시스템 재구축 중... (Model: {MODEL_SIZE})")

try:
    # CPU 코어 4개 사용
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8", cpu_threads=4)
    translator = GoogleTranslator(source='auto', target='ko')
except Exception as e:
    print(f"❌ 초기화 실패: {e}")
    os._exit(1)

audio_queue = queue.Queue()
stop_event = threading.Event()

def get_loopback_device(p):
    try:
        wasapi = p.get_host_api_info_by_type(pa.paWASAPI)
        default_out = p.get_device_info_by_index(wasapi["defaultOutputDevice"])
        for i in range(p.get_device_count()):
            dev = p.get_device_info_by_index(i)
            if dev["hostApi"] == wasapi["index"] and dev["maxInputChannels"] > 0:
                if dev["name"] == default_out["name"] or "Loopback" in dev["name"]:
                    return dev
    except: pass
    return None

def process_worker():
    """화면을 지우고 다시 그리는 방식 (중복 원천 차단)"""
    accumulated_audio = []
    last_transcribe_time = time.time()
    
    # [핵심] 확정된 문장들을 저장하는 리스트 (최근 3개만 보여줌)
    history = []
    current_sentence = ""
    current_translation = ""

    while not stop_event.is_set():
        try:
            try:
                item = audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # 문장 종료 신호
            if item is None:
                if current_sentence:
                    # 현재 문장을 역사(History)에 기록하고 버퍼 비움
                    history.append(f"🇺🇸 {current_sentence}\n🇰🇷 {current_translation}")
                    if len(history) > 3: # 화면 꽉 차니까 최근 3개만 유지
                        history.pop(0)
                    
                    current_sentence = ""
                    current_translation = ""
                    accumulated_audio = []
                    
                    # 화면 갱신
                    os.system(CLEAR_CMD)
                    print("\n".join(history))
                    print(f"\n🎧 듣는 중... (대기)")
                continue

            accumulated_audio.append(item)

            # 실시간 분석 (0.3초마다)
            if time.time() - last_transcribe_time > UPDATE_INTERVAL:
                full_audio = np.concatenate(accumulated_audio)
                
                # Whisper 인식
                segments, _ = model.transcribe(
                    full_audio,
                    beam_size=1,
                    language="en",
                    condition_on_previous_text=False
                )
                
                text = " ".join([seg.text for seg in segments]).strip()
                
                # 내용이 있고, 이전과 다를 때만 갱신
                if len(text) > 1 and text != current_sentence:
                    try:
                        kor = translator.translate(text)
                        
                        current_sentence = text
                        current_translation = kor
                        
                        # [핵심 기술] 화면 전체를 지우고(CLS) 다시 씀
                        # 이렇게 하면 글자가 겹치거나 반복될 일이 0%
                        os.system(CLEAR_CMD)
                        
                        # 1. 지나간 대화 보여주기
                        if history:
                            print("\n".join(history))
                            print("-" * 30)
                        
                        # 2. 현재 말하고 있는 문장 (실시간 업데이트)
                        print(f"▶ {current_sentence}")
                        print(f"▷ {current_translation}")
                        
                    except: pass
                
                last_transcribe_time = time.time()

        except Exception:
            pass

def main():
    p = pa.PyAudio()
    try:
        target = get_loopback_device(p)
        if not target:
            print("❌ 장치 없음")
            return

        native_rate = int(target["defaultSampleRate"])
        input_channels = target["maxInputChannels"]
        
        os.system(CLEAR_CMD)
        print(f"✅ 연결됨: {target['name']}")
        print("🚀 [라이브 캡션 V2] 화면 리프레시 모드")
        print("   (중복된 글자가 절대 쌓이지 않습니다)")
        time.sleep(2)

        stream = p.open(format=pa.paFloat32,
                        channels=input_channels,
                        rate=native_rate,
                        input=True,
                        input_device_index=target["index"])

        t = threading.Thread(target=process_worker)
        t.daemon = True
        t.start()

        is_speaking = False
        silence_start = None
        
        while True:
            try:
                chunk_len = int(native_rate * 0.1)
                raw_data = stream.read(chunk_len, exception_on_overflow=False)
                audio_float = np.frombuffer(raw_data, dtype=np.float32)
                
                if input_channels > 1:
                    audio_mono = audio_float.reshape(-1, input_channels).mean(axis=1)
                else:
                    audio_mono = audio_float
                
                num_samples = int(len(audio_mono) * TARGET_SR / native_rate)
                resampled_chunk = signal.resample(audio_mono, num_samples)
                
                rms = np.sqrt(np.mean(resampled_chunk**2))
                
                # VAD 로직
                if rms > VAD_THRESHOLD:
                    is_speaking = True
                    silence_start = None
                    audio_queue.put(resampled_chunk)
                else:
                    if is_speaking:
                        if silence_start is None:
                            silence_start = time.time()
                        
                        audio_queue.put(resampled_chunk)
                        
                        if time.time() - silence_start > SILENCE_TIMEOUT:
                            is_speaking = False
                            silence_start = None
                            audio_queue.put(None) # 문장 확정 신호
                    else:
                        pass

            except IOError: continue
            except KeyboardInterrupt: break

    except KeyboardInterrupt: print("\n종료")
    finally:
        stop_event.set()
        if 'stream' in locals(): stream.stop_stream(); stream.close()
        p.terminate()

if __name__ == "__main__":
    main()