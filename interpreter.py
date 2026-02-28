import os
import re
import time
import queue
import threading
import sys
import collections
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
SILENCE_TIMEOUT = 0.6      # 0.6초 조용하면 문장 확정 (0.8 → 0.6으로 단축)
PREROLL_CHUNKS = 3         # 프리롤 버퍼 크기 (0.3초 = 3 chunks × 0.1초)
ASYNC_TRANSLATION_MIN_CHANGE = 10  # 비동기 번역 트리거 최소 글자 변화량
# ==================================================================

# 문장 종료 부호 감지 (숫자 속 마침표, 줄임표 제외)
SENTENCE_END_PATTERN = re.compile(r'(?<!\d)(?<!\.)([.!?])(?=\s|$)')


def split_sentences(text):
    """텍스트에서 완성된 문장과 미완성 나머지를 분리"""
    matches = list(SENTENCE_END_PATTERN.finditer(text))
    if not matches:
        return [], text  # 완성된 문장 없음

    last_match = matches[-1]
    split_pos = last_match.end()

    completed = text[:split_pos].strip()
    remaining = text[split_pos:].strip()

    return [completed], remaining

# ── Step 4: Windows ANSI VT100 활성화 ─────────────────────────────
if os.name == 'nt':
    import ctypes
    _STD_OUTPUT_HANDLE = -11
    _ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
    kernel32 = ctypes.windll.kernel32
    _handle = kernel32.GetStdHandle(_STD_OUTPUT_HANDLE)
    _mode = ctypes.c_ulong(0)
    kernel32.GetConsoleMode(_handle, ctypes.byref(_mode))
    kernel32.SetConsoleMode(_handle, _mode.value | _ENABLE_VIRTUAL_TERMINAL_PROCESSING)

def _ansi_clear_screen():
    """화면 전체를 ANSI 커서 제어로 지웁니다 (깜빡임 없음)."""
    sys.stdout.write('\033[H\033[J')
    sys.stdout.flush()

def _ansi_redraw(history, live_text, live_translation, is_partial):
    """커서를 홈으로 이동한 뒤 전체 화면을 다시 그립니다."""
    sys.stdout.write('\033[H\033[J')
    if history:
        sys.stdout.write("\n".join(history) + "\n")
        sys.stdout.write("-" * 30 + "\n")
    if live_text:
        if is_partial:
            # 회색으로 표시 (partial)
            sys.stdout.write(f"\033[90m▶ {live_text}\033[0m\n")
            sys.stdout.write(f"\033[90m▷ {live_translation}\033[0m\n")
        else:
            # 흰색 굵게 표시 (final)
            sys.stdout.write(f"\033[1m▶ {live_text}\033[0m\n")
            sys.stdout.write(f"\033[1m▷ {live_translation}\033[0m\n")
    else:
        sys.stdout.write("\n🎧 듣는 중... (대기)\n")
    sys.stdout.flush()
# ──────────────────────────────────────────────────────────────────

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

# ── Step 2: AsyncTranslator ────────────────────────────────────────
class AsyncTranslator:
    """부분 인식 중에는 비동기로, 문장 확정 시에는 동기로 번역합니다."""
    def __init__(self, translator):
        self.translator = translator
        self.last_translated = ""
        self.last_source = ""
        self.lock = threading.Lock()
        self._running = False

    def request(self, text, is_final=False):
        if is_final:
            try:
                result = self.translator.translate(text)
                with self.lock:
                    self.last_translated = result
                    self.last_source = text
            except Exception: pass
            return

        with self.lock:
            if abs(len(text) - len(self.last_source)) < ASYNC_TRANSLATION_MIN_CHANGE:
                return
            if self._running:
                return
            self._running = True

        def _do_translate():
            try:
                result = self.translator.translate(text)
                with self.lock:
                    self.last_translated = result
                    self.last_source = text
            except Exception: pass
            finally:
                with self.lock:
                    self._running = False

        threading.Thread(target=_do_translate, daemon=True).start()

    @property
    def current(self):
        with self.lock:
            return self.last_translated
# ──────────────────────────────────────────────────────────────────

# ── Step 1: stabilize_text ─────────────────────────────────────────
def stabilize_text(previous: str, current: str, max_rewrite_words: int = 8) -> str:
    """공통 접두(LCP)를 고정하고 변경 범위를 마지막 N단어로 제한합니다."""
    prev_words = previous.split()
    curr_words = current.split()

    common_len = 0
    for p, c in zip(prev_words, curr_words):
        if p == c:
            common_len += 1
        else:
            break

    locked_part = prev_words[:common_len]
    new_tail = curr_words[common_len:]

    if len(new_tail) > max_rewrite_words:
        new_tail = new_tail[-max_rewrite_words:]

    return " ".join(locked_part + new_tail)
# ──────────────────────────────────────────────────────────────────

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
    """Live/Final 2트랙 + prefix-lock + 비동기 번역 + ANSI UI"""
    accumulated_audio = []
    last_transcribe_time = time.time()

    # [핵심] 확정된 문장들을 저장하는 리스트 (최근 3개만 보여줌)
    history = []
    current_sentence = ""
    async_translator = AsyncTranslator(translator)

    while not stop_event.is_set():
        try:
            try:
                item = audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # ── Step 5: Final Track (문장 종료 신호) ──────────────────
            if item is None:
                if accumulated_audio:
                    full_audio = np.concatenate(accumulated_audio)
                    # Final: beam_size=3으로 정확한 재전사
                    segments, _ = model.transcribe(
                        full_audio,
                        beam_size=3,
                        language="en",
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=500),
                        condition_on_previous_text=False
                    )
                    final_text = " ".join([seg.text for seg in segments]).strip()

                    if final_text:
                        # 동기 번역 (final)
                        async_translator.request(final_text, is_final=True)
                        final_translation = async_translator.current

                        history.append(f"🇺🇸 {final_text}\n🇰🇷 {final_translation}")
                        if len(history) > 3:
                            history.pop(0)

                current_sentence = ""
                accumulated_audio = []
                _ansi_redraw(history, "", "", False)
                continue
            # ─────────────────────────────────────────────────────────

            accumulated_audio.append(item)

            # ── Step 5: Live Track (0.3초마다) ────────────────────────
            if time.time() - last_transcribe_time > UPDATE_INTERVAL:
                full_audio = np.concatenate(accumulated_audio)

                # Live: beam_size=1, vad_filter=True — 빠른 부분 자막
                segments, _ = model.transcribe(
                    full_audio,
                    beam_size=1,
                    language="en",
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500),
                    condition_on_previous_text=False
                )

                raw_text = " ".join([seg.text for seg in segments]).strip()

                if len(raw_text) > 1:
                    # Step 1: prefix-lock 안정화
                    stable_text = stabilize_text(current_sentence, raw_text)

                    # 문장 종료 부호 감지 → 즉시 확정
                    completed_sentences, remaining = split_sentences(stable_text)

                    if completed_sentences:
                        for sentence in completed_sentences:
                            async_translator.request(sentence, is_final=True)
                            translation = async_translator.current
                            history.append(f"🇺🇸 {sentence}\n🇰🇷 {translation}")
                            if len(history) > 3:
                                history.pop(0)

                        current_sentence = remaining

                        # 오디오 버퍼 정리: 나머지 텍스트 비율만큼만 유지
                        if remaining and stable_text:
                            ratio = len(remaining) / len(stable_text)
                            total_samples = sum(len(a) for a in accumulated_audio)
                            keep_samples = int(total_samples * ratio)
                            full_audio = np.concatenate(accumulated_audio)
                            accumulated_audio = [full_audio[-keep_samples:]] if keep_samples > 0 else []
                        else:
                            accumulated_audio = []

                        _ansi_redraw(history, remaining, async_translator.current, is_partial=bool(remaining))
                    else:
                        current_sentence = stable_text

                        # Step 2: 비동기 번역 요청 (partial)
                        async_translator.request(stable_text, is_final=False)

                        _ansi_redraw(history, stable_text, async_translator.current, is_partial=True)

                last_transcribe_time = time.time()
            # ─────────────────────────────────────────────────────────

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

        _ansi_clear_screen()
        print(f"✅ 연결됨: {target['name']}")
        print("🚀 [라이브 캡션 V3] 라이브캡션 수준 업그레이드")
        print("   Prefix-lock | 비동기 번역 | VAD 프리롤 | ANSI UI")
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
        # ── Step 3: 프리롤 버퍼 (최근 0.3초 = 3 chunks) ──────────────
        preroll_buf = collections.deque(maxlen=PREROLL_CHUNKS)
        # ─────────────────────────────────────────────────────────────

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
                    if not is_speaking:
                        # ── Step 3: 발화 시작 시 프리롤 데이터를 먼저 전송 ──
                        for pre_chunk in preroll_buf:
                            audio_queue.put(pre_chunk)
                        # ────────────────────────────────────────────────────
                    is_speaking = True
                    silence_start = None
                    audio_queue.put(resampled_chunk)
                else:
                    # 침묵 구간은 프리롤 버퍼에 보관
                    preroll_buf.append(resampled_chunk)
                    if is_speaking:
                        if silence_start is None:
                            silence_start = time.time()

                        audio_queue.put(resampled_chunk)

                        if time.time() - silence_start > SILENCE_TIMEOUT:
                            is_speaking = False
                            silence_start = None
                            audio_queue.put(None)  # 문장 확정 신호

            except IOError: continue
            except KeyboardInterrupt: break

    except KeyboardInterrupt: print("\n종료")
    finally:
        stop_event.set()
        if 'stream' in locals(): stream.stop_stream(); stream.close()
        p.terminate()

if __name__ == "__main__":
    main()