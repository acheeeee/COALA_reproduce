"""檢查兩個預訓練模型的規格，驗證論文 3.4 節的參數量說法。"""

from transformers import WhisperModel, AutoModelForCausalLM

def count_params(module):
      """回傳一個 module 裡所有參數的總數量。"""
      return sum(p.numel() for p in module.parameters())

def human(n):
      """把 640123456 這種數字轉成 640.1M 這種易讀格式。"""
      return f"{n / 1e6:.1f}M"

def main():
      print("=" * 62)
      print("載入 Whisper-large-v2（第一次會下載約 6GB，請等）...")
      whisper = WhisperModel.from_pretrained("openai/whisper-large-v2")
      encoder = whisper.get_encoder()

      n_enc = count_params(encoder)
      n_whisper_all = count_params(whisper)

      print(f"  encoder 參數量       : {human(n_enc):>10}   (論文說 640M)")
      print(f"  整個 WhisperModel    : {human(n_whisper_all):>10}   <- 含用不到的 decoder")
      print(f"  encoder d_model      : {whisper.config.d_model}")
      print(f"  encoder 層數         : {whisper.config.encoder_layers}")

      print("=" * 62)
      print("載入 SmolLM2-135M-Instruct ...")
      lm = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")

      n_lm = count_params(lm)

      print(f"  參數量               : {human(n_lm):>10}   (論文說 135M)")
      print(f"  hidden_size (= Eq.2 的 D) : {lm.config.hidden_size}")
      print(f"  vocab_size           : {lm.config.vocab_size}")
      print(f"  層數                 : {lm.config.num_hidden_layers}")

      print("=" * 62)
      total = n_enc + n_lm
      print(f"  encoder + LM         : {human(total):>10}")
      print(f"  論文總參數量         : {'777.0M':>10}")
      print(f"  差額                 : {human(777e6 - total):>10}   <- 我要自己訓練的部分")
      print("=" * 62)

if __name__ == "__main__":
      main()

