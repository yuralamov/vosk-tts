#!/usr/bin/env python3
import os
import sys
from timeit import default_timer as timer

import torchaudio
import torch

from dots_tts.runtime import DotsTtsRuntime

runtime = DotsTtsRuntime.from_pretrained(
    "dottts-test/dots.tts-soar",
    precision="bfloat16",
)

spkmap = {}
for i, line in enumerate(open("eval-speakers-text/metadata-phones-ids.csv.test-ref")):
    items = line.strip().split("|")
    spk = items[0].split("/")[0]
    spkmap[spk] = (items[0].replace("/", "_"), items[1])

def main_synth(out_dir=None):

    os.makedirs(out_dir, exist_ok=True)

    start = timer()
    total_len = 0
    for i, line in enumerate(open("eval.csv", encoding='utf-8')):
        items = line.strip().split("|")
        fitems = items[0].split("/")
        uid = fitems[2] + "_" + fitems[-1][:-4]
        spk = items[0].split("/")[2]
        text = items[1].replace("+", "")

        reffn, reftext = spkmap[spk]
        reftext = reftext.replace("+", "")
        reffn = "eval-speakers-text/wav/" + reffn

        result = runtime.generate(
            text=text,
            prompt_audio_path=reffn,
            prompt_text=reftext,
            num_steps=10,
            guidance_scale=1.2,
        )
        out_wav = result["audio"].squeeze().cpu()
        total_len += len(out_wav)

        torchaudio.save(out_dir + "/" + uid + ".wav", out_wav, result["sample_rate"], backend='soundfile')

    end = timer()

    audio_duration_sec = float(total_len) / 48000
    infer_sec = end - start
    real_time_factor = (infer_sec / audio_duration_sec if audio_duration_sec > 0 else 0.0)
    print(f"Real-time factor: {real_time_factor:.4f} (infer={infer_sec:.2f} sec, audio={audio_duration_sec:.2f} sec)")
    
main_synth(out_dir = 'out')
