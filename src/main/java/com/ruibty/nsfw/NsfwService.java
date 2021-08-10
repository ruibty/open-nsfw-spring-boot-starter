package com.ruibty.nsfw;

public interface NsfwService {
    float getPrediction(byte[] imgBytes);
}
