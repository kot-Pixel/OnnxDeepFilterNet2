#ifndef DF_FEATURES_H
#define DF_FEATURES_H

/**
 * 与 libDF (libDF/src/lib.rs) 对齐的特征：feat_erb、feat_spec（band_unit_norm_t）。
 * 训练侧见 DeepFilterNet/df/enhance.py 中 df_features()。
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "kissfft/kiss_fft.h"

namespace df {

inline float freq2erb(float freq_hz) {
    return 9.265f * std::log1p(freq_hz / (24.7f * 9.265f));
}

inline float erb2freq(float n_erb) {
    return 24.7f * 9.265f * (std::exp(n_erb / 9.265f) - 1.f);
}

/** 与 DFState::erb_fb 一致：各 ERB 带包含的 FFT bin 个数，总和 = fft_size/2+1 */
inline std::vector<int> erbFb(int sr, int fft_size, int nb_bands, int min_nb_freqs) {
    const int nyq_freq = sr / 2;
    const float freq_width = static_cast<float>(sr) / static_cast<float>(fft_size);
    const float erb_low = freq2erb(0.f);
    const float erb_high = freq2erb(static_cast<float>(nyq_freq));
    std::vector<int> erb(static_cast<size_t>(nb_bands), 0);
    const float step = (erb_high - erb_low) / static_cast<float>(nb_bands);
    int prev_freq = 0;
    int freq_over = 0;
    for (int i = 1; i <= nb_bands; ++i) {
        const float f = erb2freq(erb_low + static_cast<float>(i) * step);
        int fb = static_cast<int>(std::round(f / freq_width));
        int nb_freqs = fb - prev_freq - freq_over;
        if (nb_freqs < min_nb_freqs) {
            freq_over = min_nb_freqs - nb_freqs;
            nb_freqs = min_nb_freqs;
        } else {
            freq_over = 0;
        }
        erb[static_cast<size_t>(i - 1)] = nb_freqs;
        prev_freq = fb;
    }
    erb[static_cast<size_t>(nb_bands - 1)] += 1;
    const int sum_want = fft_size / 2 + 1;
    int s = 0;
    for (int w : erb) s += w;
    const int too_large = s - sum_want;
    if (too_large > 0)
        erb[static_cast<size_t>(nb_bands - 1)] -= too_large;
    return erb;
}

/** 由 erb 宽度得到每个带在 STFT 上的 [start, end) bin 区间 */
inline std::vector<std::pair<int, int>> erbBinRanges(const std::vector<int> &erb_widths) {
    std::vector<std::pair<int, int>> out;
    out.reserve(erb_widths.size());
    int start = 0;
    for (int w : erb_widths) {
        out.emplace_back(start, start + w);
        start += w;
    }
    return out;
}

/** norm_tau（config [df]）与 hop/sr 得到 alpha，同 get_norm_alpha / _calculate_norm_alpha */
inline float normAlphaFromTau(int sr, int hop_size, float norm_tau) {
    const float dt = static_cast<float>(hop_size) / static_cast<float>(sr);
    return std::exp(-dt / norm_tau);
}

/** MEAN_NORM_INIT: 线性 -60 … -90（nb_erb 个带） */
inline void initMeanNormState(std::vector<float> &state, int nb_erb) {
    constexpr float kMin = -60.f;
    constexpr float kMax = -90.f;
    state.resize(static_cast<size_t>(nb_erb));
    if (nb_erb <= 1) {
        if (nb_erb == 1)
            state[0] = kMin;
        return;
    }
    const float step = (kMax - kMin) / static_cast<float>(nb_erb - 1);
    for (int i = 0; i < nb_erb; ++i)
        state[static_cast<size_t>(i)] = kMin + static_cast<float>(i) * step;
}

/** UNIT_NORM_INIT: 线性 0.001 … 0.0001（nb_df 个频点） */
inline void initUnitNormState(std::vector<float> &state, int nb_df) {
    constexpr float kMin = 0.001f;
    constexpr float kMax = 0.0001f;
    state.resize(static_cast<size_t>(nb_df));
    if (nb_df <= 1) {
        if (nb_df == 1)
            state[0] = kMin;
        return;
    }
    const float step = (kMax - kMin) / static_cast<float>(nb_df - 1);
    for (int i = 0; i < nb_df; ++i)
        state[static_cast<size_t>(i)] = kMin + static_cast<float>(i) * step;
}

/**
 * libDF feat_erb：带内平均功率 → 10*log10 → 指数均值归一化（band_mean_norm_erb）。
 * spec 须已含 wnorm（与 libDF analysis 输出一致）。
 */
inline void featErb(const std::vector<int> &erb_widths,
                    const kiss_fft_cpx *spec,
                    int /*n_bins*/,
                    float alpha,
                    std::vector<float> &mean_state,
                    float *out_erb,
                    int nb_erb) {
    int bsum = 0;
    for (int b = 0; b < nb_erb; ++b) {
        const int w = erb_widths[static_cast<size_t>(b)];
        float acc = 0.f;
        const float kinv = 1.f / static_cast<float>(w);
        for (int j = 0; j < w; ++j) {
            const int idx = bsum + j;
            const float re = spec[idx].r;
            const float im = spec[idx].i;
            acc += (re * re + im * im) * kinv;
        }
        bsum += w;
        float x = 10.f * std::log10(acc + 1e-10f);
        float &s = mean_state[static_cast<size_t>(b)];
        s = x * (1.f - alpha) + s * alpha;
        out_erb[b] = (x - s) / 40.f;
    }
}

/**
 * band_unit_norm_t：输出 layout 与 ONNX [1,2,1,nb_df] 展平一致 —— 先 nb_df 个 re，再 nb_df 个 im。
 */
inline void featSpecUnitNorm(const kiss_fft_cpx *spec,
                             int nb_df,
                             float alpha,
                             std::vector<float> &unit_state,
                             float *out_spec) {
    for (int k = 0; k < nb_df; ++k) {
        const float re = spec[k].r;
        const float im = spec[k].i;
        const float mag = std::sqrt(re * re + im * im + 1e-20f);
        float &s = unit_state[static_cast<size_t>(k)];
        s = mag * (1.f - alpha) + s * alpha;
        const float inv = 1.f / std::sqrt(s + 1e-20f);
        out_spec[k] = re * inv;
        out_spec[nb_df + k] = im * inv;
    }
}

/** libDF 分析窗：vorbis 式 sin( pi/2 * sin^2( pi*(i+0.5)/(N/2) ) ) */
inline void fillVorbisWindow(int fft_size, std::vector<float> &window) {
    window.resize(static_cast<size_t>(fft_size));
    const int window_size_h = fft_size / 2;
    const double pi = 3.14159265358979323846;
    for (int i = 0; i < fft_size; ++i) {
        const double s =
            std::sin(0.5 * pi * (static_cast<double>(i) + 0.5) / static_cast<double>(window_size_h));
        window[static_cast<size_t>(i)] = static_cast<float>(std::sin(0.5 * pi * s * s));
    }
}

/** wnorm = 2*hop / fft_size^2，与 DFState::new 一致 */
inline float analysisWnorm(int fft_size, int hop_size) {
    return 1.f / (static_cast<float>(fft_size * fft_size) /
                  (2.f * static_cast<float>(hop_size)));
}

} // namespace df

#endif
