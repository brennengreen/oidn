// Copyright 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cpu_temporal_accumulation.h"
#include "cpu_common.h"
#include <cmath>

OIDN_NAMESPACE_BEGIN

  CPUTemporalAccumulation::CPUTemporalAccumulation(CPUEngine* engine)
    : engine(engine)
  {}

  namespace
  {
    oidn_inline int clampInt(int x, int lo, int hi)
    {
      return x < lo ? lo : (x > hi ? hi : x);
    }

    // Catmull-Rom (cubic) interpolation weights for the 4 taps at offsets
    // -1, 0, 1, 2 around the sample. Catmull-Rom has mild negative lobes, so it
    // is sharpening: unlike bilinear it does not blur the history under
    // sub-pixel motion, which is the standard fix for TAA accumulation blur.
    oidn_inline void catmullRomWeights(float t, float w[4])
    {
      const float t2 = t * t, t3 = t2 * t;
      w[0] = -0.5f*t3 +      t2 - 0.5f*t;
      w[1] =  1.5f*t3 - 2.5f*t2          + 1.f;
      w[2] = -1.5f*t3 + 2.0f*t2 + 0.5f*t;
      w[3] =  0.5f*t3 - 0.5f*t2;
    }

    // Sharp 4x4 Catmull-Rom sample of an image at floating-point pixel coords.
    oidn_inline vec3f sampleCatmullRom(const ImageAccessor& img, float x, float y)
    {
      x = math::clamp(x, 0.f, float(img.W - 1));
      y = math::clamp(y, 0.f, float(img.H - 1));

      const int ix = int(std::floor(x));
      const int iy = int(std::floor(y));
      float wx[4], wy[4];
      catmullRomWeights(x - float(ix), wx);
      catmullRomWeights(y - float(iy), wy);

      vec3f acc(0.f);
      for (int j = 0; j < 4; ++j)
      {
        const int sy = clampInt(iy - 1 + j, 0, img.H - 1);
        vec3f row(0.f);
        for (int i = 0; i < 4; ++i)
        {
          const int sx = clampInt(ix - 1 + i, 0, img.W - 1);
          row = row + img.get3<float>(sy, sx) * wx[i];
        }
        acc = acc + row * wy[j];
      }
      return acc;
    }
  }

  void CPUTemporalAccumulation::submitKernels(const Ref<CancellationToken>& ct)
  {
    check();

    const ImageAccessor color   = *this->color;
    const ImageAccessor history = *this->history;
    const ImageAccessor dst     = *this->dst;
    const ImageAccessor output  = *this->output;
    const bool hasFlow = bool(this->flow);
    const ImageAccessor flow = hasFlow ? ImageAccessor(*this->flow) : ImageAccessor{};

    const int H = dst.H;
    const int W = dst.W;
    const float alpha = this->alpha;
    const float clampStrength = this->clampStrength;
    const float sharpness = this->sharpness;
    const bool reset = this->reset;

    engine->submitFunc([=]
    {
      // Pass 1: motion-compensated temporal accumulation -> dst (next history)
      parallel_for(H, [&](int h)
      {
        for (int w = 0; w < W; ++w)
        {
          vec3f cur = math::nan_to_zero(color.get3<float>(h, w));

          if (reset)
          {
            dst.set3(h, w, cur);
            continue;
          }

          // Reproject the previous frame using the backward motion vectors
          float px = float(w);
          float py = float(h);
          if (hasFlow)
          {
            const vec3f mv = flow.get3<float>(h, w);
            px += mv.x;
            py += mv.y;
          }

          // Reject history that reprojects outside the image (disocclusion)
          const bool valid = (px >= 0.f && px <= float(W - 1) &&
                              py >= 0.f && py <= float(H - 1));
          if (!valid)
          {
            dst.set3(h, w, cur);
            continue;
          }

          // Sharp (Catmull-Rom) reprojection avoids bilinear accumulation blur
          vec3f hist = math::nan_to_zero(sampleCatmullRom(history, px, py));

          // History rejection via neighborhood variance clipping (SVGF/TAA):
          // clamp the reprojected history to the local color statistics of the
          // current frame to suppress ghosting on moving/changing content.
          if (clampStrength > 0.f)
          {
            vec3f m1(0.f);
            vec3f m2(0.f);
            int n = 0;
            for (int dy = -1; dy <= 1; ++dy)
            {
              const int hh = clampInt(h + dy, 0, H - 1);
              for (int dx = -1; dx <= 1; ++dx)
              {
                const int ww = clampInt(w + dx, 0, W - 1);
                const vec3f c = math::nan_to_zero(color.get3<float>(hh, ww));
                m1 = m1 + c;
                m2 = m2 + c * c;
                ++n;
              }
            }
            const float invN = 1.f / float(n);
            const vec3f mean = m1 * invN;
            const vec3f var  = math::max(m2 * invN - mean * mean, vec3f(0.f));
            const vec3f sigma(std::sqrt(var.x), std::sqrt(var.y), std::sqrt(var.z));
            const vec3f lo = mean - sigma * clampStrength;
            const vec3f hi = mean + sigma * clampStrength;
            hist = math::min(math::max(hist, lo), hi);
          }

          // Catmull-Rom can overshoot slightly; keep colors non-negative
          hist = math::max(hist, vec3f(0.f));

          // Exponential blend of the (clamped, reprojected) history and current
          const vec3f accum = hist * (1.f - alpha) + cur * alpha;
          dst.set3(h, w, accum);
        }
      });

      // Pass 2: resolve to output, with optional contrast-limited sharpening.
      // Reads the fully-written accumulation buffer (dst) and never writes back
      // to the history, so the sharpening cannot compound or destabilize.
      parallel_for(H, [&](int h)
      {
        for (int w = 0; w < W; ++w)
        {
          const vec3f c = dst.get3<float>(h, w);

          if (sharpness <= 0.f)
          {
            output.set3(h, w, c);
            continue;
          }

          // 3x3 neighborhood: blurred mean (for unsharp) and min/max (limiter)
          vec3f sum(0.f), lo = c, hi = c;
          for (int dy = -1; dy <= 1; ++dy)
          {
            const int hh = clampInt(h + dy, 0, H - 1);
            for (int dx = -1; dx <= 1; ++dx)
            {
              const int ww = clampInt(w + dx, 0, W - 1);
              const vec3f s = dst.get3<float>(hh, ww);
              sum = sum + s;
              lo = math::min(lo, s);
              hi = math::max(hi, s);
            }
          }
          const vec3f mean = sum * (1.f / 9.f);

          // Unsharp mask, limited to an expanded local range to allow real
          // sharpening while still suppressing edge halos/ringing
          const vec3f ext = (hi - lo) * 0.5f;
          const vec3f loE = lo - ext;
          const vec3f hiE = hi + ext;
          vec3f o = c + (c - mean) * sharpness;
          o = math::min(math::max(o, loE), hiE);
          o = math::max(o, vec3f(0.f));
          output.set3(h, w, o);
        }
      });
    }, ct);
  }

OIDN_NAMESPACE_END
