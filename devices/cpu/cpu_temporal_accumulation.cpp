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

    // Bilinear sample of an image at floating-point pixel coordinates
    oidn_inline vec3f sampleBilinear(const ImageAccessor& img, float x, float y)
    {
      const float xc = math::clamp(x, 0.f, float(img.W - 1));
      const float yc = math::clamp(y, 0.f, float(img.H - 1));

      const int x0 = int(std::floor(xc));
      const int y0 = int(std::floor(yc));
      const int x1 = clampInt(x0 + 1, 0, img.W - 1);
      const int y1 = clampInt(y0 + 1, 0, img.H - 1);

      const float fx = xc - float(x0);
      const float fy = yc - float(y0);

      const vec3f c00 = img.get3<float>(y0, x0);
      const vec3f c10 = img.get3<float>(y0, x1);
      const vec3f c01 = img.get3<float>(y1, x0);
      const vec3f c11 = img.get3<float>(y1, x1);

      const vec3f c0 = c00 * (1.f - fx) + c10 * fx;
      const vec3f c1 = c01 * (1.f - fx) + c11 * fx;
      return c0 * (1.f - fy) + c1 * fy;
    }
  }

  void CPUTemporalAccumulation::submitKernels(const Ref<CancellationToken>& ct)
  {
    check();

    const ImageAccessor color   = *this->color;
    const ImageAccessor history = *this->history;
    const ImageAccessor dst     = *this->dst;
    const bool hasFlow = bool(this->flow);
    const ImageAccessor flow = hasFlow ? ImageAccessor(*this->flow) : ImageAccessor{};

    const int H = dst.H;
    const int W = dst.W;
    const float alpha = this->alpha;
    const float clampStrength = this->clampStrength;
    const bool reset = this->reset;

    engine->submitFunc([=]
    {
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

          vec3f hist = math::nan_to_zero(sampleBilinear(history, px, py));

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

          // Exponential blend of the (clamped, reprojected) history and current
          const vec3f accum = hist * (1.f - alpha) + cur * alpha;
          dst.set3(h, w, accum);
        }
      });
    }, ct);
  }

OIDN_NAMESPACE_END
