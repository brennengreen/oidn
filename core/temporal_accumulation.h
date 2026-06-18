// Copyright 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "op.h"
#include "image.h"

OIDN_NAMESPACE_BEGIN

  // Motion-compensated temporal accumulation operation.
  //
  // Improves the temporal stability (reduces flickering) of a sequence of
  // independently denoised frames by blending the current denoised frame with
  // the previous accumulated frame, which is reprojected using motion vectors
  // and rejected where it disagrees with the local neighborhood of the current
  // frame (variance/AABB clamping, as in SVGF/TAA). This requires no retraining
  // of the denoising network.
  //
  //   accum = lerp(clamp(reproject(history), neighborhood(color)), color, alpha)
  //
  class TemporalAccumulation : public BaseOp
  {
  public:
    // Current denoised frame (read-only input)
    void setColor(const Ref<Image>& color) { this->color = color; }

    // Previous accumulated frame (read-only history input)
    void setHistory(const Ref<Image>& history) { this->history = history; }

    // Per-pixel backward motion vectors mapping the current pixel to its
    // position in the previous frame (optional; null = static reprojection)
    void setFlow(const Ref<Image>& flow) { this->flow = flow; }

    // Accumulated output frame (must differ from color and history)
    void setDst(const Ref<Image>& dst) { this->dst = dst; }

    // Weight of the current frame in [0, 1]. Smaller values are more stable but
    // introduce more lag; larger values respond faster but are less stable.
    void setAlpha(float alpha) { this->alpha = alpha; }

    // Strength of neighborhood color clamping (history rejection) in [0, +inf).
    // 0 disables clamping; larger values widen the accepted color range, i.e.
    // keep more history at the cost of more potential ghosting.
    void setClampStrength(float clampStrength) { this->clampStrength = clampStrength; }

    // If set, the history is ignored and the output equals the current frame.
    // Used for the first frame of a sequence (or after a scene cut).
    void setReset(bool reset) { this->reset = reset; }

  protected:
    void check()
    {
      if (!color || !history || !dst)
        throw std::logic_error("temporal accumulation color/history/destination not set");
      if (color->getW() != dst->getW() || color->getH() != dst->getH() ||
          history->getW() != dst->getW() || history->getH() != dst->getH())
        throw std::out_of_range("temporal accumulation image size mismatch");
      if (flow && (flow->getW() != dst->getW() || flow->getH() != dst->getH()))
        throw std::out_of_range("temporal accumulation flow image size mismatch");
    }

    Ref<Image> color;   // current denoised frame
    Ref<Image> history; // previous accumulated frame
    Ref<Image> flow;    // motion vectors (optional)
    Ref<Image> dst;     // accumulated output frame

    float alpha = 0.2f;
    float clampStrength = 1.0f;
    bool reset = false;
  };

OIDN_NAMESPACE_END
