/* Copyright 2025 The xLLM Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://github.com/jd-opensource/xllm/blob/main/LICENSE
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==============================================================================
 */

 #pragma once

 #include <torch/torch.h>
 #include <torch_npu/csrc/aten/NPUNativeFunctions.h>
 #include <torch_npu/csrc/core/npu/NPUStream.h>
 #include <torch_npu/torch_npu.h>
 
 namespace xllm::kernel::npu {
 
 torch::Tensor npu_causal_conv1d_update(
    const torch::Tensor& x,
    const torch::Tensor& conv_state,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    bool activation = true, 
    const c10::optional<torch::Tensor>& conv_state_indices = c10::nullopt,
    const c10::optional<torch::Tensor>& num_accepted_tokens = c10::nullopt,
    const c10::optional<torch::Tensor>& query_start_loc = c10::nullopt,
    int32_t max_query_len = -1,
    int32_t pad_slot_id = -1,
    const c10::optional<torch::Tensor>& block_idx_last_scheduled_token = c10::nullopt,
    const c10::optional<torch::Tensor>& initial_state_idx = c10::nullopt,
    bool validate_data = false);
 
 }  // namespace xllm::kernel::npu
 
 
 