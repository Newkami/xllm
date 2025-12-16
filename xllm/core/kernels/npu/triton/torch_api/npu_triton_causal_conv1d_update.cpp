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

 #include "npu_triton_causal_conv1d_update.h"
 #include <torch_npu/csrc/core/npu/NPUStream.h>
 #include <cmath>
 #include <limits>
 #include "experiment/runtime/runtime/rt.h"
 #include "kernel_launchers.h"
 #include "utils.h"

namespace xllm::kernel::npu {

torch::Tensor npu_causal_conv1d_update(
    const torch::Tensor& x,
    const torch::Tensor& conv_state,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    bool activation, 
    const c10::optional<torch::Tensor>& conv_state_indices,
    const c10::optional<torch::Tensor>& num_accepted_tokens,
    const c10::optional<torch::Tensor>& query_start_loc,
    int32_t max_query_len,
    int32_t pad_slot_id,
    const c10::optional<torch::Tensor>& block_idx_last_scheduled_token,
    const c10::optional<torch::Tensor>& initial_state_idx,
    bool validate_data) {
    
    torch::Tensor out = torch::empty(
        x.sizes(), torch::TensorOptions().dtype(x.dtype()).device(x.device()));
    void* workspace_addr = nullptr;
    void* sync_block_lock = nullptr;
    // auto ret = setup("causal_conv1d_update_kernel", &workspace_addr, &sync_block_lock, gridX * gridY * gridZ);
    if (ret != ACL_ERROR_NONE) {
        LOG(ERROR) << "Failed to setup workspace and sync block lock for kernel "
        << "causal_conv1d_update_kernel" << " : error=" << ret;
        return 
    }
    // launch kernel

    if (ret != ACL_ERROR_NONE) {
        LOG(ERROR) << "Failed to launch kernel "
        << "causal_conv1d_update_kernel" << " : error=" << ret;
    }
    // cleanup(workspace_addr, sync_block_lock);
    return out;
}
    
}  // namespace xllm::kernel::npu
    