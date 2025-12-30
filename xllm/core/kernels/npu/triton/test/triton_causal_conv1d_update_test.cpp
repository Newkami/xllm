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

#include <acl/acl.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch_npu/torch_npu.h>
#include <torch/nn/functional/conv.h>

#include <optional>

#include "kernel_loader.h"
#include "test_utils.h"
#include "torch_api/triton_ops_api.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"

namespace xllm::kernel::npu {

constexpr float kTolerance = 5e-2f;  // bfloat16 tolerance
constexpr int32_t kDeviceId = 0;

torch::Tensor causal_conv1d_update_ref(
    torch::Tensor x,
    torch::Tensor conv_state,
    torch::Tensor weight,
    const std::optional<torch::Tensor>& bias,
    bool activation) {
  auto dtype_in = x.scalar_type();
  bool unsqueeze = x.dim() == 2;
  if (unsqueeze) {
    x = x.unsqueeze(-1);
  }

  int64_t batch = x.size(0);
  int64_t dim = x.size(1);
  int64_t seqlen = x.size(2);
  int64_t width = weight.size(1);
  int64_t state_len = conv_state.size(2);

  // Concatenate conv_state and x along the last dimension
  auto x_new = torch::cat({conv_state, x}, -1).to(weight.dtype());
  
  // Update conv_state with the last state_len elements
  conv_state.copy_(x_new.slice(2, seqlen, seqlen + state_len));

  // Prepare weight for grouped convolution: (dim, width) -> (dim, 1, width)
  // For grouped conv1d, weight should be (dim, 1, width)
  auto weight_expanded = weight.unsqueeze(1);  // (dim, 1, width)
  
  // x_new is (batch, dim, state_len + seqlen)
  // For grouped convolution with groups=dim, we need to reshape:
  // x_new: (batch, dim, L) -> (batch, dim, 1, L) -> (batch * dim, 1, L)
  // But actually, PyTorch conv1d expects (N, C, L) where N=batch, C=dim
  // For grouped conv, it will automatically handle it
  
  // Use functional conv1d
  torch::Tensor out;
  if (bias.has_value()) {
    out = torch::nn::functional::conv1d(x_new, weight_expanded, bias.value(),
                                         torch::nn::functional::Conv1dFuncOptions().groups(dim).padding(0));
  } else {
    out = torch::nn::functional::conv1d(x_new, weight_expanded, torch::Tensor(),
                                         torch::nn::functional::Conv1dFuncOptions().groups(dim).padding(0));
  }
  
  // Take only the last seqlen elements
  out = out.slice(2, -seqlen);

  // Apply activation if needed
  if (activation) {
    out = torch::silu(out);
  }

  if (unsqueeze) {
    out = out.squeeze(-1);
  }

  return out.to(dtype_in);
}

class TritonCausalConv1dUpdateTest : public ::testing::Test {
 protected:
  void SetUp() override {
    try {
      torch::zeros({1}, torch::TensorOptions().device("npu:0"));
      tensor_options_ =
          torch::TensorOptions().dtype(torch::kBFloat16).device("npu:0");
      npu_available_ = true;
    } catch (...) {
      tensor_options_ =
          torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCPU);
      npu_available_ = false;
      return;
    }

    torch::manual_seed(42);
    torch_npu::init_npu(device_str_);
    kernel_name_ = "_causal_conv1d_update_kernel_no_cache_len_no_mtp";
    binary_filename_ = "_causal_conv1d_update_kernel_no_cache_len_no_mtp.npubin";
    binary_path_ = GetKernelBinaryPath(binary_filename_);
    auto& loader = KernelLoader::get_instance();
    auto handle = loader.get_kernel(kernel_name_);
    if (!handle.is_valid()) {
      handle = loader.load_kernel(kernel_name_, binary_path_);
    }
    // Note: kernel loading might fail if binary doesn't exist, but we can still test
    // the API call
  }

  void TearDown() override {
    if (npu_available_) {
      try {
        torch_npu::finalize_npu();
      } catch (...) {
      }
    }
  }

  torch::TensorOptions tensor_options_;
  bool npu_available_ = false;
  std::string device_str_ = "npu:" + std::to_string(kDeviceId);
  std::string binary_filename_;
  std::string kernel_name_;
  std::string binary_path_;
};

TEST_F(TritonCausalConv1dUpdateTest, MultiBatchTest) {
  if (!npu_available_) {
    GTEST_SKIP() << "NPU device not available";
  }

  auto device = at::Device(device_str_);
  constexpr int64_t batch = 4;
  constexpr int64_t dim = 2048;
  constexpr int64_t width = 4;
  constexpr int64_t seqlen = 1;
  constexpr bool has_bias = false;
  constexpr bool silu_activation = true;

  torch::manual_seed(0);
  auto dtype = torch::kBFloat16;
  float rtol = 1e-2f;
  float atol = 5e-2f;

  // Create input tensors
  auto x = torch::randn({batch, dim, seqlen}, 
                       torch::TensorOptions().dtype(dtype).device(device));
  auto x_ref = x.clone().cpu();
  
  auto conv_state = torch::randn({batch, dim, width - 1}, 
                                 torch::TensorOptions().dtype(dtype).device(device));
  auto conv_state_ref = conv_state.detach().clone().cpu();
  
  auto weight = torch::randn({dim, width}, 
                             torch::TensorOptions().dtype(dtype).device(device));
  auto weight_ref = weight.clone().cpu();
  
  std::optional<torch::Tensor> bias = std::nullopt;
  if (has_bias) {
    bias = torch::randn({dim}, torch::TensorOptions().dtype(dtype).device(device));
  }

  // Create conv_state_indices for continuous batching
  auto conv_state_indices = torch::arange(batch, torch::kInt32, device);

  // Run reference implementation on CPU
  auto out_ref = causal_conv1d_update_ref(
      x_ref, conv_state_ref, weight_ref, std::nullopt, silu_activation);

  // Run NPU kernel
  auto npu_stream = c10_npu::getCurrentNPUStream(kDeviceId);
  auto out = npu_causal_conv1d_update(
      x, conv_state, weight, 
      bias.has_value() ? bias.value() : torch::Tensor(),
      silu_activation,
      std::nullopt,  // cache_seqlens
      conv_state_indices,
      std::nullopt,  // num_accepted_tokens
      std::nullopt,  // query_start_loc
      -1,            // max_query_len
      std::nullopt,  // intermediate_conv_window
      -1,            // pad_slot_id
      false          // validate_data
  );
  aclrtSynchronizeStream(npu_stream.stream());

  // Compare results
  auto out_cpu = out.cpu();
  auto output_diff = (out_ref - out_cpu).abs();
  float max_diff = output_diff.max().item<float>();
  
  EXPECT_LT(max_diff, atol) 
      << "Output mismatch: max diff = " << max_diff
      << ", tolerance = " << atol
      << ", shape: " << out_cpu.sizes()
      << ", ref range [" << out_ref.min().item<float>() << ", " << out_ref.max().item<float>() << "]"
      << ", actual range [" << out_cpu.min().item<float>() << ", " << out_cpu.max().item<float>() << "]";

  // Compare conv_state (it should be updated)
  auto conv_state_cpu = conv_state.cpu();
  auto state_diff = (conv_state_ref - conv_state_cpu).abs();
  float max_state_diff = state_diff.max().item<float>();
  
  // Note: conv_state comparison might have some differences due to numerical precision
  // We use a more relaxed tolerance for state comparison
  EXPECT_LT(max_state_diff, atol * 2.0f)
      << "Conv state mismatch: max diff = " << max_state_diff
      << ", tolerance = " << (atol * 2.0f);
}

}  // namespace xllm::kernel::npu

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  bool npu_available = false;
  std::string device_str =
      "npu:" + std::to_string(xllm::kernel::npu::kDeviceId);
  try {
    auto test_tensor =
        torch::zeros({1}, torch::TensorOptions().device(device_str));
    (void)test_tensor;
    npu_available = true;
  } catch (...) {
    npu_available = false;
  }

  if (!npu_available) {
    LOG(WARNING) << "NPU device not available, skipping all tests.";
    return 0;
  }

  int result = RUN_ALL_TESTS();
  return result;
}

