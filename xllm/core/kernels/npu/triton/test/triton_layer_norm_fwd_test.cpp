#include <acl/acl.h>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch_npu/torch_npu.h>

#include <glog/logging.h>

#include "torch_api/triton_ops_api.h"
#include "kernel_loader.h"
#include "test_utils.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"

namespace xllm::kernel::npu {
constexpr int32_t kDeviceId = 0;

class TritonLayerNormFwdTest : public ::testing::Test {
 protected:
  void SetUp() override {
    try {
      torch::zeros({1}, torch::TensorOptions().device("npu:0"));
      tensor_options_ =
          torch::TensorOptions().dtype(torch::kFloat16).device("npu:0");
      npu_available_ = true;
    } catch (...) {
      tensor_options_ =
          torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU);
      npu_available_ = false;
      return;
    }

    kernel_name_ = "layer_norm_fwd_kernel";
    binary_filename_ = "layer_norm_fwd_kernel.npubin";

    torch::manual_seed(42);
    torch_npu::init_npu(device_str_);

    binary_path_ = GetKernelBinaryPath(binary_filename_);
    auto& loader = KernelLoader::get_instance();
    auto handle = loader.get_kernel(kernel_name_);
    if (!handle.is_valid()) {
      handle = loader.load_kernel(kernel_name_, binary_path_);
    }
    ASSERT_TRUE(handle.is_valid()) << "Failed to load Kernel: " << kernel_name_
                                   << " from " << binary_path_;
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

TEST_F(TritonLayerNormFwdTest, KernelTest) {
    if (!npu_available_) {
        GTEST_SKIP() << "NPU device not available";
    }

    auto device = at::Device(device_str_);

    int64_t batch_size = 4;
    int64_t seq_len = 16;
    int64_t hidden_dim = 64;
    float eps = 1e-6;
    int64_t group_size=seq_len;
 
    auto x = torch::randn({batch_size, seq_len, hidden_dim}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    auto weight = torch::randn({hidden_dim}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    auto bias = torch::randn({hidden_dim}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    auto z = torch::randn({batch_size, seq_len, hidden_dim}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    c10::optional<torch::Tensor> z_optional = z;

    auto npu_stream = c10_npu::getCurrentNPUStream(0);
    auto output = xllm::kernel::npu::layer_norm_fwd(x, weight, bias, eps, z_optional, group_size, true, false);
    aclrtSynchronizeStream(npu_stream.stream());
}
}

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

// int main(int argc, char** argv) {
//     bool npu_available = false;
//     // std::string device_str =
//     //     "npu:" + std::to_string(xllm::kernel::npu::kDeviceId);
//     std::string device_str = "npu:0";
//     try {
//         auto test_tensor =
//             torch::zeros({1}, torch::TensorOptions().device(device_str));
//         (void)test_tensor;
//         npu_available = true;
//     } catch (...) {
//         npu_available = false;
//     }

//     if (!npu_available) {
//         LOG(WARNING) << "NPU device not available, skipping all tests.";
//         return 0;
//     }

//     torch::manual_seed(42);
//     torch_npu::init_npu(device_str);

//     auto device = at::Device(device_str);
//     int64_t batch_size = 4;
//     int64_t seq_len = 16;
//     int64_t hidden_dim = 64;
//     float eps = 1e-6;
//     int64_t group_size=seq_len;
 
//     auto x = torch::randn({batch_size, seq_len, hidden_dim}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
//     auto weight = torch::randn({hidden_dim}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
//     auto bias = torch::randn({hidden_dim}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
//     auto z = torch::randn({batch_size, seq_len, hidden_dim}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
//     c10::optional<torch::Tensor> z_optional = z;

//     auto npu_stream = c10_npu::getCurrentNPUStream(0);
//     auto output = xllm::kernel::npu::layer_norm_fwd(x, weight, bias, eps, z_optional, group_size, true, false);
//     aclrtSynchronizeStream(npu_stream.stream());

//     return 0;
// }

