#include <acl/acl.h>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch_npu/torch_npu.h>

#include <glog/logging.h>

#include "torch_api/triton_ops_api.h"
#include "kernel_loader.h"
#include "test_utils.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include <iostream>
#include <iomanip>
#include <vector>

namespace xllm::kernel::npu {
constexpr int32_t kDeviceId = 0;
constexpr float kTolerance = 1e-3f;

// 自定义函数：复刻 Python 风格打印 3 维张量 (B, S, D)
void print_tensor_like_python(const torch::Tensor& tensor, int64_t max_seq_show = 6, int64_t max_feat_show = 6) {
    // 前置校验：确保是 3 维 float32 张量（可根据需求扩展）
    TORCH_CHECK(tensor.dim() == 3, "仅支持 3 维张量打印");
    TORCH_CHECK(tensor.dtype() == torch::kFloat32, "仅支持 float32 张量打印");
    TORCH_CHECK(tensor.is_cpu(), "仅支持 CPU 张量打印");

    // 获取张量维度
    int64_t B = tensor.size(0);  // batch_size
    int64_t S = tensor.size(1);  // seq_len
    int64_t D = tensor.size(2);  // feat_dim
    const float* data = tensor.data_ptr<float>();

    // 配置输出精度（与 Python 默认一致）
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "tensor([";

    // 遍历 Batch 维度（最外层括号）
    for (int64_t b = 0; b < B; ++b) {
        // Batch 内部换行 + 对齐（与 Python 一致）
        if (b > 0) {
            std::cout << ",\n        ";
        }
        std::cout << "[";

        // 计算 Seq 维度的显示范围（超过 max_seq_show 则显示前N个+后N个，中间用...省略）
        bool seq_need_ellipsis = (S > max_seq_show);
        std::vector<int64_t> seq_indices;  // 要显示的 seq 索引

        if (seq_need_ellipsis) {
            // 前 3 个 + 后 3 个（可自定义数量，与 Python 一致）
            for (int64_t s = 0; s < 3; ++s) seq_indices.push_back(s);
            seq_indices.push_back(-1);  // 标记省略号
            for (int64_t s = S - 3; s < S; ++s) seq_indices.push_back(s);
        } else {
            // 全部显示
            for (int64_t s = 0; s < S; ++s) seq_indices.push_back(s);
        }

        // 遍历 Seq 维度（中间层括号）
        for (size_t si = 0; si < seq_indices.size(); ++si) {
            int64_t s = seq_indices[si];

            // 处理省略号
            if (s == -1) {
                if (si > 0) std::cout << ",\n         ";
                std::cout << "...";
                continue;
            }

            // Seq 内部换行 + 对齐
            if (si > 0) {
                std::cout << ",\n         ";
            }
            std::cout << "[";

            // 计算 Feat 维度的显示范围（超过 max_feat_show 则显示前N个+后N个，中间用...省略）
            bool feat_need_ellipsis = (D > max_feat_show);
            std::vector<int64_t> feat_indices;  // 要显示的 feat 索引

            if (feat_need_ellipsis) {
                // 前 3 个 + 后 3 个（可自定义数量，与 Python 一致）
                for (int64_t d = 0; d < 3; ++d) feat_indices.push_back(d);
                feat_indices.push_back(-1);  // 标记省略号
                for (int64_t d = D - 3; d < D; ++d) feat_indices.push_back(d);
            } else {
                // 全部显示
                for (int64_t d = 0; d < D; ++d) feat_indices.push_back(d);
            }

            // 遍历 Feat 维度（内层括号，元素值）
            for (size_t di = 0; di < feat_indices.size(); ++di) {
                int64_t d = feat_indices[di];

                // 处理省略号
                if (d == -1) {
                    if (di > 0) std::cout << ", ";
                    std::cout << "...";
                    continue;
                }

                // 元素值对齐输出
                if (di > 0) {
                    std::cout << ", ";
                }
                // 计算元素索引（C 风格行优先存储）
                int64_t idx = b * S * D + s * D + d;
                // 格式化输出（占宽 10 位，对齐Python样式）
                std::cout << std::setw(10) << data[idx];
            }

            std::cout << "]";
        }

        std::cout << "]";
    }

    std::cout << "])" << std::endl;
    // 恢复默认输出格式
    std::cout << std::resetiosflags(std::ios::fixed);
}

torch::Tensor layer_norm_golden_cpu(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    double eps = 1e-6,
    const c10::optional<torch::Tensor>& z = c10::nullopt,
    const int64_t group_size = -1,
    bool norm_before_gate = true,
    bool is_rms_norm = false
) {

    // 1. 输入预处理
    auto x_shape_og = x.sizes();
    int64_t last_dim = x.size(-1);
    torch::Tensor x_2d = x.reshape({-1, last_dim}).cpu().contiguous(); // (M, N) = (batch*seq_len, feat_dim)
    int64_t M = x_2d.size(0);
    int64_t N = x_2d.size(1);

    // 2. 辅助变量初始化（确保设备/类型一致，内存连续）
    torch::Tensor weight_cpu = weight.cpu().contiguous();
    torch::Tensor bias_cpu;
    if (bias.defined()) {
        bias_cpu = bias.cpu().contiguous();
    }

    torch::Tensor z_2d;
    if (z.has_value()) {
        TORCH_CHECK(z->stride(-1) == 1, "z stride(-1) must be 1");
        TORCH_CHECK(z->sizes() == x.sizes(), "z shape must match x");
        z_2d = z->cpu().reshape({-1, last_dim}).contiguous();
    }

    // 3. 分组配置
    int64_t group_size_val = group_size;

    TORCH_CHECK(N % group_size_val == 0,
                "N=" + std::to_string(N) + " must be divisible by group_size=" + std::to_string(group_size_val));
    int64_t ngroups = N / group_size_val;

    // 4. 门控逻辑（若有 Z 且后置门控：先应用 z * sigmoid(z)）
    torch::Tensor x_input;
    if (z_2d.defined() && !norm_before_gate) {
        torch::Tensor gate = z_2d * torch::sigmoid(z_2d);
        x_input = x_2d * gate; // 直接使用原始x_2d计算，无克隆
    } else {
        x_input = x_2d; // 无需门控，直接引用原始张量
    }

    // 5. 分组归一化
    // 拆分：(M, N) → (M, ngroups, group_size) → (M*ngroups, group_size)
    torch::Tensor x_grouped = x_input.unfold(1, group_size, group_size); // (M, ngroups, group_size)
    torch::Tensor x_grouped_flat = x_grouped.reshape({-1, group_size}); // (M*ngroups, group_size)

    // 6. 核心归一化
    torch::Tensor x_norm_flat;
    if (!is_rms_norm) {
        // 标准 LayerNorm：均值中心化 + 方差归一化（
        x_norm_flat = torch::layer_norm(
            x_grouped_flat,
            torch::IntArrayRef({group_size}), // normalized_shape
            torch::Tensor(), // weight = None
            torch::Tensor(), // bias = None
            eps
        );
    } else {
        // RMSNorm：无中心化，仅均方根归一化（x / sqrt(mean(x²) + eps)）
        torch::Tensor x_sq = torch::pow(x_grouped_flat, 2);
        torch::Tensor mean_sq = torch::mean(x_sq, /*dim=*/-1, /*keepdim=*/true); // (M*ngroups, 1)
        torch::Tensor rsqrt_mean_sq = torch::rsqrt(mean_sq + eps); // 1 / sqrt(mean_sq + eps)
        x_norm_flat = x_grouped_flat * rsqrt_mean_sq;
    }

    // 7. 还原分组形状：(M*ngroups, group_size) → (M, ngroups, group_size) → (M, N)
    torch::Tensor x_norm_grouped = x_norm_flat.reshape({M, ngroups, group_size}); // 还原分组维度
    torch::Tensor x_norm = x_norm_grouped.contiguous().view({M, N}); // 合并分组 → (M, N)

    // 8. 线性变换
    torch::Tensor y = x_norm * weight_cpu; // (M, N) * (N,) → 广播相乘
    if (bias_cpu.defined()) {
        y = y + bias_cpu;
    }

    // 9. 门控逻辑（若有 Z 且前置门控：归一化后应用 z * sigmoid(z)）
    if (z_2d.defined() && norm_before_gate) {
        torch::Tensor gate = z_2d * torch::sigmoid(z_2d); // (M, N)
        y = y * gate;
    }

    // 10. 恢复原始形状并返回
    return y.reshape(x_shape_og);
}

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

// (2, 8, 128, False, True, True, false, None),  
// x shape:[2*8, 128]
// weight shape:[128,]
// z shape:[2*8, 128]
// group_size:None
// norm_before_gate = true
// bool is_rms_norm = false
TEST_F(TritonLayerNormFwdTest, KernelTest2) {
    if (!npu_available_) {
        GTEST_SKIP() << "NPU device not available";
    }
   
    auto device = at::Device(device_str_);

    int64_t batch_size = 2;
    int64_t seq_len = 8;
    int64_t hidden_dim = 128;
    float eps = 1e-6;
    int64_t group_size=hidden_dim;
 
    auto dtype = torch::kFloat32;
    auto tensor_options = torch::TensorOptions().dtype(torch::kFloat32).device(device);

    auto x = torch::randn({batch_size, seq_len, hidden_dim}, tensor_options);
    auto weight = torch::randn({hidden_dim}, tensor_options);
    torch::Tensor bias;
    auto z = torch::randn({batch_size, seq_len, hidden_dim}, tensor_options);
    c10::optional<torch::Tensor> z_optional = z;

    auto output_golden = layer_norm_golden_cpu(x, weight, bias, eps, z_optional, group_size, true, false);
    auto npu_stream = c10_npu::getCurrentNPUStream(0);
    auto output = xllm::kernel::npu::layer_norm_fwd(x, weight, bias, eps, z_optional, group_size, true, false);
    aclrtSynchronizeStream(npu_stream.stream());

    auto output_golden_cpu = output_golden.cpu().contiguous();
    auto output_cpu = output.cpu().contiguous();

    std::cout << "==============output_golden_cpu==============" << std::endl;
    print_tensor_like_python(output_golden_cpu);

    std::cout << "==============output_cpu==============" << std::endl;
    print_tensor_like_python(output_cpu);

    auto output_diff = torch::abs(output_cpu - output_golden_cpu);
    float output_max_diff = torch::max(output_diff).item().to<float>();

    EXPECT_LT(output_max_diff, kTolerance) 
        << "LayerNorm output max diff (" << output_max_diff 
        << ") > tolerance (" << kTolerance << ")";
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


