#include "triton_ops_api.h"
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <cmath>
#include <limits>
#include <algorithm>  
#include <iostream>  
#include "experiment/runtime/runtime/rt.h"
#include "kernel_launchers.h"
#include "utils.h"

namespace xllm::kernel::npu {

constexpr int64_t MAX_CORES = 65535;  
constexpr int64_t MAX_FUSED_BYTES = 65536;  

inline int64_t next_power_of_2(int64_t n) {
    if (n <= 1) return 1;
    uint64_t val = static_cast<uint64_t>(n - 1);
    return 1LL << (64 - __builtin_clzll(val ? val : 1));
}

void validate_tensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.defined(), name, " tensor is not defined");
  TORCH_CHECK(tensor.is_contiguous(), name, " tensor must be contiguous");
  TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1,
              name,
              " tensor must be on NPU device");
}

torch::Tensor layer_norm_fwd(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    double eps,
    const c10::optional<torch::Tensor>& z,
    int64_t group_size,
    bool norm_before_gate,
    bool is_rms_norm
) {
    validate_tensor(x, "x");
    validate_tensor(weight, "weight");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");

    c10::IntArrayRef x_shape_og = x.sizes();
    int64_t last_dim = x.size(-1);
    torch::Tensor x_2d = x.reshape({-1, last_dim});

    TORCH_CHECK(x_2d.stride(-1) == 1, "x stride(-1) must be 1");
    TORCH_CHECK(x_2d.dim() == 2, "x must be 2-dimensional (M, N)");

    const auto M = x_2d.size(0);
    const auto N = x_2d.size(1);

    const int64_t group_size_val = group_size;
    TORCH_CHECK(N % group_size_val == 0, "N must be divisible by group_size");
    const int64_t ngroups = N / group_size_val;

    torch::Tensor z_2d;
    if (z.has_value()) {

        TORCH_CHECK(z->stride(-1) == 1, "z stride(-1) must be 1");
        TORCH_CHECK(z->sizes() == x.sizes(), "z shape must match x");
        z_2d = z->reshape({-1, last_dim});
        validate_tensor(z_2d, "z");
        TORCH_CHECK(z_2d.dtype() == torch::kFloat32, "z must be float32");
    }

    // weight检查
    TORCH_CHECK(weight.dim() == 1 && weight.size(0) == N, 
                "weight must be 1-dimensional with size N");
    TORCH_CHECK(weight.stride(-1) == 1, "weight stride(-1) must be 1");


    if (bias.defined()) {
        TORCH_CHECK(bias.stride(-1) == 1, "bias stride(-1) must be 1");
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == N, 
                    "bias must be 1-dimensional with size N");
        validate_tensor(bias, "bias");
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    }

    torch::Tensor out_tensor = torch::empty_like(x_2d);
    TORCH_CHECK(out_tensor.sizes() == x_2d.sizes(), "out shape must match x");
    TORCH_CHECK(out_tensor.stride(-1) == 1, "out stride(-1) must be 1");

    // 均值和逆标准差张量分配
    torch::Tensor mean, rstd;
    if (!is_rms_norm) {
        mean = torch::empty({ngroups * M}, torch::dtype(torch::kFloat32).device(x.device()));
    }
    rstd = torch::empty({ngroups * M}, torch::dtype(torch::kFloat32).device(x.device()));

    // 计算块大小（BLOCK_N
    const int64_t elem_size = x.element_size();
    const int64_t MAX_FUSED_SIZE = MAX_FUSED_BYTES / elem_size;
    const int64_t BLOCK_N = std::min(MAX_FUSED_SIZE, next_power_of_2(group_size_val));

    // 检查group_size限制
    TORCH_CHECK(group_size_val <= BLOCK_N, 
                "This layer norm doesn't support feature dim >= 64KB.");

    // 计算warp数量（num_warps
    const int64_t warp_base = BLOCK_N / 256;
    const int64_t num_warps = std::clamp<int64_t>(warp_base, 1, 8);

    auto npuStream = c10_npu::getCurrentNPUStream();
    rtStream_t stream = static_cast<rtStream_t>(npuStream.stream());

    int32_t gridCoreNum = std::min(M, MAX_CORES);
    int32_t gridNgroups = ngroups;
    int32_t gridZ = 1;

    void* x_2dPtr = x_2d.data_ptr();
    void* out_tensorPtr = out_tensor.data_ptr();
    void* weightPtr = weight.data_ptr();  
    void* biasPtr = nullptr; 
    if (bias.defined()) {
        biasPtr = bias.data_ptr();
    }
    void* z_2dPtr = nullptr;
    if (z_2d.defined()) {
        z_2dPtr = z_2d.data_ptr();
    }
    void* meanPtr = nullptr;
    if (mean.defined()) {
        meanPtr = mean.data_ptr();
    }
    void* rstdPtr = rstd.data_ptr();
    int32_t stride_x_row = x_2d.stride(0);
    int32_t stride_y_row = out_tensor.stride(0);
    int32_t stride_z_row = 0;
    if (z.has_value()) {
        stride_z_row = z_2d.stride(0);
    }

    void* workspace_addr = nullptr;
    void* sync_block_lock = nullptr;

    auto ret = launchers::layer_norm_fwd_kernel(
        stream,
        gridCoreNum,
        gridNgroups,
        gridZ,
        workspace_addr,
        sync_block_lock,
        x_2dPtr,
        out_tensorPtr,
        weightPtr,
        z_2dPtr,
        meanPtr,
        rstdPtr,
        stride_x_row,
        stride_y_row,
        stride_z_row,
        M,
        group_size,
        eps
    );
    if (ret != ACL_ERROR_NONE) {
        LOG(ERROR) << "Failed to launch kernel "
        << "layer_norm_fwd_kernel" << " : error=" << ret;
    }
    cleanup(workspace_addr, sync_block_lock);
    return out_tensor.reshape(x_shape_og);
}
}

