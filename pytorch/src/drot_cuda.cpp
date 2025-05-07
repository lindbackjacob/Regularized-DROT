#include <torch/extension.h>

torch::Tensor quadratic_drot_cuda(
    torch::Tensor c,
    torch::Tensor p,
    torch::Tensor q,
    float rho,
    float r_weight,
    int max_iters,
    float eps);

torch::Tensor group_lasso_drot_cuda(
    torch::Tensor c,
    torch::Tensor p,
    torch::Tensor q,
    int n_class,
    float rho,
    float r_weight,
    int max_iters,
    float eps);

void warm_start_q_drot_cuda(
    torch::Tensor c,
    torch::Tensor p,
    torch::Tensor q,
    torch::Tensor x, // torch::Tensor &x ? 
    torch::Tensor phi1,
    torch::Tensor phi2,
    torch::Tensor a,
    torch::Tensor b,
    float rho,
    float r_weight,
    int n_iters,
    float eps);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor quadratic_drot(
    torch::Tensor c,
    torch::Tensor p,
    torch::Tensor q,
    float rho,
    float r_weight,
    int max_iters,
    float eps) {
    
    CHECK_INPUT(c);
    CHECK_INPUT(p);
    CHECK_INPUT(q);
    
    return quadratic_drot_cuda(c, p, q, rho, r_weight, max_iters, eps);
}

torch::Tensor group_lasso_drot(
    torch::Tensor c,
    torch::Tensor p,
    torch::Tensor q,
    int n_class,
    float rho,
    float r_weight,
    int max_iters,
    float eps) {
    
    CHECK_INPUT(c);
    CHECK_INPUT(p);
    CHECK_INPUT(q);

    return group_lasso_drot_cuda(c, p, q, n_class, rho, r_weight, max_iters, eps);
}

void warm_start_q_drot(
    torch::Tensor c,
    torch::Tensor p,
    torch::Tensor q,
    torch::Tensor x, // torch::Tensor &x ?
    torch::Tensor phi1,
    torch::Tensor phi2,
    torch::Tensor a,
    torch::Tensor b,
    float rho,
    float r_weight,
    int n_iters,
    float eps) {

    CHECK_INPUT(c);
    CHECK_INPUT(p);
    CHECK_INPUT(q);
    CHECK_INPUT(x);
    CHECK_INPUT(phi1);
    CHECK_INPUT(phi2);
    CHECK_INPUT(a);
    CHECK_INPUT(b);

    warm_start_q_drot_cuda(c, p, q, x, phi1, phi2, a, b, rho, r_weight, n_iters, eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quadratic_drot", &quadratic_drot, "Quadratic Regularized DROT");
  m.def("group_lasso_drot", &group_lasso_drot, "Group-Lasso Regularized DROT");
  m.def("warm_start_q_drot", &warm_start_q_drot, "Warm Start QDROT");
}
