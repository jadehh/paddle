#pragma once

#include "paddle/phi/core/meta_tensor.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/common/int_array.h"

namespace phi {

void AllcloseInferMeta(const MetaTensor& x, const MetaTensor& y, const Scalar& rtol, const Scalar& atol, bool equal_nan, MetaTensor* out);

void BreluInferMeta(const MetaTensor& x, float t_min, float t_max, MetaTensor* out);

void ClipInferMeta(const MetaTensor& x, const Scalar& min, const Scalar& max, MetaTensor* out);

void CumprodInferMeta(const MetaTensor& x, int dim, MetaTensor* out);

void EluInferMeta(const MetaTensor& x, float alpha, MetaTensor* out);

void FmaxInferMeta(const MetaTensor& x, const MetaTensor& y, int axis, MetaTensor* out);

void FminInferMeta(const MetaTensor& x, const MetaTensor& y, int axis, MetaTensor* out);

void FullInferMeta(const IntArray& shape, const Scalar& value, DataType dtype, MetaTensor* out);

void Full_likeInferMeta(const MetaTensor& x, const Scalar& value, DataType dtype, MetaTensor* out);

void GeluInferMeta(const MetaTensor& x, bool approximate, MetaTensor* out);

void Hard_shrinkInferMeta(const MetaTensor& x, float threshold, MetaTensor* out);

void Hard_sigmoidInferMeta(const MetaTensor& x, float slope, float offset, MetaTensor* out);

void Hard_swishInferMeta(const MetaTensor& x, float threshold, float scale, float offset, MetaTensor* out);

void IscloseInferMeta(const MetaTensor& x, const MetaTensor& y, const Scalar& rtol, const Scalar& atol, bool equal_nan, MetaTensor* out);

void Label_smoothInferMeta(const MetaTensor& label, paddle::optional<const MetaTensor&> prior_dist, float epsilon, MetaTensor* out);

void Leaky_reluInferMeta(const MetaTensor& x, float alpha, MetaTensor* out);

void LogitInferMeta(const MetaTensor& x, float eps, MetaTensor* out);

void Matrix_powerInferMeta(const MetaTensor& x, int n, MetaTensor* out);

void Matrix_rankInferMeta(const MetaTensor& x, float tol, bool use_default_tol, bool hermitian, MetaTensor* out);

void MishInferMeta(const MetaTensor& x, float lambda, MetaTensor* out);

void PowInferMeta(const MetaTensor& x, const Scalar& s, MetaTensor* out);

void Put_along_axisInferMeta(const MetaTensor& x, const MetaTensor& index, const MetaTensor& value, int axis, const std::string& reduce, MetaTensor* out);

void ScaleInferMeta(const MetaTensor& x, const Scalar& scale, float bias, bool bias_after_scale, MetaTensor* out);

void SeluInferMeta(const MetaTensor& x, float scale, float alpha, MetaTensor* out);

void Soft_shrinkInferMeta(const MetaTensor& x, float lambda, MetaTensor* out);

void SwishInferMeta(const MetaTensor& x, float beta, MetaTensor* out);

void Take_along_axisInferMeta(const MetaTensor& x, const MetaTensor& index, int axis, MetaTensor* out);

void Thresholded_reluInferMeta(const MetaTensor& x, float threshold, MetaTensor* out);

}  // namespace phi
