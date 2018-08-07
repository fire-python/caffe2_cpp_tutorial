#include "caffe2/util/blob.h"
#include "caffe2/util/tensor.h"

#ifdef WITH_CUDA
#include <caffe2/core/context_gpu.h>
#endif

namespace caffe2 {

TensorCPU BlobUtil::Get() {
  return blob_.Get<Tensor>().Clone();
}

void BlobUtil::Set(const Tensor &value, bool force_cuda) {
#ifdef WITH_CUDA
  if (force_cuda) {
    auto tensor = blob_.GetMutableTensor(DeviceType::CUDA);
    tensor->CopyFrom(value);
    return;
  }
#endif
  auto tensor = blob_.GetMutableTensor(DeviceType::CPU);
  tensor->ResizeLike(value);
  tensor->ShareData(value);
}

void BlobUtil::Print(const std::string &name, int max) {
  auto tensor = Get();
  TensorUtil(tensor).Print(name, max);
}

}  // namespace caffe2
