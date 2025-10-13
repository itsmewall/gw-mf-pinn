import cupy as cp
print('GPUs:', cp.cuda.runtime.getDeviceCount())
x = cp.random.randn(1<<18, dtype=cp.float32)
y = cp.fft.rfft(x)
print('ok cupy fft:', y.shape)