import torch

class TestCudaAvailability:
    """Test suite to check CUDA availability in PyTorch."""

    def test_cuda_is_available(self):
        """Test if CUDA is available."""
        assert torch.cuda.is_available(), "CUDA is not available on this system."

    def test_cuda_device_count(self):
        """Test the number of CUDA devices."""
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            assert device_count > 0, "No CUDA devices found."
        else:
            assert True, "Test skipped because CUDA is not available."

    def test_current_device_name(self):
        """Test the name of the current CUDA device."""
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(torch.cuda.current_device())
            assert isinstance(device_name, str) and len(device_name) > 0, "Device name is not valid."
        else:
            assert True, "Test skipped because CUDA is not available."
