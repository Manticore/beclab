from reikna.helpers import product
import reikna.cluda.dtypes as dtypes


class Drift:

    def __init__(self, module, dtype, components=1):
        self.module = module
        self.components = components
        self.dtype = dtypes.normalize_type(dtype)

    def __process_modules__(self, process):
        return Drift(process(self.module), self.dtype, components=self.components)


class Diffusion:

    def __init__(self, module, dtype, components=1, noise_sources=1, real_noise=False):
        self.module = module
        self.components = components
        self.noise_sources = noise_sources
        self.real_noise = real_noise
        self.dtype = dtypes.normalize_type(dtype)

    def __process_modules__(self, process):
        return Diffusion(
            process(self.module), self.dtype,
            components=self.components, noise_sources=self.noise_sources,
            real_noise=self.real_noise)
