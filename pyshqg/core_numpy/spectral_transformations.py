import numpy as np
import pyshtools as pysh
import scipy.fftpack

class SpectralTransformations:
    
    def __init__(
        self,
        T,
        T_grid,
        planet_radius=1,
        planet_omega=1,
    ):
        lats, _ = pysh.expand.GLQGridCoord(T_grid)
        cost, w = pysh.expand.SHGLQ(T_grid)
        cosl = np.cos(lats*np.pi/180)
        PLM = np.zeros((T_grid+1, T+1, T+1))
        PPLM = np.zeros((T_grid+1, T+1, T+1))
        ALM = np.zeros((T_grid+1, T+1, T+1))
        PW = np.zeros((T_grid+1, T+1, T+1))
        for i in range(T_grid+1):
            p, a = pysh.legendre.PlmBar_d1(T, cost[i], cnorm=1, csphase=1)
            for l in range(T+1):
                for m in range(l+1):
                    ind = ( l * (l+1) ) // 2 + m
                    PLM[i, m, l] = p[ind]
                    PPLM[i, m, l] = p[ind] / (cosl[i] * planet_radius)
                    ALM[i, m, l] = a[ind] * cosl[i] / planet_radius
                    PW[i, m, l] = 0.5 * p[ind] * w[i]
        laplacian_spectrum = np.zeros(T+1)
        for l in range(T+1):
            laplacian_spectrum[l] = -l*(l+1) / planet_radius**2
        self.T = T
        self.T_grid = T_grid
        self.PLM = PLM
        self.PPLM = PPLM
        self.ALM = ALM
        self.PW = PW
        self.laplacian_spectrum = laplacian_spectrum
        self.planet_radius = planet_radius
        self.planet_omega = planet_omega
        self.apply_fft = self.apply_fft_v2
        self.apply_ifft = self.apply_ifft_v2
        self.lats = lats[::-1]
        self.lons = np.mod(np.linspace(
            0,
            360,
            2*T_grid+2,
            endpoint=False,
        ) - 180, 360)

    def apply_fft_v1(self, ab_im):
        shape = list(ab_im.shape)
        batch_shape = shape[:-3]
        abh = np.zeros(
            tuple(batch_shape+[self.T_grid+1, 2*self.T_grid+2]),
            dtype=np.complex128,
        )
        abh[..., :self.T+1] = ab_im[..., 0, :, :] + 1j*ab_im[..., 1, :, :]
        abh[..., -self.T:] = ab_im[..., 0, :, 1:][..., ::-1] - 1j*ab_im[..., 1, :, 1:][..., ::-1]
        return np.fft.fft(abh, axis=-1).real

    def apply_fft_v2(self, ab_im):
        shape = list(ab_im.shape)
        batch_shape = shape[:-3]
        abh = np.zeros(
            tuple(batch_shape+[self.T_grid+1, self.T_grid+2]),
            dtype=np.complex128,
        )
        abh[..., :self.T+1] = ab_im[..., 0, :, :] + 1j*ab_im[..., 1, :, :]
        return np.fft.hfft(abh, axis=-1)

    def apply_fft_v3(self, ab_im):
        shape = list(ab_im.shape)
        batch_shape = shape[:-3]
        ah = np.zeros(tuple(batch_shape+[self.T_grid+1, 2*self.T_grid+3]))
        ah[..., ::2][..., :self.T+1] = ab_im[..., 0, :, :]
        ah[..., :, 0] = ab_im[..., 0, :, 0]
        ch = scipy.fftpack.dct(ah, type=1, axis=-1)[..., :-1]        
        bh = np.zeros(tuple(batch_shape+[self.T_grid+1, 2*self.T_grid+1]))
        bh[..., 1::2][..., :self.T] = ab_im[..., 1, :, 1:]
        sh = scipy.fftpack.dst(bh, type=1, axis=-1)
        ch[..., 1:] += sh
        return ch

    def apply_ifft_v1(self, grid):
        abh = np.fft.ifft(grid, axis=-1)[..., :self.T+1]
        return np.stack([abh.real, abh.imag], axis=-3)

    def apply_ifft_v2(self, grid):
        abh = np.fft.ihfft(grid, axis=-1)[..., :self.T+1]
        return np.stack([abh.real, abh.imag], axis=-3)

    def apply_ifft_v3(self, grid):
        shape = list(grid.shape)
        batch_shape = shape[:-2]
        gh = np.zeros(tuple(batch_shape+[self.T_grid+1, 2*self.T_grid+3]))
        gh[..., 0] = 2*grid[..., 0]
        gh[..., 1:-1] = grid[..., 1:]
        ch = scipy.fftpack.dct(gh, type=1, axis=-1)#[..., ::2]
        return ch
        gh = grid[..., 1:]
        sh = scipy.fftpack.dst(gh, type=1, axis=-1)[..., 1::2]
        clm = np.zeros(tuple(batch_shape+[2, self.T_grid+1, self.T_grid+1]))
        print(ch.shape)
        print(sh.shape)
        clm[..., 0, :, :] = ch[..., :-1]
        clm[..., 1, :, 1:] = sh
        return clm
        
    def spec_to_grid_generic(self, clm, plm):
        ab_im = np.einsum('...jm,imj->...im', clm, plm)
        return self.apply_fft(ab_im)
        
    def spec_to_grid(self, clm):
        return self.spec_to_grid_generic(clm, self.PLM)
    
    def spec_to_grid_grad_theta(self, clm):
        return self.spec_to_grid_generic(clm, self.ALM)
    
    def spec_to_grid_grad_phi(self, clm):
        clm_gp = np.zeros(clm.shape)
        clm_gp[..., 0, :, :] = - np.arange(self.T+1) * clm[..., 1, :, :]
        clm_gp[..., 1, :, :] = np.arange(self.T+1) * clm[..., 0, :, :]
        return self.spec_to_grid_generic(clm_gp, self.PPLM)

    def grid_to_spec(self, grid):
        ab_im = self.apply_ifft(grid)
        return np.einsum('...im,iml->...lm', ab_im, self.PW)

    def precompute_planet_vorticity(self):
        spec_f = np.zeros((2, self.T+1, self.T+1))
        spec_f[0, 1, 0] = 2 * self.planet_omega / np.sqrt(3)
        return self.spec_to_grid(spec_f)
        