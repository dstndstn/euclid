from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', family='serif')
import numpy as np
import pylab as plt
from astrometry.util.plotutils import PlotSequence
import tractor

ps = PlotSequence('coadd', suffix='pdf')

plt.figure(figsize=(4,3))
plt.subplots_adjust(left=0.15, bottom=0.15, right=0.97, top=0.99)

# Case 1: coadd two images with same noise and PSF -- same performance as individual exposures
# Case 2: different noise, same PSF
# Case 3: same noise, different PSF
# Case 4: different noise, different PSF, but same S/N

# For each, scan through a range of coadd amplitudes
# In all cases, use the true PSF model and no actual noise first, and then simulate
# a bunch of noisy runs.

H,W = 50,50
trueflux = 100.
Nruns = 10000

# The cases we'll consider.  The list of tuples are the per-pixel
# noise and PSF standard deviation for the two images.
for name,imageset in [
        ('Case 1: Same noise and PSF', [(1., 2.), (1., 2.)]),
        ('Case 2: Different noise, same PSF', [(1., 2.), (2., 2.)]),
        ('Case 3: Same noise, different PSF', [(1., 2.), (1., 4.)]),
        ('Case 4: Different noise, different PSF', [(1., 2.), (0.5, 4.)]),
        ]:

    print()
    print(name)
    print()

    # We're simulating a single isolated point source
    src = tractor.PointSource(tractor.PixPos(W//2, H//2),
                              tractor.Flux(trueflux))

    # Produce tractor image objects with noise model, PSF model, etc
    tims = []
    for noise, psf_size in imageset:
        tim = tractor.Image(np.zeros((H,W), np.float32),
                            inverr=np.ones((H,W), np.float32) * (1./noise),
                            psf=tractor.NCircularGaussianPSF([psf_size], [1.]),
                            photocal=tractor.LinearPhotoCal(1.))
        # Create noiseless model image (simulated image)
        tr = tractor.Tractor([tim], [src])
        mod = tr.getModelImage(0)
        tim.data = mod
        tims.append(tim)

    # First we'll run without any noise added to the images, to get estimates of ideal performance.

    # Run the Simultaneous Fitting method
    tr = tractor.Tractor(tims, [src])
    src.brightness.setParams([0.])
    # Freeze the source position -- only fit for flux
    src.freezeParam('pos')
    # Freeze the image calibration parameters (PSF model, sky background, photometric calibration, etc)
    tr.freezeParam('images')
    phot = tr.optimize_forced_photometry(variance=True)
    simult_err = 1./np.sqrt(phot.IV[0])
    print('Forced-phot of individual images simultaneously =', src.getBrightness().getValue(), '+-', simult_err)
    
    # Run the on each individual image
    indiv_errs = []
    for tim in tims:
        tr = tractor.Tractor([tim], [src])
        src.brightness.setParams([0.])
        tr.freezeParam('images')
        phot = tr.optimize_forced_photometry(variance=True)
        indiv_err = 1./np.sqrt(phot.IV[0])
        indiv_errs.append(indiv_err)
        print('Forced-phot of individual image =', src.getBrightness().getValue(), '+-', indiv_err)

    # Compute the Single-Frame Average error estimate -- these are
    # Gaussian measurements, so the mean is inverse-variance-weighted,
    # and the inverse-variance is summed.
    averaging_err = 1./np.sqrt(np.sum(1./np.array(indiv_errs)**2))
        
    # Compute a coadd (with different coadd weights) and run forced phot on that.
    # Search this grid of alpha values
    coadd_amps = np.linspace(0.0, 1.0, 101)
    coadd_errs = np.zeros(len(coadd_amps))
    for iamp,amp0 in enumerate(coadd_amps):
        assert(len(tims) == 2)
        amp1 = 1. - amp0
        amps = [amp0, amp1]
        # Compute the coadded image, and its PSF model.
        coadd = np.zeros(tims[0].shape, np.float32)
        coadd_var = np.zeros_like(coadd)
        coadd_psf_sigma = []
        coadd_psf_weight = []
        for tim,amp in zip(tims, amps):
            # if Y = a X and X ~ N( mu, sigma^2), then
            # Y ~ N( a mu, a^2 sigma^2)
            # and if Z = b W and W ~ N( mu_w, sigma_w^2), then
            # Z ~ N( b mu_w, b^2 sigma_w^2 )
            # and Y + Z ~ N(a mu + b mu_w, a^2 sigma^2 + b^2 sigma_w^2)
            coadd += amp * tim.data
            coadd_var += amp**2 * 1./tim.getInvvar()
            coadd_psf_sigma.extend(tim.psf.sigmas)
            coadd_psf_weight.extend([w * amp for w in tim.psf.weights])

        # This is the tractor Image object for the coadd
        tim = tractor.Image(
            data=coadd, invvar=1./coadd_var,
            psf=tractor.NCircularGaussianPSF(coadd_psf_sigma,coadd_psf_weight),
            photocal=tractor.LinearPhotoCal(1.))
        # Run force photometry on the coadd.
        tr = tractor.Tractor([tim], [src])
        src.brightness.setParams([0.])
        tr.freezeParam('images')
        phot = tr.optimize_forced_photometry(variance=True)
        coadd_err = 1./np.sqrt(phot.IV[0])
        print('Forced-phot of coadd with amp:', amp0, '=',
              src.getBrightness().getValue(), '+-', coadd_err)
        coadd_errs[iamp] = coadd_err

    # Find the coadd amplitude (alpha) that yields the smallest error
    i = np.argmin(coadd_errs)
    bestamp = coadd_amps[i]
    print('Best coadd amplitude:', bestamp)
    print('Predicted error at best coadd amplitude:', coadd_errs[i])
    print('vs single-frame averaging:', averaging_err)
    print('Factor:', coadd_errs[i] / averaging_err)

    # plot styles
    stys = [dict(color='b', linestyle='dashed'),
            dict(color='b', linestyle='dotted'),]
    
    plt.clf()
    plt.xlabel('Coadd weight $\\alpha$')
    plt.ylabel('Error in photometry')
    for i,(err,(noise,seeing),sty) in enumerate(zip(indiv_errs, imageset, stys)):
        plt.axhline(err, label=('Image %s: noise %.1f, seeing %.1f' %
                                (chr(ord('A')+i), noise,seeing)),
                    zorder=18, **sty)
    plt.plot(coadd_amps, coadd_errs, 'g-', label='Method C: Coadd',
             zorder=22, lw=3, alpha=0.5)
    plt.axhline(averaging_err, color='k', lw=5, alpha=0.3,
                label='Method B: Single-frame average', zorder=20)
    plt.axhline(simult_err, color='r', label='Method A: Simultaneous fitting',
                zorder=21)
    l = plt.legend(loc='center right')
    l.set_zorder(30)
    plt.xlim(0,1)
    ps.savefig()


    # Now we run the noise-adding simulations.
    
    fluxes_simult = []
    fluxes_averaged = []
    fluxes_coadd = []

    orig_data = [tim.data for tim in tims]

    all_fluxes = np.zeros((Nruns, len(tims)))
    
    print('Simulating', Nruns, 'times...')
    for irun in range(Nruns):
        # Add a new random noise draw to each image
        for data,tim in zip(orig_data, tims):
            tim.data = data + np.random.normal(size=tim.shape) / tim.getInvError()

        # Simultaneous fitting
        tr = tractor.Tractor(tims, [src])
        tr.freezeParam('images')
        src.brightness.setParams([0.])
        phot = tr.optimize_forced_photometry()
        fluxes_simult.append(src.getBrightness().getValue())

        # Fit individual exposures
        fluxes = np.zeros(len(tims))
        fluxes_iv = np.zeros_like(fluxes)
        for itim,tim in enumerate(tims):
            tr = tractor.Tractor([tim], [src])
            tr.freezeParam('images')
            src.brightness.setParams([0.])
            phot = tr.optimize_forced_photometry(variance=True)
            f = src.getBrightness().getValue()
            fluxes[itim] = f
            fluxes_iv[itim] = phot.IV[0]
            all_fluxes[irun,itim] = f

        # Weighted sum of individual exposures (weighted by inverse-variance)
        fluxes_averaged.append((np.sum(fluxes * fluxes_iv) / np.sum(fluxes_iv)))

        # Coadd using best amplitude found above
        coadd = np.zeros(tims[0].shape, np.float32)
        coadd_var = np.zeros_like(coadd)
        coadd_psf_sigma = []
        coadd_psf_weight = []
        amp0 = bestamp
        amp1 = 1. - amp0
        amps = [amp0, amp1]
        assert(len(tims) == 2)
        for tim,amp in zip(tims, amps):
            coadd += amp * tim.data
            coadd_var += amp**2 * 1./tim.getInvvar()
            coadd_psf_sigma.extend(tim.psf.sigmas)
            coadd_psf_weight.extend([w * amp for w in tim.psf.weights])
        tim = tractor.Image(
            data=coadd, invvar=1./coadd_var,
            psf=tractor.NCircularGaussianPSF(coadd_psf_sigma,coadd_psf_weight),
            photocal=tractor.LinearPhotoCal(1.))
        if irun == 0:
            print('Using coadd amplitudes:', amps)
            print('PSF sigmas :', tim.psf.sigmas)
            print('PSF weights:', tim.psf.weights)
        tr = tractor.Tractor([tim], [src])
        tr.freezeParam('images')
        src.brightness.setParams([0.])
        phot = tr.optimize_forced_photometry()
        fluxes_coadd.append(src.getBrightness().getValue())

    # Histogram the simulation results.
    plt.clf()
    ha = dict(histtype='step', range=(80, 120), bins=40, normed=True)

    f = all_fluxes[:,0]
    ha2 = ha.copy()
    ha2.update(stys[0])
    plt.hist(f, label='Image A: std %.2f, S/N %.2f' % (np.std(f), trueflux / np.std(f)), **ha2)
    f = all_fluxes[:,1]
    ha2 = ha.copy()
    ha2.update(stys[1])
    plt.hist(f, label='Image B: std %.2f, S/N %.2f' % (np.std(f), trueflux / np.std(f)), **ha2)
    
    plt.hist(fluxes_simult, color='r',
             label=('[A] Simultaneous fitting: std %.2f, S/N %.2f' %
                    (np.std(fluxes_simult), trueflux / np.std(fluxes_simult))), lw=3, **ha)
    plt.hist(fluxes_averaged, color='k',
             label=('[B] Single-frame average: std %.2f, S/N %.2f' %
                    (np.std(fluxes_averaged), trueflux / np.std(fluxes_averaged))), **ha)
    plt.hist(fluxes_coadd, color='g',
             label=('[C] Coadded: std %.2f, S/N %.2f' %
                    (np.std(fluxes_coadd), trueflux / np.std(fluxes_coadd))), **ha)
    plt.xlabel('Measured flux')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    yl,yh = plt.ylim()
    plt.ylim(yl, yh*1.5)
    plt.xlim(80,120)
    ps.savefig()
    
