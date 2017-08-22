from __future__ import print_function
import numpy as np
import pylab as plt
from astrometry.util.plotutils import PlotSequence
import tractor

ps = PlotSequence('coadd')

# Case 1: coadd two images with same noise and PSF -- same performance as individual exposures
# Case 2: different noise, same PSF
# Case 3: same noise, different PSF
# Case 4: different noise, different PSF, but same S/N

# For each, scan through a range of coadd amplitudes
# In all cases, use the true PSF model and no actual noise first, and then simulate
# a bunch of noisy runs.

H,W = 50,50
trueflux = 100.

for name,imageset in [
        ('Case 1: Same noise and PSF', [(1., 2.), (1., 2.)]),
        ('Case 2: Different noise, same PSF', [(1., 2.), (2., 2.)]),
        ('Case 3: Same noise, different PSF', [(1., 2.), (1., 4.)]),
        ('Case 4: Different noise, different PSF', [(1., 2.), (0.5, 4.)]),
        ]:

    print()
    print(name)
    print()

    src = tractor.PointSource(tractor.PixPos(W//2, H//2),
                              tractor.Flux(trueflux))

    tims = []
    for noise, psf_size in imageset:
        tim = tractor.Image(np.zeros((H,W), np.float32),
                            inverr=np.ones((H,W), np.float32) * (1./noise),
                            psf=tractor.NCircularGaussianPSF([psf_size], [1.]),
                            photocal=tractor.LinearPhotoCal(1.))
        # Create noiseless model image.
        tr = tractor.Tractor([tim], [src])
        mod = tr.getModelImage(0)
        tim.data = mod
        tims.append(tim)

    # Run forced photometry on the set of individual images and record the flux and errors.
    tr = tractor.Tractor(tims, [src])
    src.brightness.setParams([0.])
    src.freezeParam('pos')
    #print('Before fitting: source:', src)
    tr.freezeParam('images')
    
    phot = tr.optimize_forced_photometry(variance=True)
    #print('Forced-photometry of individual images result:', phot)
    #print('Inverse-variance:', phot.IV)
    #print('Source:', src)
    print('Forced-phot of individual images simultaneously =', src.getBrightness().getValue(), '+-', 1./np.sqrt(phot.IV[0]))

    simult_err = 1./np.sqrt(phot.IV[0])
    
    # Run forced photometry on each individual image and record the flux and errors.
    indiv_errs = []
    for tim in tims:
        tr = tractor.Tractor([tim], [src])
        src.brightness.setParams([0.])
        tr.freezeParam('images')
        #print('Before fitting: source:', src)
    
        phot = tr.optimize_forced_photometry(variance=True)
        #print('Forced-photometry of individual image result:', phot)
        #print('Inverse-variance:', phot.IV)
        #print('Source:', src)
        indiv_errs.append(1./np.sqrt(phot.IV[0]))
        print('Forced-phot of individual image =', src.getBrightness().getValue(), '+-', 1./np.sqrt(phot.IV[0]))
        
    total_indiv_errs = 1./np.sqrt(np.sum(1./np.array(indiv_errs)**2))
        
    # Compute a coadd (with different coadd weights) and run forced phot on that.
    ivs = []
    coadd_amps = np.linspace(0.0, 1.0, 101)
    for amp0 in coadd_amps:
        assert(len(tims) == 2)
        amp1 = 1. - amp0
        amps = [amp0, amp1]
        #print('Coadd amplitudes:', amps)
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

        tim = tractor.Image(
            data=coadd, invvar=1./coadd_var,
            psf=tractor.NCircularGaussianPSF(coadd_psf_sigma,coadd_psf_weight),
            photocal=tractor.LinearPhotoCal(1.))

        #print('PSF sigmas:',  tim.psf.sigmas.getParams())
        #print('PSF weights:', tim.psf.weights.getParams())

        tr = tractor.Tractor([tim], [src])
        src.brightness.setParams([0.])
        tr.freezeParam('images')
        #print('Before fitting: source:', src)
        phot = tr.optimize_forced_photometry(variance=True)
        #print('Forced-photometry of coadd result:', phot)
        #print('Inverse-variance:', phot.IV)
        #print('Source:', src)
        print('Forced-phot of coadd with amp:', amp0, '=', src.getBrightness().getValue(), '+-', 1./np.sqrt(phot.IV[0]))
        flux_iv = phot.IV[0]
        ivs.append(flux_iv)

    errors = 1. / np.sqrt(np.array(ivs))

    i = np.argmin(errors)
    bestamp = coadd_amps[i]
    print('Best coadd amplitude:', bestamp)

    plt.clf()
    plt.xlabel('Coadd weight')
    plt.ylabel('Forced-photometry uncertainty')
    for err,(noise,seeing) in zip(indiv_errs, imageset):
        plt.axhline(err, label=('Individual exposure: noise %.1f, seeing %.1f' %
                                (noise,seeing)),
                    zorder=18)
    plt.plot(coadd_amps, errors, 'g-', label='Coadd',
             zorder=22)
    plt.axhline(total_indiv_errs, color='k', lw=5, alpha=0.3,
                label='Weighted sum of individual measurements',
                zorder=20,
        )
    plt.axhline(simult_err, color='r', label='Simultaneous fitting',
                zorder=21)
    l = plt.legend(loc='center right')
    l.set_zorder(30)
    plt.title(name)
    plt.xlim(0,1)
    ps.savefig()
    

    Nruns = 10000

    fluxes_simult = []
    fluxes_indiv = []
    fluxes_summed = []
    fluxes_coadd = []

    orig_data = [tim.data for tim in tims]

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

        # Fitting individual exposures
        fluxes = np.zeros(len(tims))
        fluxes_iv = np.zeros_like(fluxes)
        for itim,tim in enumerate(tims):
            tr = tractor.Tractor([tim], [src])
            tr.freezeParam('images')
            src.brightness.setParams([0.])
            phot = tr.optimize_forced_photometry(variance=True)
            fluxes[itim] = src.getBrightness().getValue()
            fluxes_iv[itim] = phot.IV[0]

        fluxes_indiv.extend(list(fluxes))

        # Weighted sum of individual exposures (weighted by inverse-variance)
        fluxes_summed.append((np.sum(fluxes * fluxes_iv) / np.sum(fluxes_iv)))

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

    plt.clf()
    ha = dict(histtype='step', range=(80, 120), bins=25, normed=True)

    plt.hist(fluxes_indiv, color='b',
             label=('Individual exposures: std %.2f, S/N %.2f' % 
                    (np.std(fluxes_indiv), trueflux / np.std(fluxes_indiv))), **ha)
    plt.hist(fluxes_simult, color='r',
             label=('Simultaneous fitting: std %.2f, S/N %.2f' %
                    (np.std(fluxes_simult), trueflux / np.std(fluxes_simult))), lw=3, **ha)
    plt.hist(fluxes_summed, color='k',
             label=('Weighted sum of individual exposures: std %.2f, S/N %.2f' %
                    (np.std(fluxes_summed), trueflux / np.std(fluxes_summed))), **ha)
    plt.hist(fluxes_coadd, color='g',
             label=('Coadded exposures: std %.2f, S/N %.2f' %
                    (np.std(fluxes_coadd), trueflux / np.std(fluxes_coadd))), **ha)
    plt.xlabel('Measured flux')
    plt.title(name)
    plt.legend(loc='upper right')
    yl,yh = plt.ylim()
    plt.ylim(yl, yh*1.4)
    plt.xlim(80,120)
    ps.savefig()
    
