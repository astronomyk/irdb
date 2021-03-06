'''The purpose of this notebook is to simulate a simple L or M band
imaging observation with METIS. We demonstrate how to change
observational parameters.'''

import numpy as np

from astropy import units as u
from astropy.table import Table
from astropy.io import fits
from photutils import CircularAperture, aperture_photometry
from matplotlib import pyplot as plt

import scopesim
import scopesim_templates as sim_tp

# Set the path to the local irdb.
from scopesim import rc
rc.__currsys__['!SIM.file.local_packages_path'] = \
    "../../"


def simulate_point_source(plot=False):
    '''Create a simulation of a point source'''

    print("------ Beginning of simulate_point_source() ----------")
    # Create the Source object, this is currently a hack. We create a
    # spectrum that is flat in lambda with magnitude 0 in the Lp
    # filter. For this purpose we are currently using a function in
    # SimMETIS, pending its inclusion in ScopeSim-Templates.

    # import simmetis
    # dummycmd = simmetis.UserCommands("metis_image_LM.config",
    #                                  sim_data_dir="../data")
    # dummycmd["INST_FILTER_TC"] = "TC_filter_Lp.dat"
    #
    # lam, spec = simmetis.source.flat_spectrum(0,
    #                                           dummycmd["INST_FILTER_TC"])

    lam = np.linspace(1, 20, 0.001)
    spec = np.ones(len(lam)) * 1e7      # 0 mag spectrum

    if plot:
        plt.plot(lam, spec)
        plt.xlabel(r"$\lambda$ [um]")
        plt.ylabel("relative flux")
        plt.title("Spectrum of input source")
        plt.show()

    # Create two source objects for two dither positions
    dither_offset = 1
    src = scopesim.Source(lam=lam * u.um, spectra=np.array([spec]),
                          ref=[0], x=[0], y=[0])
    src_dither = scopesim.Source(lam=lam * u.um, spectra=np.array([spec]),
                                 ref=[0], x=[0], y=[dither_offset])

    # Load the configuration for the METIS LM-band imaging mode.
    cmd = scopesim.UserCommands(use_instrument="METIS",
                                set_modes=["img_lm"])

    # build the optical train and adjust
    metis = scopesim.OpticalTrain(cmd)
    metis['scope_vibration'].include = False
    metis['detector_linearity'].include = False

    # Set the DIT to 1 second
    metis.cmds["!OBS.dit"] = 1.

    # Perform an observation.
    metis.observe(src, update=True)
    hdus = metis.readout()

    metis.observe(src_dither, update=True)
    hdus_dither = metis.readout()

    frame1 = hdus[0][1].data
    frame2 = hdus_dither[0][1].data
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6),
                                       sharey=True)
        f1_plot = ax1.imshow(frame1[900:1400, 940:1100], origin='lower')
        fig.colorbar(f1_plot, ax=ax1)

        f2_plot = ax2.imshow(frame2[900:1400, 940:1100], origin='lower')
        fig.colorbar(f2_plot, ax=ax2)

        plt.show()

    # Shift the dithered image back to the original position and combine
    # the two images
    pixscale = metis.cmds['INST']['pixel_scale']
    frame_sum = frame1 + np.roll(frame2,
                                 np.int(np.round(-dither_offset / pixscale)),
                                 axis=0)
    if plot:
        plt.imshow(frame_sum[940:1100, 940:1100])
        plt.colorbar()
        plt.show()

    print("Writing test_LM_framesum.fits")
    fits.writeto("test_LM_framesum.fits", frame_sum, overwrite=True)

    print("--------- End of simulate_point_source() -------------")


def vary_exposure_times(plot=False):
    '''Adjusting exposure times'''
    print("------ Beginning of vary_exposure_times() ----------")

    # Create a new source
    # import simmetis
    # dummycmd = simmetis.UserCommands("metis_image_LM.config",
    #                                  sim_data_dir="../data")
    # lam, spec = simmetis.source.flat_spectrum(17, "TC_filter_Lp.dat")

    lam = np.arange(1, 20, 0.001)
    spec = np.ones(len(lam)) * 1e7 * 2.5*np.log10(-30)     # 0 mag spectrum

    src = scopesim.Source(lam=lam * u.um, spectra=np.array([spec]),
                          ref=[0], x=[0], y=[0])

    # Load the configuration for the METIS LM-band imaging mode.
    cmd = scopesim.UserCommands(use_instrument="METIS",
                                set_modes=["img_lm"])

    # build the optical train and adjust
    metis = scopesim.OpticalTrain(cmd)
    metis['scope_vibration'].include = False
    metis['detector_linearity'].include = False

    # Observe the source
    metis.observe(src, update=True)

    # Readout with a range of DITs and NDITs
    dit = np.array([1, 1, 1, 10, 10, 10, 100, 100, 100])
    ndit = np.array([3, 10, 30, 3, 10, 30, 3, 10, 30])
    hdus = list()
    for i in range(9):
        metis.cmds["!OBS.dit"] = dit[i]
        metis.cmds["!OBS.ndit"] = ndit[i]
        thehdu = metis.readout()[0]
        hdus.append(thehdu)
        print(i, "DIT =", rc.__currsys__["!OBS.dit"],
              "   NDIT =", rc.__currsys__["!OBS.ndit"])
        print("min =", thehdu[1].data.min(), "    max =", thehdu[1].data.max())

    if plot:
        fig, axes = plt.subplots(3, 3, figsize=(12, 12),
                                 sharex=True, sharey=True)
        for i in range(9):
            theax = axes.flat[i]
            frame = hdus[i][1].data[960:1090, 960:1090]
            theplot = theax.imshow(frame, origin='lower')
            fig.colorbar(theplot, ax=theax)
            theax.set_title("DIT={}, NDIT={}, INTTIME={}".format(
                dit[i], ndit[i], dit[i] * ndit[i]))

        plt.show()


    # Perform photometry
    aperture = CircularAperture([(1024., 1024.)], r=10.)

    bglevel = np.zeros(9)
    bgnoise = np.zeros(9)
    starsum = np.zeros(9)
    starnoise = np.zeros(9)

    for i, thehdu in enumerate(hdus):
        # background stats
        bglevel[i] = np.mean(thehdu[1].data[0:800, 0:800])
        bgnoise[i] = np.std(thehdu[1].data[0:800, 0:800])
        # total signal of star
        phot_table = aperture_photometry(thehdu[1].data - bglevel[i], aperture,
                                         error=(np.ones_like(thehdu[1].data)
                                                * bgnoise[i]))
        starsum[i] = phot_table['aperture_sum'][0]
        starnoise[i] = phot_table['aperture_sum_err'][0]

    table = Table([dit, ndit, dit * ndit, bglevel, bgnoise, starsum,
                   starsum / starnoise],
                  names=["DIT", "NDIT", "INTTIME", "bg level", "bg noise",
                         "Star counts", "S/N"])
    table["bg level"].format = ".0f"
    table["bg noise"].format = ".1f"
    table["Star counts"].format = ".1f"
    table["S/N"].format = ".2f"

    table.pprint()

    if plot:
        plt.plot(table["INTTIME"], table["S/N"], "o")
        plt.xlabel("Integration time")
        plt.ylabel("Signal-to-noise ratio")
        plt.show()

    print("--------- vary_exposure_times() -------------")
